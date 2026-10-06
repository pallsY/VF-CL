"""FedProTIP task prediction adapted to four-party vertical FL."""
from copy import deepcopy

import torch

from data_utils import split_features
from metrics import select_formal_cached_batches
from .gpm import GPMCL, _stable_svd


TIP_THRESHOLD = 0.775
TIP_MAX_BATCHES = 20
EPS = 1e-12
FORMAL_EVALUATION_STATE_KEYS = frozenset({
    "task_means", "task_bases", "task_classes",
    "tip_threshold", "max_batches",
})


def _require_finite(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def energy_basis(centered, threshold):
    if centered.ndim != 2:
        raise ValueError("centered embeddings must be two-dimensional")
    _require_finite("centered embeddings", centered)
    _, singular, vh = _stable_svd(centered.float())
    energy = singular.square()
    total = energy.sum()
    if total <= 0:
        raise ValueError("centered embeddings have zero energy")
    rank = int((energy.cumsum(0) / total < threshold).sum().item()) + 1
    return vh[:rank].t().contiguous().cpu()


def normalized_relevance(embeddings, mean, basis):
    for name, value in (("embeddings", embeddings), ("mean", mean), ("basis", basis)):
        _require_finite(name, value)
    centered = embeddings - mean
    projected = (centered @ basis) @ basis.t()
    return projected.norm(dim=1) / (centered.norm(dim=1) + EPS)


def vote_tasks(party_scores, task_ids):
    if party_scores.ndim != 3:
        raise ValueError("party_scores must have shape parties x samples x tasks")
    if party_scores.size(2) != len(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must uniquely match the score columns")
    _require_finite("party_scores", party_scores)
    party_votes = party_scores.argmax(dim=2)
    predictions = []
    for sample in range(party_scores.size(1)):
        counts = torch.bincount(party_votes[:, sample], minlength=len(task_ids))
        candidates = torch.nonzero(counts == counts.max(), as_tuple=False).flatten()
        if candidates.numel() > 1:
            means = party_scores[:, sample, candidates].mean(dim=0)
            candidates = candidates[means == means.max()]
        predictions.append(min(task_ids[int(index)] for index in candidates))
    return torch.tensor(predictions, dtype=torch.long, device=party_scores.device)


def masked_predictions(logits, predicted_tasks, task_classes):
    if logits.ndim != 2 or predicted_tasks.ndim != 1 or logits.size(0) != predicted_tasks.size(0):
        raise ValueError("logits and predicted_tasks batch dimensions must match")
    _require_finite("logits", logits)
    predictions = torch.empty_like(predicted_tasks)
    for task_id in predicted_tasks.unique().tolist():
        if int(task_id) not in task_classes:
            raise ValueError(f"missing classes for predicted task {task_id}")
        classes = list(task_classes[int(task_id)])
        if not classes or len(set(classes)) != len(classes):
            raise ValueError(f"invalid classes for task {task_id}")
        rows = predicted_tasks == int(task_id)
        local = logits[rows][:, classes].argmax(dim=1)
        predictions[rows] = torch.tensor(classes, device=logits.device)[local]
    return predictions


class FedProTIPVFLCL(GPMCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = "FedProTIP_VFL"
        self.tip_threshold = float(getattr(
            args, "fedprotip_tip_threshold", TIP_THRESHOLD
        ))
        self.threshold = self.tip_threshold
        self.max_batches = int(getattr(
            args, "fedprotip_max_batches", TIP_MAX_BATCHES
        ))
        self.task_means = [{} for _ in range(args.num_parties)]
        self.task_bases = [{} for _ in range(args.num_parties)]
        self.task_classes = {}
        self._pending_task_classes = {}

    def before_task(self, task_id, new_classes, seen_classes):
        super().before_task(task_id, new_classes, seen_classes)
        classes = [int(value) for value in new_classes]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError(f"task {task_id} has invalid classes")
        self._pending_task_classes[int(task_id)] = classes

    def _update_task_reference(self, party_embeddings, task_id, task_classes):
        task_id = int(task_id)
        classes = [int(value) for value in task_classes]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError(f"task {task_id} has invalid classes")
        occupied = {
            value
            for key, values in self.task_classes.items()
            if key != task_id
            for value in values
        }
        if occupied.intersection(classes):
            raise ValueError(f"task {task_id} reuses an existing class")
        if len(party_embeddings) != self.args.num_parties:
            raise ValueError("party embedding count does not match num_parties")
        for party_id, embeddings in enumerate(party_embeddings):
            if embeddings.ndim != 2 or embeddings.size(0) < 2:
                raise ValueError("each task reference needs at least two embeddings")
            _require_finite("party embeddings", embeddings)
            mean = embeddings.float().mean(dim=0)
            basis = energy_basis(embeddings.float() - mean, self.tip_threshold)
            self.task_means[party_id][task_id] = mean.cpu()
            self.task_bases[party_id][task_id] = basis.cpu()
        self.task_classes[task_id] = classes

    @torch.no_grad()
    def _collect_party_embeddings(self, loader, max_batches=None):
        for bottom in self.trainer.bottoms:
            bottom.eval()
        rows = [[] for _ in range(self.args.num_parties)]
        labels = []
        for batch_id, (batch_x, batch_y) in enumerate(loader):
            if max_batches is not None and batch_id >= max_batches:
                break
            parts = split_features(batch_x.to(self.args.device), self.args)
            for party_id, bottom in enumerate(self.trainer.bottoms):
                rows[party_id].append(bottom(parts[party_id]).cpu())
            labels.append(batch_y.cpu())
        if not labels:
            raise ValueError("cannot build or evaluate an empty loader")
        return [torch.cat(values) for values in rows], torch.cat(labels)

    def after_task(self, train_loader, task_id):
        super().after_task(train_loader, task_id)
        party_embeddings, _ = self._collect_party_embeddings(
            train_loader, max_batches=self.max_batches
        )
        if int(task_id) not in self._pending_task_classes:
            raise ValueError(f"missing class metadata for task {task_id}")
        self._update_task_reference(
            party_embeddings,
            task_id,
            self._pending_task_classes.pop(int(task_id)),
        )

    def _validated_task_ids(self):
        task_ids = sorted(self.task_classes)
        if not task_ids:
            raise ValueError("no task references are available")
        for party_id in range(self.args.num_parties):
            if sorted(self.task_means[party_id]) != task_ids:
                raise ValueError(f"party {party_id} has incomplete task means")
            if sorted(self.task_bases[party_id]) != task_ids:
                raise ValueError(f"party {party_id} has incomplete task bases")
        return task_ids

    def predict_tasks(self, party_embeddings):
        task_ids = self._validated_task_ids()
        if len(party_embeddings) != self.args.num_parties:
            raise ValueError("party embedding count does not match num_parties")
        party_scores = []
        for party_id, embeddings in enumerate(party_embeddings):
            scores = [
                normalized_relevance(
                    embeddings,
                    self.task_means[party_id][task_id].to(embeddings.device),
                    self.task_bases[party_id][task_id].to(embeddings.device),
                )
                for task_id in task_ids
            ]
            party_scores.append(torch.stack(scores, dim=1))
        return vote_tasks(torch.stack(party_scores), task_ids)

    def _normalized_evaluation_classes(self, seen_task_classes):
        normalized = {
            int(key): [int(value) for value in values]
            for key, values in seen_task_classes.items()
        }
        if normalized != {
            key: self.task_classes[key] for key in sorted(normalized)
        }:
            raise ValueError("evaluation task mapping differs from saved references")
        return normalized

    def _evaluate_class_il_batches(self, normalized, batches_for_task):
        result = {
            name: {}
            for name in (
                "class_il_pred_task",
                "class_il_global",
                "task_il_oracle",
                "task_prediction",
                "class_il_pred_task_counts",
            )
        }
        self.trainer.top_model.eval()
        for task_id, classes in normalized.items():
            party_embeddings, labels = self._collect_party_embeddings(
                batches_for_task(task_id, classes)
            )
            predicted_tasks = self.predict_tasks(party_embeddings)
            aggregated = self.trainer._aggregate(
                [value.to(self.args.device) for value in party_embeddings]
            )
            logits = self.trainer.top_model(aggregated).cpu()
            predicted = masked_predictions(
                logits, predicted_tasks.cpu(), normalized
            )
            oracle_tasks = torch.full_like(labels, task_id)
            oracle = masked_predictions(logits, oracle_tasks, normalized)
            key = f"task_{task_id}"
            correct = int((predicted == labels).sum().item())
            total = int(labels.numel())
            result["class_il_pred_task"][key] = round(
                correct / total, 4
            )
            result["class_il_global"][key] = round(
                float((logits.argmax(1) == labels).float().mean()), 4
            )
            result["task_il_oracle"][key] = round(
                float((oracle == labels).float().mean()), 4
            )
            result["task_prediction"][key] = round(
                float((predicted_tasks.cpu() == task_id).float().mean()), 4
            )
            result["class_il_pred_task_counts"][key] = {
                "correct": correct,
                "total": total,
            }
        return result

    @torch.no_grad()
    def evaluate_class_il_readouts(self, dataset, seen_task_classes):
        normalized = self._normalized_evaluation_classes(seen_task_classes)

        def task_loader(_task_id, classes):
            _, loader = dataset.get_task_loaders(classes, shuffle_train=False)
            return loader

        return self._evaluate_class_il_batches(normalized, task_loader)

    @torch.no_grad()
    def evaluate_class_il_readouts_cached(
            self, cached_test_batches, seen_task_classes):
        normalized = self._normalized_evaluation_classes(seen_task_classes)
        return self._evaluate_class_il_batches(
            normalized,
            lambda _task_id, classes: select_formal_cached_batches(
                cached_test_batches, classes
            ),
        )

    def _validate_formal_evaluation_state(self, state):
        if type(state) is not dict or set(state) != FORMAL_EVALUATION_STATE_KEYS:
            raise ValueError("FedProTIP formal evaluation state schema is invalid")
        if (type(state["tip_threshold"]) is not float
                or state["tip_threshold"] != self.tip_threshold):
            raise ValueError("checkpoint uses a different FedProTIP tip_threshold")
        if (type(state["max_batches"]) is not int
                or state["max_batches"] != self.max_batches):
            raise ValueError("checkpoint uses a different FedProTIP max_batches")
        classes = state["task_classes"]
        means, bases = state["task_means"], state["task_bases"]
        if (type(classes) is not dict or not classes
                or type(means) is not list or len(means) != self.args.num_parties
                or type(bases) is not list or len(bases) != self.args.num_parties):
            raise ValueError("FedProTIP formal reference state is incomplete")
        task_ids = sorted(classes)
        occupied = []
        for task_id in task_ids:
            values = classes[task_id]
            if (type(task_id) is not int or type(values) is not list or not values
                    or any(type(value) is not int for value in values)
                    or len(values) != len(set(values))):
                raise ValueError("FedProTIP formal task classes are invalid")
            occupied.extend(values)
        if len(occupied) != len(set(occupied)):
            raise ValueError("FedProTIP formal task classes overlap")
        for party_id, (party_means, party_bases) in enumerate(zip(means, bases)):
            if (type(party_means) is not dict or sorted(party_means) != task_ids
                    or type(party_bases) is not dict
                    or sorted(party_bases) != task_ids):
                raise ValueError(
                    f"FedProTIP party {party_id} formal references are incomplete"
                )
            for task_id in task_ids:
                mean, basis = party_means[task_id], party_bases[task_id]
                if (not isinstance(mean, torch.Tensor) or mean.ndim != 1
                        or not torch.is_floating_point(mean)
                        or mean.device.type != "cpu" or mean.requires_grad
                        or not isinstance(basis, torch.Tensor) or basis.ndim != 2
                        or not torch.is_floating_point(basis)
                        or basis.device.type != "cpu" or basis.requires_grad
                        or basis.shape[0] != mean.numel() or basis.shape[1] <= 0):
                    raise ValueError("FedProTIP formal reference tensor is invalid")
                _require_finite("FedProTIP formal reference tensor", mean)
                _require_finite("FedProTIP formal reference tensor", basis)
        return {
            "task_means": deepcopy(means),
            "task_bases": deepcopy(bases),
            "task_classes": deepcopy(classes),
            "tip_threshold": state["tip_threshold"],
            "max_batches": state["max_batches"],
        }

    def get_formal_evaluation_state(self):
        self._validated_task_ids()
        return self._validate_formal_evaluation_state({
            "task_means": self.task_means,
            "task_bases": self.task_bases,
            "task_classes": self.task_classes,
            "tip_threshold": self.tip_threshold,
            "max_batches": self.max_batches,
        })

    def load_formal_evaluation_state(self, state):
        validated = self._validate_formal_evaluation_state(state)
        self.task_means = validated["task_means"]
        self.task_bases = validated["task_bases"]
        self.task_classes = validated["task_classes"]
        self.threshold = self.tip_threshold
        self._pending_task_classes = {}
        self._validated_task_ids()

    def get_state(self):
        state = super().get_state()
        state.update({
            "tip_threshold": self.tip_threshold,
            "max_batches": self.max_batches,
            "task_means": deepcopy(self.task_means),
            "task_bases": deepcopy(self.task_bases),
            "task_classes": deepcopy(self.task_classes),
        })
        return state

    def load_state(self, state):
        if ("max_batches" not in state
                and not bool(getattr(
                    self.args, "formal_deferred_evaluation", False))):
            state = dict(state, max_batches=TIP_MAX_BATCHES)
        for field, expected in (
                ("tip_threshold", self.tip_threshold),
                ("max_batches", self.max_batches)):
            if field not in state:
                raise ValueError(f"checkpoint is missing FedProTIP {field}")
            if state[field] != expected:
                raise ValueError(f"checkpoint uses a different FedProTIP {field}")
        if not FORMAL_EVALUATION_STATE_KEYS.issubset(state):
            raise ValueError("checkpoint FedProTIP reference state is incomplete")
        formal = self._validate_formal_evaluation_state({
            key: state[key] for key in FORMAL_EVALUATION_STATE_KEYS
        })
        super().load_state(state)
        self.load_formal_evaluation_state(formal)
