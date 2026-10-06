"""V-LETO (AAAI 2025): sum aggregation, single linear top, FIM freeze, prototype evolution."""
import json, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from copy import deepcopy
from data_utils import split_features
from determinism import derive_seed
from adaptive_head_consolidation import (
    ADAPTIVE_METHOD_VERSION,
    BIAS_BRANCH_CONFIG,
    FULL_BRANCH_CONFIG,
    AdaptiveConsolidationResult,
    adaptive_candidate_log_probabilities,
    build_adaptive_diagnostics,
    fit_adaptive_candidates,
    install_and_reload_verify,
    solve_global_mixture_weight,
)
from head_consolidation import (
    consolidate_classifier, consolidate_task_class_bias, freeze_state,
    hash_top_state,
)
from party_weight_manifest import build_task_derangement, load_manifest


DISTILL_WEIGHT_SCHEDULES = {
    'constant': (),
    'stage_f_decay_010': ((1, 4, 0.25), (5, 7, 0.15), (8, 9, 0.10)),
    'stage_f_decay_005': ((1, 4, 0.25), (5, 7, 0.15), (8, 9, 0.05)),
}


def effective_distill_weight(base_weight, schedule, task_id):
    """Return the task-local logit-KD weight for a named schedule."""
    if schedule not in DISTILL_WEIGHT_SCHEDULES:
        raise ValueError(f'unknown distillation schedule: {schedule}')
    task_id = int(task_id)
    if task_id < 0:
        raise ValueError('task_id must be non-negative')
    if schedule == 'constant' or task_id == 0:
        return float(base_weight)
    for first, last, weight in DISTILL_WEIGHT_SCHEDULES[schedule]:
        if first <= task_id <= last:
            return float(weight)
    raise ValueError(f'{schedule} does not define task {task_id}')


def effective_supcon_weight(weight, start_task, task_id):
    """Enable current-task supervised contrastive learning from start_task."""
    weight = float(weight)
    start_task = int(start_task)
    task_id = int(task_id)
    if weight < 0 or start_task < 0 or task_id < 0:
        raise ValueError('supervised contrastive settings must be non-negative')
    return weight if task_id >= start_task else 0.0


def head_consolidation_due(schedule, task_id, num_tasks):
    """Return whether the fixed consolidation schedule fires at this boundary."""
    task_id = int(task_id)
    num_tasks = int(num_tasks)
    if num_tasks <= 0 or task_id < 0 or task_id >= num_tasks:
        raise ValueError('invalid task boundary for head consolidation')
    if schedule == 'every':
        return True
    if schedule == 'final':
        return task_id == num_tasks - 1
    raise ValueError(f'unknown head consolidation schedule: {schedule}')


def herding_indices(embeddings, capacity):
    """Select a fixed-size set whose normalized mean approximates the class mean."""
    if embeddings.ndim != 2 or embeddings.size(0) == 0:
        raise ValueError('herding embeddings must be a non-empty matrix')
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError('herding capacity must be positive')
    capacity = min(capacity, embeddings.size(0))
    features = F.normalize(embeddings.float(), dim=1)
    target = features.mean(dim=0)
    running = torch.zeros_like(target)
    selected = []
    available = torch.ones(features.size(0), dtype=torch.bool)
    for count in range(capacity):
        candidate_means = (running.unsqueeze(0) + features) / (count + 1)
        distances = (candidate_means - target.unsqueeze(0)).square().sum(dim=1)
        distances[~available] = float('inf')
        index = int(distances.argmin())
        selected.append(index)
        available[index] = False
        running = running + features[index]
    return torch.tensor(selected, dtype=torch.long)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Single-view supervised contrastive loss over same-class batch pairs."""
    if features.ndim != 2 or labels.ndim != 1 or features.size(0) != labels.size(0):
        raise ValueError('expected features [N,D] and labels [N]')
    if features.size(0) == 0 or float(temperature) <= 0:
        raise ValueError('supervised contrastive batch and temperature must be positive')
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    non_self = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & non_self
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return features.sum() * 0.0
    exp_logits = torch.exp(logits) * non_self
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (
        (log_prob * positives).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_positive_log_prob[valid].mean()


def invert_class_party_weights(weights):
    """Reverse each class's party ranking while preserving its weight multiset."""
    if weights.ndim != 2:
        raise ValueError(f'expected [classes, parties] weights, got shape {tuple(weights.shape)}')
    order = torch.argsort(weights, dim=1, stable=True)
    reversed_values = torch.flip(torch.gather(weights, 1, order), dims=[1])
    return torch.empty_like(weights).scatter(1, order, reversed_values)


def reduce_proto_replay_loss(per_sample_ce, old_class_count, mode,
                             class_weights=None):
    """Reduce balanced prototype replay without task-dependent scale drift."""
    if per_sample_ce.ndim != 1 or per_sample_ce.numel() == 0:
        raise ValueError('prototype replay losses must be a non-empty vector')
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights, device=per_sample_ce.device,
            dtype=per_sample_ce.dtype,
        )
        if class_weights.shape != per_sample_ce.shape:
            raise ValueError('prototype replay weights must match losses')
        return (class_weights * per_sample_ce).mean()
    if mode == 'sample_mean':
        return per_sample_ce.mean()
    if mode == 'legacy_class_normalized':
        if int(old_class_count) <= 0:
            raise ValueError('old_class_count must be positive')
        return per_sample_ce.sum() / int(old_class_count)
    raise ValueError(f'unknown prototype replay loss normalization: {mode}')


class DependencyTracker:
    def __init__(self, num_classes, num_parties, momentum=0.9):
        self.contrib = torch.zeros(num_classes, num_parties, dtype=torch.float32)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, party_logits, labels):
        labels = labels.detach()
        for c in labels.unique():
            ci = int(c.item())
            mask = labels == c
            for p, logit_p in party_logits.items():
                val = logit_p[mask, ci].mean().detach().cpu()
                self.contrib[ci, p] = self.momentum * self.contrib[ci, p] + (1.0 - self.momentum) * val


class ProtoEvolveCL:
    """V-LETO faithful implementation."""

    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'ProtoEvolve_VFL'
        self._adaptive_pending_task_id = None
        self._adaptive_event_boundary = None
        self.global_protos = {}
        self.prev_protos = {}
        self.fim_masks = [{} for _ in range(args.num_parties)]
        self.lambda_A = getattr(args, 'proto_lambda_a', 0.1)
        self.proto_replay_loss_norm = getattr(
            args, 'proto_replay_loss_norm', 'legacy_class_normalized'
        )
        self.proto_replay_ratio = float(getattr(args, 'proto_replay_ratio', 1.0))
        if self.proto_replay_ratio <= 0:
            raise ValueError('proto_replay_ratio must be positive')
        self.k0 = 15
        self.alpha = 3
        self.gamma = 1.0
        # Knowledge distillation (restored from v4.1; the 9f9eca0 rewrite dropped it,
        # which removed the only constraint keeping the bottoms from drifting -> collapse)
        self.distill_weight = getattr(args, 'distill_weight', 0.5)
        self.distill_weight_schedule = getattr(
            args, 'distill_weight_schedule', 'constant'
        )
        self.effective_distill_weight = effective_distill_weight(
            self.distill_weight, self.distill_weight_schedule, 0
        )
        self.temperature = getattr(args, 'lwf_temperature', 2.0)
        # Feature-level distillation (PRL form, summed squared-L2): anchor the
        # current aggregated embedding to the frozen previous-task embedding on
        # current data. Logit-KD alone pins only the *composed* function on current
        # inputs, letting the bottoms rotate the feature space (old-class inputs then
        # land in random regions -> collapse regardless of head protection). Feature-
        # KD pins the space itself, so prototypes stay valid AND old inputs keep
        # mapping near them. weight~1 is calibrated for the summed-L2 form. 0=off.
        self.feat_distill_weight = getattr(args, 'feat_distill_weight', 0.0)
        self.current_supcon_weight = float(
            getattr(args, 'current_supcon_weight', 0.0)
        )
        self.current_supcon_temperature = float(
            getattr(args, 'current_supcon_temperature', 0.1)
        )
        self.current_supcon_start_task = int(
            getattr(args, 'current_supcon_start_task', 5)
        )
        effective_supcon_weight(
            self.current_supcon_weight, self.current_supcon_start_task, 0
        )
        if self.current_supcon_temperature <= 0:
            raise ValueError('current_supcon_temperature must be positive')
        self.current_task_id = 0
        self._old_bottoms = None
        self._old_top = None
        # --- Semantic Drift Compensation (Yu et al. CVPR 2020) ---
        # Stored prototypes live in the embedding space of self._old_bottoms (end
        # of the previous task). As the bottoms keep training this task that space
        # drifts, so without compensation loss_A replays stale points the head
        # classifies trivially (~0 anti-forgetting gradient). _sdc_update() shifts
        # the replayed means to their current location each epoch (from a frozen
        # anchor, so repeated shifts never compound). Exemplar-free: only means.
        self._proto_anchor = None
        self.use_sdc = getattr(args, 'proto_sdc', True)
        self.sdc_interval = getattr(args, 'sdc_interval', 1)
        self.sdc_max_samples = getattr(args, 'sdc_max_samples', 4000)
        self.dep_tracking_enabled = bool(getattr(args, 'dep_tracking_enabled', 0))
        self.dep_tracker = DependencyTracker(args.num_classes, args.num_parties,
                                            momentum=getattr(args, 'dep_tracking_momentum', 0.9))
        self.class_party_contrib = {}
        self.class_party_weights = {}
        self.current_task_classes = []
        self.party_kd_enabled = bool(getattr(args, 'party_kd_enabled', 0))
        self.party_kd_mode = getattr(args, 'party_kd_mode', 'uniform')
        self.party_kd_lambda = getattr(args, 'party_kd_lambda', 1.0)
        self.party_weight_manifest = None
        self.party_shuffle_mapping = {}
        self.party_weight_manifest_hash = ''
        if self.party_kd_mode == 'shuffled':
            self.party_weight_manifest = load_manifest(
                args.party_weight_manifest,
                expected_seed=args.seed,
                expected_parties=args.num_parties,
                expected_classes=args.num_classes,
            )
            self.party_weight_manifest_hash = self.party_weight_manifest['tensor_sha256']
            for task_id, classes in enumerate(self.party_weight_manifest['task_classes']):
                mapping = build_task_derangement(
                    classes, int(args.party_shuffle_seed) * 1000003 + task_id
                )
                self.party_shuffle_mapping.update(mapping)
        self.party_proto_enabled = bool(getattr(args, 'party_proto_enabled', 0))
        self.party_proto_mode = getattr(args, 'party_proto_mode', 'uniform')
        self._method_np_rng = None
        self._method_torch_rng = None
        self.head_consolidation_enabled = bool(
            getattr(args, 'head_consolidation_enabled', 0)
        )
        self.head_consolidation_history = []
        self.head_consolidation_regularization = float(
            getattr(args, 'head_consolidation_regularization', 0.01)
        )
        self.head_consolidation_lr = float(getattr(args, 'head_consolidation_lr', 0.01))
        self.head_consolidation_steps = int(getattr(args, 'head_consolidation_steps', 500))
        self.head_consolidation_samples_per_class = int(
            getattr(args, 'head_consolidation_samples_per_class', 20)
        )
        self.head_consolidation_schedule = getattr(
            args, 'head_consolidation_schedule', 'final'
        )
        self.head_consolidation_mode = getattr(
            args, 'head_consolidation_mode', 'full_classifier'
        )
        if self.head_consolidation_mode not in (
                'full_classifier', 'task_class_bias', 'adaptive_dual_branch'):
            raise ValueError(
                f'unknown head consolidation mode: {self.head_consolidation_mode}'
            )
        if (self.head_consolidation_mode == 'adaptive_dual_branch'
                and not bool(getattr(args, 'sanitize_cl_state', 1))):
            raise ValueError(
                'adaptive consolidation requires sanitize_cl_state=1'
            )
        if (self.head_consolidation_mode in (
                'task_class_bias', 'adaptive_dual_branch')
                and self.head_consolidation_schedule != 'final'):
            raise ValueError(
                'task/class bias and adaptive consolidation require schedule=final'
            )
        self.head_consolidation_class_regularization = float(getattr(
            args, 'head_consolidation_class_regularization', 0.01
        ))
        self.head_consolidation_task_regularization = float(getattr(
            args, 'head_consolidation_task_regularization', 0.01
        ))
        self.head_consolidation_task_weight = float(getattr(
            args, 'head_consolidation_task_weight', 1.3
        ))
        head_consolidation_due(
            self.head_consolidation_schedule,
            0,
            int(getattr(args, 'num_tasks', 1)),
        )
        self.party_drift_telemetry_enabled = bool(
            getattr(args, 'party_drift_telemetry', 0)
        )
        self.head_raw_replay = {}
        self.head_task_classes = {}
        self.head_validation_sha256 = ''
        self.adaptive_audit_bundle = None
        self._head_validation_loader_provider = None
        self._head_validation_manifest_provider = None

    def _reset_method_rng(self, task_id):
        seed = derive_seed(self.args.seed, 'method', int(task_id))
        self._method_np_rng = np.random.default_rng(seed)
        self._method_torch_rng = torch.Generator(device=torch.device(self.args.device))
        self._method_torch_rng.manual_seed(seed)

    def before_task(self, task_id, new_classes, seen_classes):
        self.current_task_id = int(task_id)
        self._reset_method_rng(task_id)
        req = max(seen_classes)+1 if seen_classes else 0
        self.trainer.top_model.expand_classes(req, self.args.device)
        if self.dep_tracking_enabled and self.current_task_classes:
            self._freeze_dependency_snapshot(self.current_task_classes)
        self.current_task_classes = list(new_classes)
        # Apply FIM freeze (computed in after_task of previous task)
        if task_id > 0:
            total_f, total_p = 0, 0
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    total_p += 1
                    if self.fim_masks[k].get(n, False):
                        p.requires_grad = False
                        total_f += 1
            print(f"  FIM freeze: {total_f}/{total_p} bottom params frozen")

    @torch.no_grad()
    def _compute_protos(self, loader, replay_capacity=0, replay_seed=0):
        """Build Gaussian prototypes and optional per-class raw replay."""
        for b in self.trainer.bottoms: b.eval()
        class_embs = {}
        class_raw = {}
        for bx, by in loader:
            raw_batch = bx.detach().cpu()
            bx = bx.to(self.args.device)
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(self.args.num_parties)]
            ge = self.trainer._aggregate(embs)
            for i in range(len(by)):
                c = by[i].item()
                if c not in class_embs: class_embs[c] = []
                class_embs[c].append(ge[i].cpu())
                if int(replay_capacity) > 0:
                    class_raw.setdefault(c, []).append(raw_batch[i].clone())
        protos = {}
        replay = {}
        for c, e in class_embs.items():
            stk = torch.stack(e)
            protos[c] = {'mean': stk.mean(0), 'std': stk.std(0).clamp(min=0.01)}
            if int(replay_capacity) > 0:
                indices = herding_indices(stk, replay_capacity)
                replay[c] = torch.stack(class_raw[c]).index_select(0, indices)
        if int(replay_capacity) > 0:
            return protos, replay
        return protos

    @torch.no_grad()
    def _build_party_drift_record(self, task_id):
        if int(task_id) <= 0 or self._old_bottoms is None:
            raise ValueError('party drift requires a previous-task teacher')
        old_classes = sorted(set(self.head_raw_replay) - set(self.current_task_classes))
        if not old_classes or any(c not in self.class_party_weights for c in old_classes):
            raise ValueError('old-class replay or contribution weights are incomplete')
        from party_drift_telemetry import summarize_party_drift
        measurement = summarize_party_drift(
            self._old_bottoms, self.trainer.bottoms,
            {c: self.head_raw_replay[c] for c in old_classes},
            {c: self.class_party_weights[c] for c in old_classes},
            lambda raw: split_features(raw.to(self.args.device), self.args),
        )
        return {
            'schema_version': 1,
            'task_id': int(task_id),
            'seed': int(self.args.seed),
            'old_class_ids': old_classes,
            'current_task_classes': list(self.current_task_classes),
            'measurement': measurement,
            'validation_used': False,
            'test_used': False,
        }

    def _write_party_drift_record(self, task_id):
        if not self.party_drift_telemetry_enabled or int(task_id) == 0:
            return None
        from adaptive_consolidation_audit import (
            _ensure_child_dir, _safe_json, atomic_write_new_json,
        )
        record = self._build_party_drift_record(task_id)
        directory = _ensure_child_dir(self.args.output_dir, 'party_drift')
        path = directory / f'event_{int(task_id)}_CIL.json'
        if path.exists():
            if _safe_json(path) != record:
                raise ValueError('existing party drift record changed on resume')
        else:
            atomic_write_new_json(path, record)
        return path

    @torch.no_grad()
    def _embed_head_raw_replay(self, allowed_classes=None):
        """Re-encode the bounded raw replay with the current bottom models."""
        if not self.head_raw_replay:
            raise ValueError('head raw replay is empty')
        class_ids = sorted(self.head_raw_replay)
        if allowed_classes is not None:
            allowed = {int(class_id) for class_id in allowed_classes}
            class_ids = [
                class_id for class_id in class_ids if int(class_id) in allowed
            ]
            if {int(class_id) for class_id in class_ids} != allowed:
                raise ValueError('retained class is missing adaptive head replay')
        for bottom in self.trainer.bottoms:
            bottom.eval()
        output = {}
        chunk_size = max(1, int(self.args.batch_size))
        for class_id in class_ids:
            raw = self.head_raw_replay[class_id]
            chunks = []
            for start in range(0, raw.size(0), chunk_size):
                batch = raw[start:start + chunk_size].to(self.args.device)
                parts = split_features(batch, self.args)
                embeddings = [
                    self.trainer.bottoms[index](parts[index])
                    for index in range(self.args.num_parties)
                ]
                chunks.append(self.trainer._aggregate(embeddings).cpu())
            output[int(class_id)] = torch.cat(chunks)
        return output

    def set_head_validation_provider(self, loader_provider, manifest_provider):
        """Install lazy validation access without materializing a loader."""
        if not callable(loader_provider) or not callable(manifest_provider):
            raise TypeError('head validation providers must be callable')
        self._head_validation_loader_provider = loader_provider
        self._head_validation_manifest_provider = manifest_provider
        if self.head_consolidation_mode == 'adaptive_dual_branch':
            # Adaptive CIL owns one explicit validation capability. Inherited
            # method hooks (notably proto_evolve_radapt.after_task) must not
            # retain a general dataset handle that can also expose test data.
            self.trainer.dataset_ref = None

    @torch.no_grad()
    def _embed_head_validation(self, ordered_classes):
        """Embed lazy validation data with a frozen current-bottom snapshot."""
        if (self._head_validation_loader_provider is None
                or self._head_validation_manifest_provider is None):
            raise RuntimeError('adaptive head validation provider is not installed')
        classes = tuple(int(class_id) for class_id in ordered_classes)
        if not classes or any(
                left >= right for left, right in zip(classes, classes[1:])):
            raise ValueError('adaptive validation classes must be strictly ordered')

        bottoms = [deepcopy(bottom).eval() for bottom in self.trainer.bottoms]
        for bottom in bottoms:
            for parameter in bottom.parameters():
                parameter.requires_grad = False
        manifest = deepcopy(self._head_validation_manifest_provider())
        if (not isinstance(manifest, dict)
                or not isinstance(manifest.get('sha256'), str)
                or not manifest['sha256']):
            raise ValueError('adaptive validation manifest identity is missing')
        loader = self._head_validation_loader_provider(classes)

        embedded, labels = [], []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.args.device)
            parts = split_features(batch_x, self.args)
            party_embeddings = [
                bottoms[index](parts[index])
                for index in range(self.args.num_parties)
            ]
            embedded.append(self.trainer._aggregate(party_embeddings).detach().cpu())
            labels.append(batch_y.detach().cpu().to(torch.long))
        if not embedded:
            raise ValueError('adaptive validation loader is empty')
        validation_x = torch.cat(embedded)
        validation_y = torch.cat(labels)
        if validation_y.ndim != 1 or set(validation_y.tolist()) != set(classes):
            raise ValueError('adaptive validation labels do not match candidate classes')
        return validation_x, validation_y, manifest

    def _compute_party_logits(self, embs, top_model):
        W = top_model.classifier.weight
        b = top_model.classifier.bias
        P = self.args.num_parties
        if self.args.aggregation == 'sum':
            party_logits = {p: F.linear(embs[p], W, None) for p in range(P)}
        else:
            d = W.size(1) // P
            party_logits = {p: F.linear(embs[p], W[:, p * d:(p + 1) * d], None) for p in range(P)}
        summed = sum(party_logits.values())
        if b is not None:
            summed = summed + b
        if os.environ.get('ASSERT_PARTY_DECOMP'):
            full = top_model(self.trainer._aggregate(embs))
            assert torch.allclose(summed, full, atol=1e-4), 'per-party logit decomposition mismatch'
        return party_logits, summed

    def _track_dependencies(self, party_logits, labels):
        if self.dep_tracking_enabled:
            self.dep_tracker.update(party_logits, labels)

    def _freeze_dependency_snapshot(self, classes):
        for c in classes:
            c = int(c)
            raw = self.dep_tracker.contrib[c].clone()
            self.class_party_contrib[c] = raw.tolist()
            denom = raw.abs().sum().item()
            if denom > 0:
                w = raw.abs() / denom
            else:
                w = torch.full_like(raw, 1.0 / len(raw))
            self.class_party_weights[c] = w.tolist()

    def _party_kd_weights(self, classes, device):
        P = self.args.num_parties
        rows = []
        for c in classes:
            c = int(c)
            if self.party_kd_mode == 'shuffled':
                source = self.party_shuffle_mapping.get(c)
                if source is None:
                    raise ValueError(f'class {c} is missing from shuffled manifest mapping')
                rows.append(self.party_weight_manifest['weights'][source])
            elif self.party_kd_mode in ('static', 'inverse', 'task_marginal') and c in self.class_party_weights:
                rows.append(torch.tensor(self.class_party_weights[c], dtype=torch.float32))
            else:
                rows.append(torch.full((P,), 1.0 / P, dtype=torch.float32))
        weights = torch.stack(rows, dim=0)
        if self.party_kd_mode == 'task_marginal':
            if getattr(self.args, 'custom_tasks', ''):
                task_of = {
                    int(c): task_id
                    for task_id, task in enumerate(self.args.custom_tasks.split('|'))
                    for c in task.split(',')
                }
            else:
                task_of = {int(c): int(c) // self.args.classes_per_task for c in classes}
            grouped = {}
            for row, c in enumerate(classes):
                grouped.setdefault(task_of[int(c)], []).append(row)
            for row_ids in grouped.values():
                mean = weights[row_ids].mean(dim=0)
                weights[row_ids] = mean
        if self.party_kd_mode == 'inverse':
            weights = invert_class_party_weights(weights)
        return weights.to(device)  # [C_old, P]

    def _class_proto_weights(self, classes, device):
        P = self.args.num_parties
        vals = []
        for c in classes:
            c = int(c)
            if self.party_proto_mode == 'static' and c in self.class_party_weights:
                w = torch.tensor(self.class_party_weights[c], dtype=torch.float32)
                vals.append(float(P * (w * w).sum().item()))
            else:
                vals.append(1.0)
        out = torch.tensor(vals, dtype=torch.float32, device=device)
        return out / out.mean().clamp(min=1e-8)

    def _party_kd_loss(self, party_logits_curr, party_logits_old, old_classes, device):
        if not self.party_kd_enabled or len(old_classes) == 0:
            return torch.tensor(0.0, device=device)
        old_classes = [int(c) for c in old_classes]
        idx = torch.tensor(old_classes, dtype=torch.long, device=device)
        w = self._party_kd_weights(old_classes, device).t().unsqueeze(0)  # [1, P, C_old]
        cur = torch.stack([party_logits_curr[p][:, idx] for p in range(self.args.num_parties)], dim=1)
        old = torch.stack([party_logits_old[p][:, idx] for p in range(self.args.num_parties)], dim=1)
        loss = (w * (cur - old) ** 2).sum(dim=(1, 2)).mean()
        return loss / max(len(old_classes), 1)

    def _proto_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        """L = (1-λ_A)*L_CE + λ_A*L_proto(head) + distill_weight*L_KD(bottoms).

        L_proto replays old-class prototype means through the head (head protection).
        L_KD distills the previous-task model's old-class logits into the current
        model WITHOUT detaching the bottom forward, so gradients flow to the bottoms
        and constrain them from drifting -> keeps the stored prototypes valid.
        """
        device = self.args.device
        dbg_A = dbg_kd = dbg_fkd = dbg_supcon = 0.0   # for instrumentation
        # (1) prototype replay on the head
        if self.global_protos:
            pc = list(self.global_protos.keys())
            pe, pl = [], []
            sampled_cls = []
            replay_count = max(1, int(round(
                batch_y.size(0) * self.proto_replay_ratio
            )))
            for _ in range(replay_count):
                c = pc[int(self._method_np_rng.integers(len(pc)))]
                mean = self.global_protos[c]['mean'].to(device)
                std = self.global_protos[c]['std'].to(device)   # use stored per-class std,
                noise = torch.randn(
                    mean.shape, dtype=mean.dtype, device=mean.device,
                    generator=self._method_torch_rng,
                )
                pe.append(mean + noise * std)  # not a fixed 0.01 -> real spread
                pl.append(c)
                sampled_cls.append(c)
            pe = torch.stack(pe)
            pl = torch.tensor(pl, dtype=torch.long, device=device)
            replay_logits = top_model(pe)
            per_sample_ce = F.cross_entropy(replay_logits, pl, reduction='none')
            if self.party_proto_enabled:
                cw = self._class_proto_weights(sampled_cls, device)
                loss_A = reduce_proto_replay_loss(
                    per_sample_ce, len(pc), self.proto_replay_loss_norm,
                    class_weights=cw,
                )
            else:
                loss_A = reduce_proto_replay_loss(
                    per_sample_ce, len(pc), self.proto_replay_loss_norm
                )
            dbg_A = float(self.lambda_A * loss_A)
            total = ((1 - self.lambda_A) * loss_ce + self.lambda_A * loss_A
                     if loss_ce is not None else self.lambda_A * loss_A)
        else:
            total = loss_ce if loss_ce is not None else torch.tensor(0.0, device=device)
        # (2) knowledge distillation to stop bottom drift
        if self._old_top is not None:
            parts = split_features(batch_x, self.args)
            with torch.no_grad():
                old_embs = [self._old_bottoms[i](parts[i]) for i in range(self.args.num_parties)]
                party_logits_old, old_logits = self._compute_party_logits(old_embs, self._old_top)
            curr_embs = [bottoms[i](parts[i]) for i in range(self.args.num_parties)]  # grad flows to bottoms
            party_logits_curr, curr_logits = self._compute_party_logits(curr_embs, top_model)
            self._track_dependencies(party_logits_curr, batch_y)
            # (2a) logit-KD over the classes both heads share. A model-rebuilding UL
            # (retrain) can leave the teacher and current head with mismatched
            # widths; min() keeps them aligned and avoids a shape crash.
            n = min(old_logits.size(1), curr_logits.size(1))
            if n > 0:
                T = self.temperature
                old_p = F.softmax(old_logits[:, :n] / T, dim=1)
                curr_lp = F.log_softmax(curr_logits[:, :n] / T, dim=1)
                loss_kd = F.kl_div(curr_lp, old_p, reduction='batchmean') * (T * T)
                dbg_kd = float(self.effective_distill_weight * loss_kd)
                total = total + self.effective_distill_weight * loss_kd
                loss_party_kd = self._party_kd_loss(
                    party_logits_curr, party_logits_old, list(range(n)), device
                )
                total = total + self.party_kd_lambda * loss_party_kd
            # (2b) feature-KD: squared-L2 anchor SUMMED over feature dims (PRL form,
            # Shi et al. NeurIPS 2024). Summing (not meaning) over the 512 dims keeps
            # the per-coordinate gradient from being diluted ~512x, so a weight ~1
            # actually bites: it pins the embedding space against the cosine-head CE,
            # which -- training only on the new classes -- otherwise drags ALL old
            # features toward the new-class directions and expels old classes
            # (argmax -> new). A cosine (1-cos) form was ~10-50x too weak here.
            # Exemplar-free (uses the frozen previous backbone + current data only).
            supcon_weight = effective_supcon_weight(
                self.current_supcon_weight,
                self.current_supcon_start_task,
                self.current_task_id,
            )
            if self.feat_distill_weight > 0 or supcon_weight > 0:
                old_agg = self.trainer._aggregate(old_embs)
                curr_agg = self.trainer._aggregate(curr_embs)
            if self.feat_distill_weight > 0:
                loss_fkd = ((curr_agg - old_agg) ** 2).sum(dim=1).mean()
                dbg_fkd = float(self.feat_distill_weight * loss_fkd)
                total = total + self.feat_distill_weight * loss_fkd
            if supcon_weight > 0:
                loss_supcon = supervised_contrastive_loss(
                    curr_agg, batch_y, self.current_supcon_temperature
                )
                dbg_supcon = float(supcon_weight * loss_supcon)
                total = total + supcon_weight * loss_supcon
        if os.environ.get('DUMP_PROTO_LOSS'):
            self._dbg = getattr(self, '_dbg', 0) + 1
            if self._dbg % 100 == 1:
                ce_v = float(loss_ce) if loss_ce is not None else 0.0
                print(f"    [proto-loss] CE={ce_v:.4f}  lamA*protoA={dbg_A:.4f}  "
                      f"dw*KD={dbg_kd:.4f}  fkd={dbg_fkd:.4f}  "
                      f"supcon={dbg_supcon:.4f}  "
                      f"(#protos={len(self.global_protos)}, "
                      f"old_teacher={'yes' if self._old_top is not None else 'no'})")
        return total

    def train_task(self, train_loader, task_id):
        self.current_task_id = int(task_id)
        self.effective_distill_weight = effective_distill_weight(
            self.distill_weight, self.distill_weight_schedule, task_id
        )
        print(
            f"  Logit-KD schedule={self.distill_weight_schedule} "
            f"task={task_id} weight={self.effective_distill_weight:.6g}"
        )
        print(
            f"  Current SupCon task={task_id} weight="
            f"{effective_supcon_weight(self.current_supcon_weight, self.current_supcon_start_task, task_id):.6g} "
            f"temperature={self.current_supcon_temperature:.6g}"
        )
        if task_id == 0:
            extra = self._track_only_loss if self.dep_tracking_enabled else None
            history, elapsed = self.trainer.train_task(
                train_loader, self.args.epochs_per_task, extra_loss_fn=extra)
        else:
            history, elapsed = self._train_task_sdc(train_loader)
        # Unfreeze all (before_task may have frozen bottom params via the FIM mask)
        for b in self.trainer.bottoms:
            for p in b.parameters(): p.requires_grad = True
        return history, elapsed

    def _train_task_sdc(self, train_loader):
        """Epoch loop with Semantic Drift Compensation of the replayed prototypes.

        Snapshot the start-of-task prototypes as a frozen anchor (they live in the
        self._old_bottoms space). Before each epoch, re-estimate how far the bottoms
        have drifted and shift global_protos to their current location so loss_A
        keeps replaying each old class where it *now* sits in feature space.
        """
        import time
        optimizers = self.trainer._create_optimizers()
        self._proto_anchor = deepcopy(self.global_protos)
        start = time.time()
        history = []
        for ep in range(self.args.epochs_per_task):
            if self.use_sdc and ep % self.sdc_interval == 0:
                self._sdc_update(train_loader)
            loss, acc = self.trainer.train_epoch(train_loader, optimizers, self._proto_loss)
            history.append({'epoch': ep, 'loss': loss, 'acc': acc})
        if self.use_sdc:
            self._sdc_update(train_loader)   # final shift into the end-of-task space
        return history, time.time() - start

    def _track_only_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        parts = split_features(batch_x, self.args)
        curr_embs = [bottoms[i](parts[i]) for i in range(self.args.num_parties)]
        party_logits_curr, _ = self._compute_party_logits(curr_embs, top_model)
        self._track_dependencies(party_logits_curr, batch_y)
        total = loss_ce if loss_ce is not None else torch.tensor(0.0, device=self.args.device)
        weight = effective_supcon_weight(
            self.current_supcon_weight,
            self.current_supcon_start_task,
            self.current_task_id,
        )
        if weight > 0:
            total = total + weight * supervised_contrastive_loss(
                self.trainer._aggregate(curr_embs), batch_y,
                self.current_supcon_temperature,
            )
        return total

    @torch.no_grad()
    def _sdc_update(self, loader):
        """SDC: shift each stored old-class prototype mean by the embedding drift
        estimated from current-task samples (exemplar-free; old data never stored).

        For sample i the drift is delta_i = phi_cur(x_i) - phi_old(x_i), where
        phi_old = self._old_bottoms (the space the prototypes were stored in) and
        phi_cur the current bottoms. The drift of prototype c is a Gaussian-weighted
        average of the delta_i, weighted by each sample's proximity (in the OLD
        space) to mu_c:
            Delta_c = sum_i w_ic * delta_i,  w_ic propto exp(-||phi_old(x_i)-mu_c||^2 / 2 sigma^2)
        Means are recomputed from the frozen anchor every call, so updates across
        epochs never compound. Degrades gracefully to the global mean drift when all
        samples are equidistant from a prototype.
        """
        if not self._proto_anchor or self._old_bottoms is None:
            return
        dev = self.args.device
        P = self.args.num_parties
        for b in self.trainer.bottoms: b.eval()
        for b in self._old_bottoms: b.eval()
        old_E, cur_E, n = [], [], 0
        for bx, _ in loader:
            bx = bx.to(dev)
            parts = split_features(bx, self.args)
            oe = self.trainer._aggregate([self._old_bottoms[i](parts[i]) for i in range(P)])
            ce = self.trainer._aggregate([self.trainer.bottoms[i](parts[i]) for i in range(P)])
            old_E.append(oe.cpu()); cur_E.append(ce.cpu())
            n += bx.size(0)
            if n >= self.sdc_max_samples: break
        old_E = torch.cat(old_E); cur_E = torch.cat(cur_E)
        delta = cur_E - old_E                                   # [N, D] per-sample drift
        cls = list(self._proto_anchor.keys())
        means = torch.stack([self._proto_anchor[c]['mean'] for c in cls])  # [C, D]
        d2 = torch.cdist(old_E, means) ** 2                     # [N, C] distances in the OLD space
        sigma2 = d2.mean().clamp(min=1e-6)                      # adaptive bandwidth
        w = torch.softmax(-d2 / (2 * sigma2), dim=0)            # [N, C] per-prototype weights over samples
        drift = w.t() @ delta                                   # [C, D] SDC drift per prototype
        for j, c in enumerate(cls):
            self.global_protos[c] = {'mean': self._proto_anchor[c]['mean'] + drift[j],
                                     'std':  self._proto_anchor[c]['std']}
        if os.environ.get('DUMP_PROTO_LOSS'):
            print(f"    [SDC] mean drift_norm={drift.norm(dim=1).mean():.4f} "
                  f"(anchor_mean_norm={means.norm(dim=1).mean():.4f}, "
                  f"#protos={len(cls)}, N={old_E.size(0)})")

    @staticmethod
    def _validated_adaptive_history(history):
        if type(history) is not list:
            raise ValueError('adaptive method history must be a list')
        try:
            return [
                AdaptiveConsolidationResult.from_dict(record).to_dict()
                for record in history
            ]
        except (TypeError, ValueError) as error:
            raise ValueError('adaptive method history is invalid') from error

    def _adaptive_ul_follows(self, task_id):
        after_tasks = {
            int(value) for value in getattr(
                self.args, 'unlearn_after_tasks', []
            )
        }
        available = len(getattr(self.args, 'unlearn_classes', []))
        scheduled = 0
        for candidate in range(int(self.args.num_tasks)):
            if candidate in after_tasks and scheduled < available:
                if candidate == int(task_id):
                    return True
                scheduled += 1
        return False

    def _validated_adaptive_pending_task(self, pending):
        if pending is None:
            return None
        final_task = int(self.args.num_tasks) - 1
        if (type(pending) is not int
                or pending != final_task
                or not head_consolidation_due(
                    self.head_consolidation_schedule,
                    pending,
                    self.args.num_tasks,
                )
                or not self._adaptive_ul_follows(pending)):
            raise ValueError('adaptive pending task mismatch')
        return pending

    def finalize_adaptive_head_after_sanitize(self):
        pending = self._validated_adaptive_pending_task(
            self._adaptive_pending_task_id
        )
        if pending is None:
            return None
        self._adaptive_pending_task_id = None
        try:
            return self._consolidate_head(pending, after_final_ul=True)
        except Exception:
            self._adaptive_pending_task_id = pending
            raise

    def set_adaptive_event_boundary(self, event_idx, event_type):
        event_idx = int(event_idx)
        event_type = str(event_type)
        if event_idx < 0 or event_type not in {'CIL', 'UL'}:
            raise ValueError('adaptive event boundary is invalid')
        self._adaptive_event_boundary = f'event_{event_idx}_{event_type}'

    def _publish_adaptive_metadata(self, history, validation_hash):
        self.head_consolidation_history = history
        self.head_validation_sha256 = validation_hash

    def _commit_adaptive_top(self, temporary_top, replacement_history,
                             replacement_validation_hash,
                             replacement_audit_bundle=None):
        live_top = self.trainer.top_model
        original_state = freeze_state(live_top.state_dict())
        original_hash = hash_top_state(original_state)
        original_history = deepcopy(self.head_consolidation_history)
        original_validation_hash = self.head_validation_sha256
        original_audit_bundle = deepcopy(self.adaptive_audit_bundle)
        target_state = freeze_state(temporary_top.state_dict())
        target_hash = hash_top_state(target_state)
        try:
            live_top.load_state_dict(target_state, strict=True)
            if hash_top_state(live_top) != target_hash:
                raise RuntimeError('adaptive live-head verification failed')
            self.adaptive_audit_bundle = replacement_audit_bundle
            self._publish_adaptive_metadata(
                replacement_history, replacement_validation_hash,
            )
        except Exception:
            live_top.load_state_dict(original_state, strict=True)
            self.head_consolidation_history = original_history
            self.head_validation_sha256 = original_validation_hash
            self.adaptive_audit_bundle = original_audit_bundle
            if hash_top_state(live_top) != original_hash:
                raise RuntimeError('adaptive live-head rollback failed')
            raise

    def _consolidate_head(self, task_id, after_final_ul=False):
        if not self.head_consolidation_enabled:
            return None
        if not head_consolidation_due(
                self.head_consolidation_schedule, task_id, self.args.num_tasks):
            print(
                f"  Head consolidation: deferred "
                f"(schedule={self.head_consolidation_schedule})"
            )
            return None
        if (self.head_consolidation_mode == 'adaptive_dual_branch'
                and not after_final_ul and self._adaptive_ul_follows(task_id)):
            self._adaptive_pending_task_id = int(task_id)
            print('  Head consolidation: deferred until final UL sanitization')
            return None
        seed = derive_seed(self.args.seed, 'head_consolidation', int(task_id))
        pre_top = (
            deepcopy(self.trainer.top_model).eval()
            if self.head_consolidation_mode == 'adaptive_dual_branch'
            else None
        )
        if self.head_consolidation_mode == 'adaptive_dual_branch':
            retained_classes = tuple(sorted(
                int(class_id) for class_id in self.global_protos
            ))
            if not retained_classes:
                raise ValueError('adaptive retained class set is empty')
            retained = set(retained_classes)
            replay_embeddings = self._embed_head_raw_replay(retained_classes)
            prototypes = {
                class_id: self.global_protos[class_id]
                for class_id in retained_classes
            }
            task_classes = {
                int(task_id): kept
                for task_id, classes in self.head_task_classes.items()
                if (kept := [
                    int(class_id) for class_id in classes
                    if int(class_id) in retained
                ])
            }
            raw_count = sum(
                self.head_raw_replay[class_id].size(0)
                for class_id in retained_classes
            )
            candidates = fit_adaptive_candidates(
                pre_top,
                replay_embeddings,
                prototypes,
                task_classes,
                raw_count,
                seed,
                self.args.device,
            )
            event_order = ['candidates_frozen']
            validation_x, validation_y, manifest = self._embed_head_validation(
                candidates.ordered_classes
            )
            from adaptive_consolidation_audit import _canonical_validation_evidence
            manifest, validation_order = _canonical_validation_evidence(
                manifest, validation_y, candidates.ordered_classes
            )
            validation_x = validation_x.index_select(0, validation_order)
            validation_y = validation_y.index_select(0, validation_order)
            # The immutable evidence is recomputed on CPU during audit.  Build
            # the solver evidence on that same canonical device so a genuine
            # CUDA run cannot disagree with its own strict audit.
            canonical_pre = deepcopy(pre_top).cpu().eval()
            validation_x = validation_x.detach().cpu().clone()
            validation_y = validation_y.detach().cpu().clone()
            event_order.append('validation_iterated')
            full_probabilities, bias_probabilities = (
                adaptive_candidate_log_probabilities(
                    canonical_pre, candidates, validation_x
                )
            )
            gate = solve_global_mixture_weight(
                full_probabilities,
                bias_probabilities,
                validation_y,
                candidates.ordered_classes,
            )
            event_order.append('gate_solved')
            temporary_top = install_and_reload_verify(
                canonical_pre, candidates, gate
            )
            event_order.append('state_installed')
            result = AdaptiveConsolidationResult(
                pre_head_sha256=candidates.pre_head_sha256,
                candidate_hashes={
                    'pre': candidates.pre_head_sha256,
                    'full': candidates.full_head_sha256,
                    'bias': candidates.bias_head_sha256,
                },
                candidate_configs={
                    'full': FULL_BRANCH_CONFIG,
                    'bias': BIAS_BRANCH_CONFIG,
                },
                gate=gate,
                validation_manifest=manifest,
                ordered_classes=candidates.ordered_classes,
                task_id=int(task_id),
                task_boundary=(
                    self._adaptive_event_boundary
                    or f'event_{int(task_id)}_CIL'
                ),
            )
            from adaptive_consolidation_audit import _replay_manifest, _tensor_sha256
            retained_raw_replay = {
                class_id: self.head_raw_replay[class_id].detach().cpu().clone()
                for class_id in retained_classes
            }
            diagnostics = build_adaptive_diagnostics(
                canonical_pre, temporary_top, candidates, replay_embeddings,
                validation_x, validation_y, task_classes, gate,
            )
            event_order.append('diagnostics_computed')
            audit_bundle = {
                'method_version': ADAPTIVE_METHOD_VERSION,
                'result': result.to_dict(),
                'pre_state': dict(canonical_pre.state_dict()),
                'full_state': dict(candidates.full_state),
                'bias_state': dict(candidates.bias_state),
                'installed_state': dict(temporary_top.state_dict()),
                'replay_embeddings': deepcopy(replay_embeddings),
                'replay_raw': retained_raw_replay,
                'replay_manifest': _replay_manifest(
                    retained_raw_replay, replay_embeddings
                ),
                'validation_embeddings': validation_x.detach().cpu().clone(),
                'validation_labels': validation_y.detach().cpu().clone(),
                'validation_embeddings_sha256': _tensor_sha256(validation_x),
                'validation_labels_sha256': _tensor_sha256(validation_y),
                'task_classes': deepcopy(task_classes),
                'diagnostics': diagnostics,
                'event_order': event_order,
            }
            history_record = result.to_dict()
            replacement_history = self._validated_adaptive_history(
                self.head_consolidation_history
            ) + [history_record]
            replacement_validation_hash = str(manifest['sha256'])
            self._commit_adaptive_top(
                temporary_top,
                replacement_history,
                replacement_validation_hash,
                audit_bundle,
            )
            return result
        replay_embeddings = self._embed_head_raw_replay()
        raw_count = sum(
            values.size(0) for values in self.head_raw_replay.values()
        )
        if self.head_consolidation_mode == 'task_class_bias':
            audit = consolidate_task_class_bias(
                self.trainer.top_model,
                replay_embeddings,
                self.head_task_classes,
                self.head_consolidation_class_regularization,
                self.head_consolidation_task_regularization,
                self.head_consolidation_task_weight,
                self.head_consolidation_steps,
                self.head_consolidation_lr,
                self.head_consolidation_samples_per_class,
                self.args.device,
                persistent_raw_example_count=raw_count,
                replay_source='balanced_current_encoder_raw_replay',
            )
        else:
            audit = consolidate_classifier(
                self.trainer.top_model,
                self.global_protos,
                self.head_consolidation_regularization,
                self.head_consolidation_steps,
                self.head_consolidation_lr,
                self.head_consolidation_samples_per_class,
                seed,
                self.args.device,
                replay_embeddings=replay_embeddings,
                replay_source='balanced_current_encoder_raw_replay',
                persistent_raw_example_count=raw_count,
            )
            audit['mode'] = 'full_classifier'
        audit['task_id'] = int(task_id)
        audit['task_boundary'] = f'event_{int(task_id)}_CIL'
        audit['schedule'] = self.head_consolidation_schedule
        audit['replay_selection'] = 'normalized_feature_herding'
        self.head_consolidation_history.append(audit)
        output_dir = os.path.join(self.args.output_dir, 'head_consolidation')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"event_{int(task_id)}_CIL.json")
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
        print(
            f"  Head consolidation ({audit['mode']}): "
            f"classes={audit['class_count']} "
            f"samples={audit['fit_sample_count']} "
            f"CE={audit['cross_entropy_before']:.4f}->"
            f"{audit['cross_entropy_after']:.4f}"
        )
        return audit

    def after_task(self, train_loader, task_id):
        """FIM freeze mask + prototype store.

        Old-class prototypes were already SDC-compensated into the current
        embedding space during train_task, so here we only add the fresh
        current-task prototypes and re-cache this model as the frozen reference
        (KD teacher + SDC anchor) for the next task.
        """
        # 1. FIM (global_protos still holds only the OLD classes at this point, so
        #    the radapt subclass can identify this task's new classes by difference)
        self._compute_fim(train_loader, task_id)
        # 2. Current-task prototypes and bounded raw replay (same count as ER).
        if self.head_consolidation_enabled:
            post, post_replay = self._compute_protos(
                train_loader,
                replay_capacity=self.head_consolidation_samples_per_class,
                replay_seed=derive_seed(self.args.seed, 'head_replay', int(task_id)),
            )
        else:
            post = self._compute_protos(train_loader)
            post_replay = {}
        # 3. Merge drift-compensated old state with the new classes.
        if task_id == 0:
            self.global_protos = post.copy()
            self.head_raw_replay = post_replay.copy()
        else:
            for c, p in post.items():
                self.global_protos[c] = p
            for c, values in post_replay.items():
                self.head_raw_replay[c] = values
        self._write_party_drift_record(task_id)
        # 4. Re-encode bounded replay, then apply the configured final head step.
        self.head_task_classes[int(task_id)] = list(self.current_task_classes)
        self._consolidate_head(task_id)
        self.prev_protos = post.copy()
        if self.dep_tracking_enabled and self.current_task_classes:
            self._freeze_dependency_snapshot(self.current_task_classes)
        # Cache the consolidated post-task model as the next frozen teacher/anchor.
        self._old_bottoms = [deepcopy(b).eval() for b in self.trainer.bottoms]
        self._old_top = deepcopy(self.trainer.top_model).eval()
        for ob in self._old_bottoms:
            for p in ob.parameters(): p.requires_grad = False
        for p in self._old_top.parameters(): p.requires_grad = False

    # The legacy _evolve() scalar mean-scaling drift heuristic and its
    # before_train_compute_pre_protos() pre-pass were removed: under the cosine
    # head a scalar mean-scale is a no-op (the head L2-normalises the input), so
    # they delivered zero drift compensation. _sdc_update() replaces them with a
    # true vector shift. runner / probe scripts call before_train_compute_pre_protos
    # behind a hasattr() guard, so its removal is a safe no-op there.

    def _compute_fim(self, loader, task_id):
        for b in self.trainer.bottoms: b.train()
        self.trainer.top_model.train()
        fim = [{n: torch.zeros_like(p) for n, p in self.trainer.bottoms[k].named_parameters()}
               for k in range(self.args.num_parties)]
        nb = 0
        for bx, by in loader:
            bx, by = bx.to(self.args.device), by.to(self.args.device)
            for b in self.trainer.bottoms: b.zero_grad()
            self.trainer.top_model.zero_grad()
            parts = split_features(bx, self.args)
            embs = [self.trainer.bottoms[i](parts[i]) for i in range(self.args.num_parties)]
            out = self.trainer.top_model(self.trainer._aggregate(embs))
            nn.CrossEntropyLoss()(out, by).backward()
            for k in range(self.args.num_parties):
                for n, p in self.trainer.bottoms[k].named_parameters():
                    if p.grad is not None:
                        fim[k][n] += p.grad.data**2
            nb += 1

        frac = getattr(self.args, 'fim_freeze_frac', 0.25)
        for k in range(self.args.num_parties):
            for n in fim[k]: fim[k][n] /= max(nb,1)
            # Per-tensor importance = mean FIM of that parameter tensor.
            imps = {n: fim[k][n].mean().item() for n in fim[k]}
            vals = torch.tensor(list(imps.values()))
            if vals.numel() == 0 or vals.sum() == 0:
                # Don't wipe accumulated mask just because this task gave no signal
                continue
            # Freeze the top `frac` most-important tensors (quantile threshold).
            # BUG FIX: the old threshold kappa = mean - (k0 + alpha*log(t+2))*std
            # with k0=15 went hugely negative, so EVERY non-negative FIM value
            # passed -> 100% of bottom params frozen -> backbone never adapts.
            kappa = torch.quantile(vals, max(0.0, 1.0 - frac)).item()
            # Accumulate mask across tasks: once a param is marked important, keep it frozen
            nf = 0
            n_new_freeze = 0
            for n in fim[k]:
                is_important_now = imps[n] >= kappa
                was_important_before = self.fim_masks[k].get(n, False)
                new_state = was_important_before or is_important_now
                if new_state and not was_important_before:
                    n_new_freeze += 1
                self.fim_masks[k][n] = new_state
                if new_state: nf += 1
            print(f"    Party {k}: {nf}/{len(fim[k])} frozen (+{n_new_freeze} new this task)")

        for b in self.trainer.bottoms:
            b.zero_grad()
            for p in b.parameters(): p.requires_grad = True

    def _adaptive_top_metadata(self):
        top = self.trainer.top_model
        required = (
            '_adaptive_version', '_adaptive_enabled',
            '_adaptive_class_order', '_adaptive_gate',
        )
        if any(not hasattr(top, name) for name in required):
            if self.head_consolidation_mode == 'adaptive_dual_branch':
                raise ValueError('adaptive top version state is unavailable')
            return None, [], None
        enabled = bool(top._adaptive_enabled)
        return (
            int(top._adaptive_version.item()),
            [int(class_id) for class_id in top._adaptive_class_order.tolist()],
            float(top._adaptive_gate.item()) if enabled else None,
        )

    def _validate_adaptive_resume_state(self, state):
        if self.head_consolidation_mode != 'adaptive_dual_branch':
            return None, None
        saved_method_version = state.get('adaptive_method_version')
        if (type(saved_method_version) is not int
                or saved_method_version != ADAPTIVE_METHOD_VERSION):
            raise ValueError('adaptive method version mismatch')
        top_version, class_order, gate = self._adaptive_top_metadata()
        saved_top_version = state.get('adaptive_top_version')
        if (type(saved_top_version) is not int
                or saved_top_version != top_version):
            raise ValueError('adaptive top version mismatch')
        saved_class_order = state.get('adaptive_class_order')
        if (type(saved_class_order) is not list
                or any(type(class_id) is not int
                       for class_id in saved_class_order)
                or saved_class_order != class_order):
            raise ValueError('adaptive class order mismatch')
        saved_gate = state.get('adaptive_gate')
        if ((gate is None and saved_gate is not None)
                or (gate is not None
                    and (type(saved_gate) is not float
                         or saved_gate != gate))):
            raise ValueError('adaptive gate mismatch')

        history = self._validated_adaptive_history(
            state.get('head_consolidation_history')
        )
        validation_hash = state.get('head_validation_sha256')
        if not isinstance(validation_hash, str):
            raise ValueError('adaptive validation hash mismatch')
        if history:
            if len(history) != 1:
                raise ValueError('adaptive method history mismatch')
            last = history[-1]
            if last.get('method_version') != ADAPTIVE_METHOD_VERSION:
                raise ValueError('adaptive method version mismatch')
            if (last.get('task_id') != int(self.args.num_tasks) - 1
                    or not head_consolidation_due(
                        self.head_consolidation_schedule,
                        last.get('task_id'),
                        self.args.num_tasks,
                    )):
                raise ValueError('adaptive method history mismatch')
            if last.get('ordered_classes') != class_order:
                raise ValueError('adaptive class order mismatch')
            history_gate = last.get('gate', {}).get('g')
            if (type(history_gate) is not float
                    or history_gate != gate):
                raise ValueError('adaptive gate mismatch')
            if last.get('validation_manifest', {}).get('sha256') != validation_hash:
                raise ValueError('adaptive validation hash mismatch')
            if top_version != ADAPTIVE_METHOD_VERSION:
                raise ValueError('adaptive top version mismatch')
        elif validation_hash or top_version != 0 or class_order or gate is not None:
            raise ValueError('adaptive method history mismatch')
        pending = self._validated_adaptive_pending_task(
            state.get('adaptive_pending_task_id')
        )
        if pending is not None and (
                history or validation_hash or top_version != 0
                or class_order or gate is not None):
            raise ValueError('adaptive pending task state mismatch')
        return history, pending

    def get_state(self):
        classes = sorted(self.class_party_weights)
        effective_weights = {}
        if classes:
            weights = self._party_kd_weights(classes, 'cpu')
            effective_weights = {c: weights[i].tolist() for i, c in enumerate(classes)}
        top_version, adaptive_classes, adaptive_gate = self._adaptive_top_metadata()
        return {
            'global_protos': deepcopy(self.global_protos),
            'prev_protos': deepcopy(self.prev_protos),
            'fim_masks': deepcopy(self.fim_masks),
            'dep_tracker_contrib': self.dep_tracker.contrib.clone(),
            'current_task_classes': list(self.current_task_classes),
            'has_old_teacher': self._old_top is not None,
            'class_party_contrib': deepcopy(self.class_party_contrib),
            'class_party_weights': deepcopy(self.class_party_weights),
            'effective_class_party_weights': effective_weights,
            'party_weight_manifest_hash': getattr(self, 'party_weight_manifest_hash', ''),
            'party_shuffle_seed': getattr(self.args, 'party_shuffle_seed', -1),
            'party_shuffle_mapping': deepcopy(getattr(self, 'party_shuffle_mapping', {})),
            'distill_weight_schedule': self.distill_weight_schedule,
            'effective_distill_weight': self.effective_distill_weight,
            'current_task_id': self.current_task_id,
            'head_raw_replay': deepcopy(self.head_raw_replay),
            'head_task_classes': deepcopy(self.head_task_classes),
            'head_consolidation_history': deepcopy(self.head_consolidation_history),
            'adaptive_method_version': (
                ADAPTIVE_METHOD_VERSION
                if self.head_consolidation_mode == 'adaptive_dual_branch'
                else None
            ),
            'adaptive_top_version': top_version,
            'adaptive_class_order': adaptive_classes,
            'adaptive_gate': adaptive_gate,
            'head_validation_sha256': self.head_validation_sha256,
            'adaptive_pending_task_id': self._adaptive_pending_task_id,
            'adaptive_audit_bundle': deepcopy(self.adaptive_audit_bundle),
        }
    def load_state(self, s):
        adaptive_history, adaptive_pending = (
            self._validate_adaptive_resume_state(s)
        )
        self.global_protos = deepcopy(s.get('global_protos', {}))
        self.prev_protos = deepcopy(s.get('prev_protos', {}))
        self.fim_masks = deepcopy(
            s.get('fim_masks', [{} for _ in range(self.args.num_parties)])
        )
        contrib = s.get('dep_tracker_contrib')
        if contrib is not None:
            self.dep_tracker.contrib.copy_(torch.as_tensor(contrib))
        self.current_task_classes = list(s.get('current_task_classes', []))
        self.class_party_contrib = deepcopy(s.get('class_party_contrib', {}))
        self.class_party_weights = deepcopy(s.get('class_party_weights', {}))
        self.party_shuffle_mapping = deepcopy(
            s.get('party_shuffle_mapping', self.party_shuffle_mapping)
        )
        saved_schedule = s.get(
            'distill_weight_schedule', self.distill_weight_schedule
        )
        if saved_schedule != self.distill_weight_schedule:
            raise ValueError(
                'checkpoint distillation schedule does not match configuration'
            )
        self.effective_distill_weight = float(s.get(
            'effective_distill_weight', self.effective_distill_weight
        ))
        self.current_task_id = int(s.get('current_task_id', self.current_task_id))
        self.head_raw_replay = deepcopy(s.get('head_raw_replay', {}))
        self.head_task_classes = deepcopy(s.get('head_task_classes', {}))
        self.head_consolidation_history = deepcopy(
            adaptive_history
            if adaptive_history is not None
            else s.get('head_consolidation_history', [])
        )
        self.head_validation_sha256 = s.get('head_validation_sha256', '')
        self.adaptive_audit_bundle = deepcopy(
            s.get('adaptive_audit_bundle')
        )
        self._adaptive_pending_task_id = adaptive_pending
        if s.get('has_old_teacher', False):
            self._old_bottoms = [
                deepcopy(bottom).eval() for bottom in self.trainer.bottoms
            ]
            self._old_top = deepcopy(self.trainer.top_model).eval()
            for bottom in self._old_bottoms:
                for parameter in bottom.parameters():
                    parameter.requires_grad = False
            for parameter in self._old_top.parameters():
                parameter.requires_grad = False
