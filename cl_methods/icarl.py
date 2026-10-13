"""iCaRL: herded exemplars, binary distillation, nearest-exemplar-mean readout."""
import copy
import random

import torch
import torch.nn.functional as F

from data_utils import split_features
from .er import ERCL


def herding_indices(features, count):
    """Greedily approximate the normalized class mean without duplicates."""
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError('herding requires a nonempty feature matrix')
    values = F.normalize(features.float(), dim=1)
    mean = values.mean(0)
    selected = []
    running = torch.zeros_like(mean)
    available = torch.ones(values.size(0), dtype=torch.bool,
                           device=values.device)
    for step in range(1, min(int(count), len(values)) + 1):
        distances = ((running + values) / step - mean).square().sum(1)
        distances[~available] = float('inf')
        index = int(distances.argmin())
        selected.append(index)
        available[index] = False
        running += values[index]
    return selected


class ICaRLCL(ERCL):
    def __init__(self, trainer, args):
        super().__init__(trainer, args)
        self.name = 'iCaRL_VFL'
        self.total_budget = 20 * int(args.num_classes)
        if self.total_budget <= 0:
            raise ValueError('iCaRL requires a positive final class count')
        self.new_classes = []
        self.old_classes = []
        self.teacher_bottoms = None
        self.teacher_top = None
        self._task_current_count = None

    def before_task(self, task_id, new_classes, seen_classes):
        super().before_task(task_id, new_classes, seen_classes)
        self.new_classes = list(new_classes)
        self.old_classes = [int(label) for label in seen_classes
                            if label not in new_classes]

    def _features(self, examples, bottoms=None, normalize=True,
                  record_comm=True):
        bottoms = self.trainer.bottoms if bottoms is None else bottoms
        pieces = split_features(examples.to(self.args.device), self.args)
        embeddings = [bottom(piece) for bottom, piece in zip(bottoms, pieces)]
        if record_comm:
            self.trainer.bytes_transmitted += sum(
                value.numel() * value.element_size() * 2 for value in embeddings)
            self.trainer.comm_rounds += 1
        aggregate = self.trainer._aggregate(embeddings)
        return F.normalize(aggregate, dim=1) if normalize else aggregate

    def _icarl_loss(self, bottoms, top_model, batch_x, batch_y, loss_ce=None):
        stored = [(example, label)
                  for label, examples in self.buffer.data.items()
                  for example in examples]
        if stored:
            picked = random.sample(stored, min(self.buffer_batch_size, len(stored)))
            old_x = torch.stack([example for example, _ in picked]).to(self.args.device)
            old_y = torch.tensor([label for _, label in picked],
                                 dtype=torch.long, device=self.args.device)
            examples = torch.cat((batch_x, old_x), 0)
            labels = torch.cat((batch_y, old_y), 0)
        else:
            examples, labels = batch_x, batch_y
        logits = top_model(self._features(examples, bottoms, normalize=False))
        targets = torch.zeros_like(logits)
        targets.scatter_(1, labels[:, None], 1.0)
        if self.old_classes:
            if self.teacher_bottoms is None or self.teacher_top is None:
                raise ValueError('iCaRL old-class teacher is absent')
            with torch.no_grad():
                teacher_logits = self.teacher_top(
                    self._features(examples, self.teacher_bottoms,
                                   normalize=False))
                old_indices = torch.tensor(self.old_classes,
                                           dtype=torch.long, device=labels.device)
                targets[:, old_indices] = teacher_logits[:, old_indices].sigmoid()
        per_example = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none').mean(1)
        if not stored:
            return per_example.mean()
        fresh_count = len(batch_y)
        current_total = (self._task_current_count
                         if self._task_current_count is not None
                         else fresh_count)
        old_total = len(stored)
        return ((current_total * per_example[:fresh_count].mean()
                 + old_total * per_example[fresh_count:].mean())
                / (current_total + old_total))

    def train_task(self, train_loader, task_id):
        if not hasattr(train_loader, 'dataset') or len(train_loader.dataset) < 1:
            raise ValueError('iCaRL requires a nonempty current-task dataset')
        self._task_current_count = len(train_loader.dataset)
        try:
            return self.trainer.train_task(
                train_loader, self.args.epochs_per_task,
                extra_loss_fn=self._icarl_loss)
        finally:
            self._task_current_count = None

    @torch.no_grad()
    def after_task(self, train_loader, task_id):
        per_class = self.total_budget // (len(self.old_classes) + len(self.new_classes))
        if per_class < 1:
            raise ValueError('iCaRL budget cannot retain every seen class')
        for class_id in self.old_classes:
            self.buffer.data[class_id] = self.buffer.data[class_id][:per_class]
        was_training = [bottom.training for bottom in self.trainer.bottoms]
        for bottom in self.trainer.bottoms:
            bottom.eval()
        examples_by_class = {int(c): [] for c in self.new_classes}
        features_by_class = {int(c): [] for c in self.new_classes}
        try:
            for examples, labels in train_loader:
                features = self._features(examples).cpu()
                for example, label, feature in zip(examples, labels, features):
                    class_id = int(label)
                    if class_id in examples_by_class:
                        examples_by_class[class_id].append(example.detach().cpu().clone())
                        features_by_class[class_id].append(feature)
            for class_id in self.new_classes:
                examples = examples_by_class[class_id]
                if not examples:
                    raise ValueError(f'iCaRL class {class_id} has no training examples')
                feature_matrix = torch.stack(features_by_class[class_id])
                selected = herding_indices(feature_matrix, per_class)
                self.buffer.data[class_id] = [examples[index] for index in selected]
                self.buffer.seen_count[class_id] = len(examples)
        finally:
            for bottom, training in zip(self.trainer.bottoms, was_training):
                bottom.train(training)
        self.teacher_bottoms = copy.deepcopy(self.trainer.bottoms)
        self.teacher_top = copy.deepcopy(self.trainer.top_model)
        for model in [*self.teacher_bottoms, self.teacher_top]:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _class_means(self):
        labels, means = [], []
        for class_id in sorted(self.buffer.data):
            examples = self.buffer.data[class_id]
            if not examples:
                continue
            features = self._features(torch.stack(examples), record_comm=False)
            labels.append(class_id)
            means.append(F.normalize(features.mean(0), dim=0))
        if not labels:
            raise ValueError('iCaRL cannot infer without exemplars')
        return torch.tensor(labels, dtype=torch.long,
                            device=self.args.device), torch.stack(means)

    @torch.no_grad()
    def _evaluate(self, task_classes, batches_for_task):
        modules = [*self.trainer.bottoms, self.trainer.top_model]
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            return self._evaluate_inference(task_classes, batches_for_task)
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)

    @torch.no_grad()
    def _evaluate_inference(self, task_classes, batches_for_task):
        class_ids, means = self._class_means()
        result = {name: {} for name in (
            'class_il_pred_task', 'class_il_global', 'task_il_oracle',
            'task_prediction', 'class_il_pred_task_counts')}
        class_task = {int(label): int(task)
                      for task, classes in task_classes.items()
                      for label in classes}
        for task, classes in task_classes.items():
            correct = head_correct = oracle_correct = predicted_task = total = 0
            allowed = torch.tensor([class_id in classes for class_id in class_ids.tolist()],
                                   dtype=torch.bool, device=self.args.device)
            allowed_indices = torch.where(allowed)[0]
            if not len(allowed_indices):
                raise ValueError('iCaRL task lacks exemplar means')
            for examples, labels in batches_for_task(task, classes):
                features = self._features(examples, record_comm=False)
                distances = torch.cdist(features, means)
                predictions = class_ids[distances.argmin(1)]
                oracle = class_ids[allowed_indices[distances[:, allowed].argmin(1)]]
                head = self.trainer.top_model(
                    self._features(examples, normalize=False,
                                   record_comm=False)).argmax(1)
                target = labels.to(self.args.device)
                correct += int((predictions == target).sum())
                oracle_correct += int((oracle == target).sum())
                head_correct += int((head == target).sum())
                predicted_task += sum(class_task[int(value)] == task
                                      for value in predictions.tolist())
                total += len(target)
            if total == 0:
                raise ValueError('iCaRL evaluation task is empty')
            key = f'task_{task}'
            result['class_il_pred_task'][key] = round(correct / total, 4)
            result['class_il_global'][key] = round(head_correct / total, 4)
            result['task_il_oracle'][key] = round(oracle_correct / total, 4)
            result['task_prediction'][key] = round(predicted_task / total, 4)
            result['class_il_pred_task_counts'][key] = {
                'correct': correct, 'total': total}
        return result

    def evaluate_class_il_readouts(self, dataset, seen_task_classes):
        return self._evaluate(
            seen_task_classes,
            lambda _task, classes: dataset.get_task_loaders(
                classes, shuffle_train=False)[1],
        )

    def evaluate_class_il_readouts_cached(self, cached_test_batches,
                                          seen_task_classes):
        from runner import select_formal_cached_batches
        return self._evaluate(
            seen_task_classes,
            lambda _task, classes: select_formal_cached_batches(
                cached_test_batches, classes),
        )

    def get_state(self):
        state = super().get_state()
        state.update(
            total_budget=self.total_budget,
            new_classes=list(self.new_classes),
            old_classes=list(self.old_classes),
            teacher_bottoms=(None if self.teacher_bottoms is None else
                             [model.state_dict() for model in self.teacher_bottoms]),
            teacher_top=(None if self.teacher_top is None else
                         self.teacher_top.state_dict()),
        )
        return state

    def load_state(self, state):
        if state.get('total_budget') != self.total_budget:
            raise ValueError('iCaRL checkpoint uses a different total exemplar budget')
        super().load_state(state)
        self.new_classes = list(state.get('new_classes', []))
        self.old_classes = list(state.get('old_classes', []))
        teacher_top = state.get('teacher_top')
        teacher_bottoms = state.get('teacher_bottoms')
        if teacher_top is None:
            self.teacher_bottoms = self.teacher_top = None
            return
        if not isinstance(teacher_bottoms, list):
            raise ValueError('iCaRL teacher bottom states are missing')
        from models import TopModel
        self.teacher_bottoms = copy.deepcopy(self.trainer.bottoms)
        if len(self.teacher_bottoms) != len(teacher_bottoms):
            raise ValueError('iCaRL teacher party count differs')
        for model, packed in zip(self.teacher_bottoms, teacher_bottoms):
            model.load_state_dict(packed)
        teacher_classes = teacher_top['classifier.weight'].shape[0]
        current = self.trainer.top_model
        self.teacher_top = TopModel(
            current.classifier.in_features, teacher_classes,
            cosine=current.cosine).to(self.args.device)
        self.teacher_top.load_state_dict(teacher_top)
        for model in [*self.teacher_bottoms, self.teacher_top]:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
