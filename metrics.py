"""Evaluation metrics for CL, UL, and federated communication."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_utils import split_features
import torch.nn.functional as F


def cache_formal_batches(loader):
    """Materialize one loader iteration as detached, cloned CPU batches."""
    cached = []
    for batch in loader:
        if (type(batch) not in (list, tuple) or len(batch) != 2
                or not all(isinstance(value, torch.Tensor) for value in batch)):
            raise ValueError('formal test batches must be tensor input/label pairs')
        batch_x, batch_y = batch
        if (batch_y.ndim != 1 or batch_x.ndim < 1
                or batch_x.size(0) != batch_y.size(0)):
            raise ValueError('formal test batch dimensions are invalid')
        cached.append((
            batch_x.detach().cpu().clone(),
            batch_y.detach().cpu().clone(),
        ))
    if not cached:
        raise ValueError('formal test loader produced no batches')
    return tuple(cached)


def select_formal_cached_batches(cached_batches, classes):
    """Select class rows from an immutable CPU cache without reopening data."""
    ordered = [int(class_id) for class_id in classes]
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError('formal cached selection classes are invalid')
    selected = []
    for batch in cached_batches:
        if (type(batch) is not tuple or len(batch) != 2
                or not all(isinstance(value, torch.Tensor) for value in batch)):
            raise ValueError('formal cache batch schema is invalid')
        batch_x, batch_y = batch
        if (batch_x.device.type != 'cpu' or batch_y.device.type != 'cpu'
                or batch_y.ndim != 1 or batch_x.size(0) != batch_y.size(0)):
            raise ValueError('formal cache must contain aligned CPU tensors')
        mask = torch.zeros_like(batch_y, dtype=torch.bool)
        for class_id in ordered:
            mask |= batch_y == class_id
        if bool(mask.any()):
            selected.append((batch_x[mask], batch_y[mask]))
    if not selected:
        raise ValueError('formal cache has no labels for requested classes')
    return tuple(selected)


class MetricsTracker:
    """Track and compute all metrics across the experiment timeline."""

    def __init__(self):
        self.task_acc_matrix = []  # acc_matrix[i][j] = acc on task j after training task i
        self.ul_metrics = []
        self.comm_stats = []
        self.timing = []
        self.step_results = []  # All intermediate results

    def record_task_accuracies(self, step_name, per_task_accs, overall_acc,
                               per_task_debiased=None, per_task_taskil=None,
                               companion_readouts=None):
        """Record accuracy on each task group after a CL or UL event.

        per_task_debiased / per_task_taskil are optional companion readouts
        (recency-debiased class-IL and within-task task-IL); stored alongside
        the raw class-IL so the same matrix can be reported three ways.
        """
        entry = {
            'step': step_name,
            'per_task_accs': per_task_accs,
            'overall_acc': overall_acc,
        }
        if per_task_debiased is not None:
            entry['per_task_accs_debiased'] = per_task_debiased
        if per_task_taskil is not None:
            entry['per_task_accs_taskil'] = per_task_taskil
        if companion_readouts:
            entry['companion_readouts'] = companion_readouts
        self.task_acc_matrix.append(entry)

    def record_ul_result(self, step_name, ul_result):
        self.ul_metrics.append({'step': step_name, **ul_result})

    def record_comm(self, step_name, comm):
        self.comm_stats.append({'step': step_name, **comm})

    def record_timing(self, step_name, elapsed):
        self.timing.append({'step': step_name, 'time_seconds': round(elapsed, 2)})

    def record_step(self, step_info):
        """Record any intermediate step result."""
        self.step_results.append(step_info)

    def load_dict(self, state):
        """Restore a previously exported tracker state."""
        self.task_acc_matrix = list(state.get('task_acc_history', []))
        self.ul_metrics = list(state.get('ul_metrics', []))
        self.comm_stats = list(state.get('comm_stats', []))
        self.timing = list(state.get('timing', []))
        self.step_results = list(state.get('step_results', []))

    def compute_cl_metrics(self):
        """Compute CL metrics, properly separating CIL and UL events.

        Returns:
            AA_cil: mean AA across CIL events only (true CL performance).
            AA_ul: mean AA across UL events only (post-unlearning utility).
            AA_final: AA after the final event in the timeline.
            BWT: Backward Transfer, computed only between consecutive CIL events
                 (using the intersection of evaluated tasks).
            AA_trajectory: per-event AA values for plotting.
        """
        if len(self.task_acc_matrix) < 1:
            return {'AA_cil': 0.0, 'AA_ul': 0.0, 'AA_final': 0.0, 'BWT': 0.0, 'AA_trajectory': []}

        aa_trajectory = []
        for entry in self.task_acc_matrix:
            accs = [v for v in entry['per_task_accs'].values() if v is not None]
            aa_trajectory.append({
                'step': entry['step'],
                'AA': round(float(np.mean(accs)), 4) if accs else 0.0
            })

        cil_aa = [e['AA'] for e in aa_trajectory if 'CIL' in e['step']]
        ul_aa = [e['AA'] for e in aa_trajectory if 'UL' in e['step']]
        AA_cil = round(float(np.mean(cil_aa)), 4) if cil_aa else 0.0
        AA_ul = round(float(np.mean(ul_aa)), 4) if ul_aa else 0.0
        AA_final = aa_trajectory[-1]['AA'] if aa_trajectory else 0.0

        # BWT: only between consecutive CIL events, only on tasks present in BOTH.
        # We do NOT compare across UL events because UL changes the task set.
        cil_entries = [e for e in self.task_acc_matrix if 'CIL' in e['step']]
        bwt_values = []
        for i in range(1, len(cil_entries)):
            curr = cil_entries[i]['per_task_accs']
            prev = cil_entries[i - 1]['per_task_accs']
            for task_key in prev:
                if task_key in curr and prev[task_key] is not None and curr[task_key] is not None:
                    bwt_values.append(curr[task_key] - prev[task_key])
        BWT = round(float(np.mean(bwt_values)), 4) if bwt_values else 0.0
        deferred_final = next((
            entry for entry in reversed(cil_entries)
            if entry.get('deferred_final') is True
        ), None)
        if deferred_final is not None:
            diagonal = deferred_final.get('deferred_diagonal', {})
            final_row = deferred_final['per_task_accs']
            ordered = sorted(
                set(diagonal) & set(final_row),
                key=lambda key: int(key.rsplit('_', 1)[1]),
            )
            final_task = deferred_final.get('deferred_final_task')
            comparisons = [key for key in ordered if key != final_task]
            BWT = round(float(np.mean([
                final_row[key] - diagonal[key] for key in comparisons
            ])), 4) if comparisons else 0.0

        result = {
            'AA_cil': AA_cil,
            'AA_ul': AA_ul,
            'AA_final': AA_final,
            'BWT': BWT,
            'AA_trajectory': aa_trajectory,
            # Keep legacy 'AA' (= mean across all events) for backward compat,
            # but the three above are the authoritative metrics.
            'AA': round(float(np.mean([e['AA'] for e in aa_trajectory])), 4),
        }

        # Companion readouts (present iff the eval recorded them; older runs
        # without these keys are unaffected). Recency-debiased class-IL removes
        # the shared-head task-recency bias; task-IL is within-task argmax.
        def _aa_traj(key):
            tr = []
            for entry in self.task_acc_matrix:
                accs = [v for v in entry.get(key, {}).values() if v is not None]
                tr.append({'step': entry['step'],
                           'AA': round(float(np.mean(accs)), 4) if accs else 0.0})
            return tr
        for key, suffix in (('per_task_accs_debiased', 'debiased'),
                            ('per_task_accs_taskil', 'taskil')):
            if any(key in e for e in self.task_acc_matrix):
                tr = _aa_traj(key)
                cil = [e['AA'] for e in tr if 'CIL' in e['step']]
                result['AA_final_' + suffix] = tr[-1]['AA'] if tr else 0.0
                result['AA_cil_' + suffix] = round(float(np.mean(cil)), 4) if cil else 0.0
                result['AA_trajectory_' + suffix] = tr
        companion_names = sorted({
            name
            for entry in self.task_acc_matrix
            for name in entry.get('companion_readouts', {})
        })
        for name in companion_names:
            trajectory = []
            for entry in self.task_acc_matrix:
                values = entry.get('companion_readouts', {}).get(name, {})
                valid = [value for value in values.values() if value is not None]
                trajectory.append({
                    'step': entry['step'],
                    'AA': round(float(np.mean(valid)), 4) if valid else 0.0,
                })
            cil = [entry['AA'] for entry in trajectory if 'CIL' in entry['step']]
            result[f'AA_final_{name}'] = trajectory[-1]['AA'] if trajectory else 0.0
            result[f'AA_cil_{name}'] = round(float(np.mean(cil)), 4) if cil else 0.0
            result[f'AA_trajectory_{name}'] = trajectory
        return result

    def to_dict(self):
        """Export all metrics as a dictionary."""
        cl = self.compute_cl_metrics()
        return {
            'cl_metrics': cl,
            'task_acc_history': self.task_acc_matrix,
            'ul_metrics': self.ul_metrics,
            'comm_stats': self.comm_stats,
            'timing': self.timing,
            'step_results': self.step_results,
        }


def collect_global_probs(trainer, test_loader, num_classes, args):
    """Softmax outputs placed into global-class-indexed columns [N, num_classes].

    A model's top layer outputs over its own class space (column j = global class
    id j, since classes are global indices and out_dim = max(eff)+1). We scatter
    those into a fixed [N, num_classes] matrix so Oracle and any method are
    comparable column-by-column regardless of their individual top-layer width.
    """
    for b in trainer.bottoms: b.eval()
    trainer.top_model.eval()
    rows, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(args.device)
            parts = split_features(batch_x, args)
            embs = [trainer.bottoms[i](parts[i]) for i in range(len(trainer.bottoms))]
            agg = trainer._aggregate(embs)
            out = trainer.top_model(agg)
            p = torch.softmax(out, dim=1).cpu()
            full = torch.zeros(p.size(0), num_classes)
            # Some CL methods keep an augmented head (out_dim > num_classes, e.g.
            # rotation-SSL); columns 0..num_classes-1 are the real global classes
            # (consistent with how evaluate() argmaxes). Clip to num_classes.
            k = min(p.size(1), num_classes)
            full[:, :k] = p[:, :k]
            rows.append(full); labels.append(batch_y.cpu())
    if not rows:
        return np.zeros((0, num_classes), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return torch.cat(rows).numpy().astype(np.float32), torch.cat(labels).numpy().astype(np.int64)


def compute_kl_to_oracle(oracle_probs, method_probs, retained_classes, eps=1e-8):
    """Mean per-sample KL(p_oracle || p_method) on the retained-class support.

    Both prob matrices are global-class-indexed and evaluated on the same
    (shuffle=False) retained test set, so rows align sample-for-sample. We slice
    the retained columns, renormalize, and average KL over samples. Lower = the
    method's output distribution is closer to the Oracle (the paper's core judge).
    """
    if oracle_probs.shape[0] == 0 or method_probs.shape[0] == 0:
        return None
    n = min(oracle_probs.shape[0], method_probs.shape[0])
    cols = list(retained_classes)
    po = oracle_probs[:n][:, cols].astype(np.float64)
    pm = method_probs[:n][:, cols].astype(np.float64)
    po = po / (po.sum(1, keepdims=True) + eps)
    pm = pm / (pm.sum(1, keepdims=True) + eps)
    po = np.clip(po, eps, 1.0); pm = np.clip(pm, eps, 1.0)
    kl = (po * (np.log(po) - np.log(pm))).sum(1)
    return float(np.mean(kl))


def evaluate_per_task(trainer, dataset, task_classes_dict, device):
    """Evaluate model accuracy on each task's test data separately."""
    per_task_accs = {}
    for task_id, classes in task_classes_dict.items():
        _, test_loader = dataset.get_task_loaders(classes, shuffle_train=False)
        acc, _, _ = trainer.evaluate(test_loader)
        per_task_accs[f'task_{task_id}'] = round(acc, 4)
    return per_task_accs


def evaluate_per_task_full(trainer, dataset, task_classes_dict, device, eps=1e-12):
    """Per-task accuracy under THREE exemplar-free readouts (no retraining):

      - class_il (raw):   full-head argmax over all columns — the headline
                          class-incremental accuracy. Identical to
                          evaluate_per_task (same probs, same argmax), so the
                          existing AA/BWT are unchanged.
      - class_il_debiased: logit-adjusted argmax over SEEN classes. We subtract
                          log(pi_c), where pi_c is class c's mean predicted
                          softmax mass over the balanced seen test set, i.e. the
                          shared head's systematic per-class bias. On a balanced
                          test set an unbiased head has pi_c = 1/|seen|; the
                          deviation is pure task-recency bias, and removing it
                          (standard logit adjustment, Menon et al. 2021) recovers
                          the old-class information the softmax head was masking.
      - task_il:          argmax restricted to each task's own class columns
                          (within-task discriminability; the representation view).

    Returns (class_il, class_il_debiased, task_il), each a dict keyed
    'task_{id}'. All exemplar-free: pi_c uses model predictions only, no labels.
    """
    task_probs, task_labels = {}, {}
    for task_id, classes in task_classes_dict.items():
        _, test_loader = dataset.get_task_loaders(classes, shuffle_train=False)
        _, probs, labels = trainer.evaluate(test_loader)
        task_probs[task_id] = probs.cpu().numpy()
        task_labels[task_id] = labels.cpu().numpy()
    return _full_task_readouts(
        task_probs, task_labels, task_classes_dict, eps=eps
    )


def evaluate_per_task_full_cached(
        trainer, cached_test_batches, task_classes_dict, device, eps=1e-12):
    """Loader-equivalent readouts using only a detached all-class CPU cache."""
    task_probs, task_labels = {}, {}
    for task_id, classes in task_classes_dict.items():
        try:
            selected = select_formal_cached_batches(
                cached_test_batches, classes
            )
        except ValueError as error:
            raise ValueError(
                f'formal cache has no labels for task {task_id}'
            ) from error
        _, probs, labels = trainer.evaluate(selected)
        task_probs[task_id] = probs.detach().cpu().numpy()
        task_labels[task_id] = labels.detach().cpu().numpy()
    return _full_task_readouts(
        task_probs, task_labels, task_classes_dict, eps=eps
    )


def _full_task_readouts(task_probs, task_labels, task_classes_dict, eps=1e-12):
    seen = {
        int(class_id)
        for classes in task_classes_dict.values()
        for class_id in classes
    }

    class_il, class_il_deb, task_il = {}, {}, {}
    if not seen or not task_probs:
        return class_il, class_il_deb, task_il

    allP = np.concatenate([task_probs[t] for t in task_probs], axis=0)
    C = allP.shape[1]
    seen_mask = np.zeros(C, dtype=bool)
    seen_mask[sorted(seen)] = True
    log_pi = np.log(np.where(seen_mask, allP.mean(0), 1.0) + eps)  # per-class head bias
    NEG = -1e30

    for task_id, classes in task_classes_dict.items():
        P, y = task_probs[task_id], task_labels[task_id]
        # raw class-IL: full-head argmax (matches trainer.evaluate)
        class_il[f'task_{task_id}'] = round(float((P.argmax(1) == y).mean()), 4)
        # debiased class-IL: subtract head bias, argmax over seen classes only
        adj = np.where(seen_mask, np.log(P + eps) - log_pi, NEG)
        class_il_deb[f'task_{task_id}'] = round(float((adj.argmax(1) == y).mean()), 4)
        # task-IL: argmax within this task's own columns
        cls = list(classes)
        pred_in = np.asarray(cls)[P[:, cls].argmax(1)]
        task_il[f'task_{task_id}'] = round(float((pred_in == y).mean()), 4)
    return class_il, class_il_deb, task_il


def evaluate_unlearning(trainer, dataset, forget_classes, retain_classes, args):
    """Evaluate unlearning effectiveness."""
    results = {}

    # Accuracy on forgotten classes (should be low = good unlearning)
    if forget_classes:
        _, forget_test = dataset.get_task_loaders(forget_classes, shuffle_train=False)
        forget_acc, forget_probs, forget_labels = trainer.evaluate(forget_test)
        results['forget_acc'] = round(forget_acc, 4)
    else:
        results['forget_acc'] = None

    # Accuracy on retained classes (should stay high)
    if retain_classes:
        _, retain_test = dataset.get_task_loaders(retain_classes, shuffle_train=False)
        retain_acc, _, _ = trainer.evaluate(retain_test)
        results['retain_acc'] = round(retain_acc, 4)
    else:
        results['retain_acc'] = None

    # MIA: Membership Inference Attack on forget data
    if forget_classes and retain_classes:
        loaders = dataset.get_forget_retain_loaders(forget_classes,
                                                     forget_classes + retain_classes)
        mia_score = compute_mia(trainer, loaders, args)
        results['mia_score'] = round(mia_score, 4)
    else:
        results['mia_score'] = None

    return results


def compute_mia(trainer, loaders, args):
    """Membership Inference Attack using logistic regression on prediction entropy."""
    retain_probs = _collect_probs(trainer, loaders['retain_train'], args)
    forget_probs = _collect_probs(trainer, loaders['forget_train'], args)
    test_probs = _collect_probs(trainer, loaders['retain_test'], args)

    retain_entropy = _entropy(retain_probs).numpy().reshape(-1, 1)
    forget_entropy = _entropy(forget_probs).numpy().reshape(-1, 1)
    test_entropy = _entropy(test_probs).numpy().reshape(-1, 1)

    # Train: retain (member=1) vs test (non-member=0)
    X_train = np.concatenate([retain_entropy, test_entropy])
    y_train = np.concatenate([np.ones(len(retain_entropy)), np.zeros(len(test_entropy))])

    # Predict on forget set
    X_forget = forget_entropy

    if len(X_train) < 2 or len(X_forget) < 1:
        return 0.5

    clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_forget)
    # MIA score: fraction predicted as "member". Lower = better unlearning (closer to 0.5)
    return float(preds.mean())


def _collect_probs(trainer, loader, args):
    """Collect softmax probabilities."""
    for b in trainer.bottoms: b.eval()
    trainer.top_model.eval()
    probs = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device)
            parts = split_features(batch_x, args)
            embs = [trainer.bottoms[i](parts[i]) for i in range(len(trainer.bottoms))]
            agg = trainer._aggregate(embs)
            output = trainer.top_model(agg)
            probs.append(torch.softmax(output, dim=1).cpu())
    return torch.cat(probs) if probs else torch.tensor([])


def _entropy(probs, dim=-1):
    """Compute entropy of probability distributions."""
    if probs.numel() == 0:
        return torch.tensor([])
    return -torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs)).sum(dim=dim)
