"""Verification probes for the CBM-based VFL CLU paper direction.

We have NOT committed to a method yet. Before doing so we test two empirical
preconditions on the existing V-LETO backbone:

Q1 — Per-party knowledge localization.
    Top model is a single Linear (models.py), so for sum/concat aggregation the
    per-party contribution to any class logit decomposes EXACTLY as
        logit_y = sum_k (W_y_slice_k . e_k)
    No leave-one-out approximation needed. We measure how concentrated this
    distribution is over parties. If 1-2 parties dominate each class's logit
    on average, knowledge IS spatially localized in VFL and CBM-style
    attribution has structural traction.

Q2 — Single-party class recognizability.
    For each party we freeze its bottom and train a linear probe on its
    embedding to classify all seen classes. If some classes are recognizable
    from one party alone (high per-party probe accuracy), the ownership
    structure is strong enough to support surgical (sparse) unlearning.

Decision criteria (parameterized on num_parties P, uniform baseline = 1/P):
    Q1 STRONG    : avg_max_share > min(0.6, 1/P + 0.3)
    Q1 MODERATE  : avg_max_share > 1/P + 0.15
    Q2 STRONG    : >50% of classes have a single party achieving probe acc > 0.70
    Q2 MODERATE  : >30% of classes have a single party achieving probe acc > 0.50

Outputs:
    {output_dir}/probe_report.json   -- machine-readable, includes per-task
                                          ownership snapshots so we can also
                                          observe ownership drift across CL.
    Console: summary table.

Usage (example, on cluster with the same env as the main grid):
    python -u probe_attribution.py \
        --data cifar10 --num_classes 10 \
        --custom_tasks "0,1|2,3|4,5|6,7|8,9" \
        --num_parties 4 --aggregation sum --model_type resnet18 \
        --epochs_per_task 20 --batch_size 128 --lr 0.01 \
        --replay_mode prototype --cl_method proto_evolve \
        --seed 42 --device cuda:0 \
        --unlearn_after_tasks 99,99 --unlearn_classes "0;0" \
        --exp_name probe_attribution

The unlearn_* flags are required by config.py but never triggered (no UL
events are dispatched here).
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import get_config
from data_utils import VFLDataset, TaskManager, split_features
from models import build_models
from vfl_trainer import VFLTrainer
from cl_methods import get_cl_method


# ---------- Q1: closed-form per-party contribution ----------

def compute_per_party_contributions(trainer, dataset, classes, args):
    """For test samples of each class, compute per-party contribution to the
    sample's own-class logit. Linear top + sum/concat -> EXACT decomposition.

    Returns: {class -> np.ndarray of shape (n_samples, num_parties)} with
    SIGNED contributions (so a party that pushes the logit DOWN shows up as
    negative; we work in absolute values for ownership shares).
    """
    device = args.device
    P = args.num_parties
    W = trainer.top_model.classifier.weight.detach()  # (C, top_dim)
    embed_dim = W.size(1) // (P if args.aggregation == 'concat' else 1)
    if args.aggregation == 'concat':
        W_parts = [W[:, i*embed_dim:(i+1)*embed_dim] for i in range(P)]

    for b in trainer.bottoms: b.eval()
    contribs = {c: [] for c in classes}
    _, loader = dataset.get_task_loaders(classes, shuffle_train=False)
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device); by = by.to(device)
            parts = split_features(bx, args)
            embs = [trainer.bottoms[i](parts[i]) for i in range(P)]  # (B, embed_dim) each
            for i in range(bx.size(0)):
                y = int(by[i].item())
                if y not in contribs:
                    continue
                if args.aggregation == 'sum':
                    cs = [float((W[y] * embs[k][i]).sum().item()) for k in range(P)]
                else:
                    cs = [float((W_parts[k][y] * embs[k][i]).sum().item()) for k in range(P)]
                contribs[y].append(cs)
    return {c: np.array(v) for c, v in contribs.items() if v}


def summarize_ownership(contribs, P):
    """Per-class summary statistics over the |contribution| share distribution.

    For each class y:
        share_{k} = mean over test samples of  |contrib_{k}| / sum_j |contrib_{j}|
    Then max_share, normalized entropy (0=spike, 1=uniform), Gini, argmax.
    """
    out = {}
    for c, mat in contribs.items():
        abs_mat = np.abs(mat)
        sums = abs_mat.sum(axis=1, keepdims=True)
        sums[sums < 1e-12] = 1.0
        shares = abs_mat / sums            # (N, P)
        mean_share = shares.mean(axis=0)   # (P,)
        max_share = float(mean_share.max())
        eps = 1e-12
        entropy = float(-(mean_share * np.log2(mean_share + eps)).sum())
        max_ent = np.log2(P) if P > 1 else 1.0
        norm_ent = entropy / max_ent if max_ent > 0 else 1.0
        # Gini
        sorted_s = np.sort(mean_share)
        n = len(sorted_s)
        denom = n * sorted_s.sum() + 1e-12
        gini = float(((2 * np.arange(1, n+1) - n - 1) @ sorted_s) / denom)
        out[c] = {
            'mean_share_per_party': mean_share.tolist(),
            'max_share': max_share,
            'argmax_party': int(mean_share.argmax()),
            'normalized_entropy': norm_ent,
            'gini': gini,
            'n_samples': int(mat.shape[0]),
        }
    return out


# ---------- Q2: per-party linear probe ----------

def per_party_linear_probe(trainer, dataset, classes, args, max_iter=400,
                            max_train=5000):
    """For each party, fit a logistic regression on its frozen embedding to
    classify `classes`. Return (per_party_overall_acc, per_class_acc).

    per_class_acc[c][k] = recall on class c by party-k's probe.

    Speed: subsample train to `max_train` (stratified-ish via shuffle) and use
    saga; full max_iter=3000/full-data was ~hours under GPU contention with
    negligible verdict change.
    """
    P = args.num_parties
    device = args.device
    for b in trainer.bottoms: b.eval()
    train_loader, test_loader = dataset.get_task_loaders(classes, shuffle_train=False)

    def extract(loader):
        embs_per_party = [[] for _ in range(P)]
        labels = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                parts = split_features(bx, args)
                for k in range(P):
                    embs_per_party[k].append(trainer.bottoms[k](parts[k]).cpu().numpy())
                labels.append(by.numpy())
        return [np.concatenate(e) for e in embs_per_party], np.concatenate(labels)

    Xtr_list, ytr = extract(train_loader)
    Xte_list, yte = extract(test_loader)

    # Subsample train for speed (shuffled, capped at max_train)
    if max_train and len(ytr) > max_train:
        rng = np.random.RandomState(0)
        sel = rng.permutation(len(ytr))[:max_train]
        Xtr_list = [X[sel] for X in Xtr_list]
        ytr = ytr[sel]

    per_party_overall = []
    per_class_acc = {int(c): {} for c in classes}
    for k in range(P):
        # Scale + LogReg: convergence without scaling fails on raw ResNet18
        # embeddings even at max_iter=500, biasing Q2 verdicts downward.
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=max_iter, solver='saga',
                                               n_jobs=-1))
        clf.fit(Xtr_list[k], ytr)
        preds = clf.predict(Xte_list[k])
        per_party_overall.append(float((preds == yte).mean()))
        for c in classes:
            mask = (yte == c)
            per_class_acc[int(c)][k] = float((preds[mask] == yte[mask]).mean()) if mask.any() else None
    return per_party_overall, per_class_acc


# ---------- Bottom-embedding similarity sanity ----------

def bottom_diversity_sanity(trainer, dataset, classes, args, n_batches=4):
    """Pairwise cosine similarity between party embeddings on a few batches.
    If parties collapsed to near-identical features, all attribution is moot.
    Returns mean pairwise cosine across all party-pairs.
    """
    P = args.num_parties
    if P < 2:
        return None
    device = args.device
    for b in trainer.bottoms: b.eval()
    _, loader = dataset.get_task_loaders(classes, shuffle_train=False)
    cos_vals = []
    with torch.no_grad():
        for bi, (bx, _) in enumerate(loader):
            if bi >= n_batches: break
            bx = bx.to(device)
            parts = split_features(bx, args)
            embs = [trainer.bottoms[k](parts[k]) for k in range(P)]  # (B, D)
            embs_n = [F.normalize(e, dim=1) for e in embs]
            for i in range(P):
                for j in range(i+1, P):
                    # avg cosine across the batch
                    cos_vals.append(float((embs_n[i] * embs_n[j]).sum(dim=1).mean().item()))
    return float(np.mean(cos_vals)) if cos_vals else None


# ---------- Verdict logic ----------

def q1_verdict(avg_max_share, P):
    uniform = 1.0 / P
    if avg_max_share > min(0.60, uniform + 0.30):
        return "STRONG"
    if avg_max_share > uniform + 0.15:
        return "MODERATE"
    return "WEAK"


def q2_verdict(per_class_max_acc, n_classes):
    if n_classes == 0:
        return "UNDEFINED"
    p70 = sum(1 for v in per_class_max_acc.values() if v is not None and v > 0.70) / n_classes
    p50 = sum(1 for v in per_class_max_acc.values() if v is not None and v > 0.50) / n_classes
    if p70 > 0.50:
        return "STRONG"
    if p50 > 0.30:
        return "MODERATE"
    return "WEAK"


# ---------- Main ----------

def main():
    args = get_config()
    print(f"[probe] cfg: data={args.data} P={args.num_parties} agg={args.aggregation} "
          f"cl={args.cl_method} model={args.model_type} seed={args.seed}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = VFLDataset(args)
    task_mgr = TaskManager(args)
    timeline = task_mgr.get_timeline()

    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    cl_method = get_cl_method(args.cl_method, trainer, args)

    per_task_ownership = []

    for idx, event in enumerate(timeline):
        if event['type'] != 'CIL':
            continue  # probe ignores UL events
        tid = event['task_id']
        new_cls = event['new_classes']
        task_mgr.advance_task(tid)
        eff = task_mgr.get_effective_classes()
        task_loader, _ = dataset.get_task_loaders(new_cls)
        print(f"\n[probe] CIL task {tid}: new={new_cls} eff={eff}")
        cl_method.before_task(tid, new_cls, eff)
        if hasattr(cl_method, 'before_train_compute_pre_protos'):
            cl_method.before_train_compute_pre_protos(task_loader)
        train_loader = (dataset.get_task_loaders(eff)[0]
                        if args.replay_mode == 'full' else task_loader)
        t0 = time.time()
        cl_method.train_task(train_loader, tid)
        cl_method.after_task(task_loader, tid)
        print(f"  trained in {time.time()-t0:.1f}s")

        # Q1 snapshot
        contribs = compute_per_party_contributions(trainer, dataset, eff, args)
        owner = summarize_ownership(contribs, args.num_parties)
        per_task_ownership.append({
            'task': tid, 'eff_classes': eff, 'per_class': owner,
        })
        avg_max = float(np.mean([owner[c]['max_share'] for c in eff]))
        avg_ent = float(np.mean([owner[c]['normalized_entropy'] for c in eff]))
        print(f"  Q1 snapshot: avg_max_share={avg_max:.3f} (uniform={1/args.num_parties:.3f}), "
              f"avg_norm_entropy={avg_ent:.3f}")

    # Q2 on final effective set
    eff_final = task_mgr.get_effective_classes()
    print(f"\n[probe] Q2: per-party linear probe on {len(eff_final)} classes...")
    t0 = time.time()
    per_party_overall, per_class_acc = per_party_linear_probe(
        trainer, dataset, eff_final, args)
    print(f"  Q2 probes trained in {time.time()-t0:.1f}s")

    max_party_per_class = {}
    for c in eff_final:
        accs = [v for v in per_class_acc[c].values() if v is not None]
        if accs:
            max_party_per_class[c] = max(accs)

    cls_70 = sum(1 for v in max_party_per_class.values() if v > 0.70)
    cls_50 = sum(1 for v in max_party_per_class.values() if v > 0.50)

    # Redundancy heterogeneity: how much does max-party-acc vary ACROSS classes?
    # High std/range -> some classes are "single-party recognizable", others need
    # multi-party combination. Low std/range -> uniform redundancy structure.
    mpa_vals = list(max_party_per_class.values())
    if mpa_vals:
        het = {
            'max_party_acc_mean':  float(np.mean(mpa_vals)),
            'max_party_acc_std':   float(np.std(mpa_vals)),
            'max_party_acc_min':   float(np.min(mpa_vals)),
            'max_party_acc_max':   float(np.max(mpa_vals)),
            'max_party_acc_range': float(np.max(mpa_vals) - np.min(mpa_vals)),
            'easy_classes_above_0.60': sorted(
                int(c) for c, v in max_party_per_class.items() if v > 0.60),
            'hard_classes_below_0.30': sorted(
                int(c) for c, v in max_party_per_class.items() if v < 0.30),
        }
    else:
        het = {}

    final_avg_max_share = float(np.mean(
        [per_task_ownership[-1]['per_class'][c]['max_share'] for c in eff_final]))
    final_avg_entropy = float(np.mean(
        [per_task_ownership[-1]['per_class'][c]['normalized_entropy'] for c in eff_final]))

    # Sanity: party embedding diversity
    cos_sim = bottom_diversity_sanity(trainer, dataset, eff_final, args)

    report = {
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (str, int, float, bool, list, type(None)))},
        'sanity': {
            'mean_pairwise_party_embedding_cosine': cos_sim,
            'note': '> 0.95 means parties collapsed to ~identical features; '
                    'attribution becomes meaningless.',
        },
        'per_task_ownership': per_task_ownership,
        'q2': {
            'per_party_overall_acc': per_party_overall,
            'per_class_max_party_acc': max_party_per_class,
            'per_class_per_party_acc': {int(c): {int(k): v for k, v in d.items()}
                                         for c, d in per_class_acc.items()},
            'n_classes_max_party_above_0.70': int(cls_70),
            'n_classes_max_party_above_0.50': int(cls_50),
            'n_classes_total': len(eff_final),
            'heterogeneity': het,
        },
        'decision_summary': {
            'num_parties': args.num_parties,
            'uniform_share_baseline': 1.0 / args.num_parties,
            'final_avg_max_share': final_avg_max_share,
            'final_avg_normalized_entropy': final_avg_entropy,
            'q1_verdict': q1_verdict(final_avg_max_share, args.num_parties),
            'q2_verdict': q2_verdict(max_party_per_class, len(eff_final)),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'probe_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Console summary
    ds = report['decision_summary']
    print("\n" + "=" * 72)
    print("  PROBE SUMMARY")
    print("=" * 72)
    print(f"  data={args.data} P={args.num_parties} agg={args.aggregation} "
          f"cl={args.cl_method} seed={args.seed}")
    if cos_sim is not None:
        flag = "  [WARNING: bottoms collapsed]" if cos_sim > 0.95 else ""
        print(f"  Sanity: mean pairwise party-embedding cosine = {cos_sim:.3f}{flag}")
    print("\n  Q1 (per-party logit attribution, FINAL TASK):")
    print(f"    uniform baseline             = {ds['uniform_share_baseline']:.3f}")
    print(f"    avg max-share across classes = {ds['final_avg_max_share']:.3f}")
    print(f"    avg normalized entropy       = {ds['final_avg_normalized_entropy']:.3f}  (0=spike, 1=uniform)")
    print(f"    -> VERDICT: {ds['q1_verdict']}")
    print("\n  Q1 ownership drift across CL tasks:")
    for snap in per_task_ownership:
        eff = snap['eff_classes']
        ams = float(np.mean([snap['per_class'][c]['max_share'] for c in eff]))
        ent = float(np.mean([snap['per_class'][c]['normalized_entropy'] for c in eff]))
        print(f"    after task {snap['task']}: avg_max_share={ams:.3f}  "
              f"avg_norm_entropy={ent:.3f}  ({len(eff)} classes)")
    print("\n  Q2 (per-party linear probe):")
    print("    per-party overall acc: " +
          ", ".join(f"P{i}={a:.3f}" for i, a in enumerate(per_party_overall)))
    print(f"    classes with max-party probe acc > 0.70: "
          f"{cls_70}/{len(eff_final)}")
    print(f"    classes with max-party probe acc > 0.50: "
          f"{cls_50}/{len(eff_final)}")
    print(f"    -> VERDICT: {ds['q2_verdict']}")
    print("\n  ====> Overall recommendation <====")
    q1v, q2v = ds['q1_verdict'], ds['q2_verdict']
    if q1v == "STRONG" and q2v in ("STRONG", "MODERATE"):
        rec = "GO — CBM/concept-ownership direction is structurally supported."
    elif q1v == "MODERATE" or q2v == "MODERATE":
        rec = "CONDITIONAL — some structure exists; concept layer may amplify it. Worth a small CBM pilot before committing."
    else:
        rec = "RECONSIDER — knowledge is too uniformly distributed; CBM's attribution advantage will not materialize. Look for a different angle."
    print(f"    {rec}")
    print(f"\n  Report: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
