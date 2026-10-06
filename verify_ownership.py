"""Mathematical verification of the ownership-decomposition innovation.

Reasoning being tested (problem-first, from the linear top model):

  logit_c(x) = b_c + sum_k s_{c,k}(x),   s_{c,k}=<w_c,e_k> (sum) | <w_c^(k),e_k> (concat)

This EXACT additive per-party decomposition is the VFL-specific lever. The
load-bearing empirical claims:

  P1 (exactness)      : sum_k s_{c,k} + b_c == logit_c  (machine precision).
  P2 (heterogeneity)  : ownership pi_c is concentrated & varies across classes
                        (avg max-share >> 1/P; normalized entropy < 1).
  THM1 (surgery)      : forgetting class c by zeroing the contribution of the
                        top-|S*| owning parties drops it below the runner-up;
                        |S*(c)| is small for concentrated classes, and the
                        ownership ordering needs FEWER parties than reverse
                        ordering -> ownership is the right axis.
  THM1-frag           : |S*| grows with fragmentation P (run across P=2,4,8).
  PARTY-LEVEL         : dropping whole parties (party exit) trades forget-c
                        collapse against retained accuracy via redundancy.

No UL training loop is needed: once a backbone is trained, the interventions
are exact arithmetic on the per-party logit terms.

Usage mirrors probe_attribution.py; loops over P live in a driver script.
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from data_utils import VFLDataset, TaskManager, split_features
from models import build_models
from vfl_trainer import VFLTrainer
from cl_methods import get_cl_method


# ---------- training (mirrors probe_attribution.main CIL loop) ----------

def train_backbone(args):
    dataset = VFLDataset(args)
    task_mgr = TaskManager(args)
    timeline = task_mgr.get_timeline()
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    cl_method = get_cl_method(args.cl_method, trainer, args)

    for event in timeline:
        if event['type'] != 'CIL':
            continue  # no UL events in this verification
        tid = event['task_id']
        new_cls = event['new_classes']
        task_mgr.advance_task(tid)
        eff = task_mgr.get_effective_classes()
        task_loader, _ = dataset.get_task_loaders(new_cls)
        cl_method.before_task(tid, new_cls, eff)
        if hasattr(cl_method, 'before_train_compute_pre_protos'):
            cl_method.before_train_compute_pre_protos(task_loader)
        train_loader = (dataset.get_task_loaders(eff)[0]
                        if args.replay_mode == 'full' else task_loader)
        t0 = time.time()
        cl_method.train_task(train_loader, tid)
        cl_method.after_task(task_loader, tid)
        print(f"  [train] task {tid} eff={eff} in {time.time()-t0:.1f}s")

    eff_final = task_mgr.get_effective_classes()
    return trainer, dataset, eff_final


# ---------- exact per-(sample,class,party) contribution tensor ----------

@torch.no_grad()
def contribution_tensor(trainer, dataset, classes, args):
    """Return S (N,C,K) signed per-party contributions to EVERY class logit,
    b (C,), labels y (N,), and the model's own logits L (N,C) for exactness."""
    device = args.device
    P = args.num_parties
    classes = sorted(int(c) for c in classes)
    cidx = {c: i for i, c in enumerate(classes)}
    W = trainer.top_model.classifier.weight.detach()          # (Cfull, top_dim)
    b = trainer.top_model.classifier.bias.detach()            # (Cfull,)
    embed_dim = W.size(1) // (P if args.aggregation == 'concat' else 1)
    if args.aggregation == 'concat':
        W_parts = [W[:, k*embed_dim:(k+1)*embed_dim] for k in range(P)]  # each (Cfull,d)

    for bm in trainer.bottoms: bm.eval()
    _, loader = dataset.get_task_loaders(classes, shuffle_train=False)

    S_chunks, y_chunks, L_chunks = [], [], []
    Wc = W[classes]                                           # (C, top_dim) for sum
    bc = b[classes]                                           # (C,)
    for bx, by in loader:
        bx = bx.to(device)
        parts = split_features(bx, args)
        embs = [trainer.bottoms[k](parts[k]) for k in range(P)]   # (B,d) each
        B = bx.size(0)
        Sb = torch.empty(B, len(classes), P, device=device)
        for k in range(P):
            if args.aggregation == 'sum':
                Sb[:, :, k] = embs[k] @ Wc.t()               # <w_c, e_k>
            else:
                Sb[:, :, k] = embs[k] @ W_parts[k][classes].t()
        S_chunks.append(Sb.cpu())
        y_chunks.append(by.clone())
        # model's own logits restricted to `classes`
        agg = trainer._aggregate(embs)
        L_chunks.append(trainer.top_model(agg)[:, classes].cpu())

    S = torch.cat(S_chunks).numpy()                           # (N,C,K)
    y = torch.cat(y_chunks).numpy()
    L = torch.cat(L_chunks).numpy()                           # (N,C)
    return S, bc.cpu().numpy(), y, L, classes


# ---------- ownership pi_c, redundancy ----------

def ownership(S, y, classes):
    """pi[c] = mean over D_c of |s_{c,k}| / sum_k|s_{c,k}|  -> (C_forget, K)."""
    P = S.shape[2]
    pi, maxshare, norment = {}, {}, {}
    for ci, c in enumerate(classes):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        a = np.abs(S[mask, ci, :])                            # (n_c, K) own-class contrib
        sh = a / (a.sum(axis=1, keepdims=True) + 1e-12)
        m = sh.mean(axis=0)                                   # (K,)
        pi[c] = m
        maxshare[c] = float(m.max())
        ent = -(m * np.log2(m + 1e-12)).sum()
        norment[c] = float(ent / (np.log2(P) if P > 1 else 1.0))
    return pi, maxshare, norment


# ---------- THM1: targeted minimal forget set |S*(c)| ----------

def minimal_forget_set(S, b, y, L, classes, tau=0.05):
    """For each candidate forget class c: greedily zero the contribution of the
    top-owning parties to c's logit ONLY; record how many parties needed for
    forget-acc <= tau. Targeted surgery does not change other classes' logits,
    so retained accuracy is provably unaffected.

    Returns per-class dict with |S*| under ownership order and reverse order,
    plus the full forget-acc curve under ownership order."""
    C, K = len(classes), S.shape[2]
    pi, _, _ = ownership(S, y, classes)
    full_logit = S.sum(axis=2) + b[None, :]                   # (N,C) reconstructed
    out = {}
    for ci, c in enumerate(classes):
        mask = (y == c)
        if mask.sum() == 0 or c not in pi:
            continue
        order_own = list(np.argsort(-pi[c]))                 # high ownership first
        order_rev = list(np.argsort(pi[c]))                  # low ownership first

        def sstar_and_curve(order):
            curve = []
            sstar = K
            for m in range(K + 1):
                drop = order[:m]
                logit = full_logit[mask].copy()
                # zero class c's contribution from the dropped parties only
                if m > 0:
                    logit[:, ci] -= S[mask][:, ci, drop].sum(axis=1)
                pred = logit.argmax(axis=1)
                facc = float((pred == ci).mean())
                curve.append(facc)
                if facc <= tau and sstar == K:
                    sstar = m
            return sstar, curve

        s_own, curve_own = sstar_and_curve(order_own)
        s_rev, _ = sstar_and_curve(order_rev)
        out[c] = {'sstar_ownership': s_own, 'sstar_reverse': s_rev,
                  'forget_curve_ownership': curve_own,
                  'max_share': float(pi[c].max())}
    return out


# ---------- PARTY-LEVEL: drop whole parties (party exit) ----------

def party_drop(S, b, y, classes):
    """Drop whole parties (set e_k=0 for ALL classes) in descending order of
    average ownership over all classes; record overall accuracy and, for the
    single most-concentrated class, its accuracy. Shows redundancy tradeoff."""
    K = S.shape[2]
    full_logit = S.sum(axis=2) + b[None, :]
    # global party importance = mean |contribution| over all samples/classes
    imp = np.abs(S).mean(axis=(0, 1))                         # (K,)
    order = list(np.argsort(-imp))
    cls_arr = np.array(classes)
    ytrue_idx = np.array([np.where(cls_arr == yy)[0][0] if yy in classes else -1 for yy in y])
    valid = ytrue_idx >= 0
    curve = []
    for m in range(K + 1):
        keep = [k for k in range(K) if k not in order[:m]]
        logit = (S[:, :, keep].sum(axis=2) if keep else np.zeros_like(full_logit)) + b[None, :]
        pred = logit.argmax(axis=1)
        acc = float((pred[valid] == ytrue_idx[valid]).mean())
        curve.append(acc)
    return {'party_importance': imp.tolist(), 'drop_order': [int(o) for o in order],
            'overall_acc_vs_parties_dropped': curve}


# ---------- main ----------

def main():
    args = get_config()
    print(f"[verify] data={args.data} P={args.num_parties} agg={args.aggregation} "
          f"cl={args.cl_method} epochs/task={args.epochs_per_task} seed={args.seed}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer, dataset, eff = train_backbone(args)
    S, b, y, L, classes = contribution_tensor(trainer, dataset, eff, args)

    # P1: exactness
    recon = S.sum(axis=2) + b[None, :]
    exact_err = float(np.abs(recon - L).max())

    # P2: ownership concentration
    pi, maxshare, norment = ownership(S, y, classes)
    avg_max = float(np.mean(list(maxshare.values())))
    avg_ent = float(np.mean(list(norment.values())))

    # THM1
    forget = minimal_forget_set(S, b, y, L, classes, tau=0.05)
    sstar_own = [v['sstar_ownership'] for v in forget.values()]
    sstar_rev = [v['sstar_reverse'] for v in forget.values()]
    K = args.num_parties

    # PARTY-LEVEL
    pdrop = party_drop(S, b, y, classes)

    report = {
        'config': {'data': args.data, 'P': K, 'aggregation': args.aggregation,
                   'cl_method': args.cl_method, 'epochs_per_task': args.epochs_per_task,
                   'seed': args.seed, 'n_classes': len(classes), 'n_test': int(len(y))},
        'P1_exactness_max_abs_err': exact_err,
        'P2_ownership': {
            'uniform_baseline': 1.0 / K,
            'avg_max_share': avg_max,
            'avg_norm_entropy': avg_ent,
            'per_class_max_share': {int(c): maxshare[c] for c in maxshare},
        },
        'THM1_minimal_forget_set': {
            'tau': 0.05,
            'mean_sstar_ownership': float(np.mean(sstar_own)),
            'mean_sstar_reverse': float(np.mean(sstar_rev)),
            'frac_sstar_eq_1': float(np.mean([s == 1 for s in sstar_own])),
            'frac_sstar_lt_K': float(np.mean([s < K for s in sstar_own])),
            'frac_needs_all_K': float(np.mean([s == K for s in sstar_own])),
            'per_class': {int(c): forget[c] for c in forget},
        },
        'PARTY_LEVEL_drop': pdrop,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'ownership_verify.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # console
    print("\n" + "=" * 70)
    print(f"  OWNERSHIP VERIFICATION  (data={args.data} P={K} agg={args.aggregation})")
    print("=" * 70)
    print(f"  P1  exactness max|recon-logit|   = {exact_err:.2e}   (want ~0)")
    print(f"  P2  uniform share baseline       = {1.0/K:.3f}")
    print(f"      avg max-share across classes = {avg_max:.3f}   (want >> uniform)")
    print(f"      avg normalized entropy       = {avg_ent:.3f}   (0=spike,1=uniform)")
    print(f"  THM1 mean |S*| (ownership order) = {np.mean(sstar_own):.2f} / {K}")
    print(f"       mean |S*| (reverse  order)  = {np.mean(sstar_rev):.2f} / {K}   (want > ownership)")
    print(f"       frac classes |S*|=1         = {np.mean([s==1 for s in sstar_own]):.2f}")
    print(f"       frac classes |S*|<K         = {np.mean([s<K for s in sstar_own]):.2f}")
    print(f"  PARTY drop overall-acc curve     = {[round(a,3) for a in pdrop['overall_acc_vs_parties_dropped']]}")
    print(f"\n  Report: {out}")
    print("=" * 70)


if __name__ == '__main__':
    main()
