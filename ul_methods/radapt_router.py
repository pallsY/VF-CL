"""Redundancy-Adaptive UL Router.

Reads per-class redundancy info that the matching CL method
(ProtoEvolveRadaptCL) stashed on `trainer.radapt_state`, then dispatches
the actual unlearning to one of two existing backends per forget class:

    redundancy_score(c) >= easy_thresh  ->  LIGHT path = LUV
        (the class can be recognized from a single party's features, so
        a top-only update suffices; touches no bottom => cheap)

    redundancy_score(c) <  easy_thresh  ->  HEAVY path = FedOSD
        (the class is distributed across parties; need orthogonal-descent
        coordination across all bottoms to actually remove it)

Multi-class forget request: we partition the classes by redundancy and
run LIGHT and HEAVY in sequence. Comm and time are accumulated; the per-
class routing decisions are returned in `history` for the paper table.

Fallback: if redundancy_state is empty (e.g., paired with a non-radapt CL),
default to HEAVY for ALL classes (safe but loses the comm benefit).
"""
from __future__ import annotations
import time
from ul_methods.luv import LUVUL
from ul_methods.fedosd import FedOSDUL


class RadaptRouterUL:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        self.name = 'RadaptRouter'
        self.light = LUVUL(trainer, args)
        self.heavy = FedOSDUL(trainer, args)
        self.default_thresh = float(getattr(args, 'radapt_easy_thresh', 0.50))

    def _split_classes_by_redundancy(self, forget_classes):
        state = getattr(self.trainer, 'radapt_state', None) or {}
        scores = state.get('per_class_max_party_acc', {})
        thresh = float(state.get('easy_thresh', self.default_thresh))
        easy, hard, decisions = [], [], {}
        for c in forget_classes:
            s = scores.get(int(c))
            if s is None:
                decisions[int(c)] = {'score': None, 'route': 'HEAVY(no-score)'}
                hard.append(c)
            elif s >= thresh:
                decisions[int(c)] = {'score': s, 'route': 'LIGHT(LUV)'}
                easy.append(c)
            else:
                decisions[int(c)] = {'score': s, 'route': 'HEAVY(FedOSD)'}
                hard.append(c)
        return easy, hard, decisions, thresh

    def unlearn(self, forget_classes, retain_train_loader, forget_train_loader, **kw):
        easy, hard, decisions, thresh = self._split_classes_by_redundancy(forget_classes)
        # Console-visible routing summary -> ends up in the run log
        print(f"  [radapt-router] thresh={thresh:.2f}  easy={easy}  hard={hard}")
        for c, dec in decisions.items():
            print(f"    class {c}: score={dec['score']!r}  -> {dec['route']}")

        start = time.time()
        history = []

        # NOTE: we share the SAME retain/forget loaders for both sub-calls.
        # That's the same input each baseline receives in runner.py, so we
        # are NOT helping the router via privileged data.
        if hard:
            print(f"  [radapt-router] HEAVY path (FedOSD) on {len(hard)} classes")
            r_hard = self.heavy.unlearn(hard, retain_train_loader, forget_train_loader, **kw)
            history.append({'path': 'HEAVY', 'classes': hard,
                            'sub_history': r_hard.get('history', []),
                            'sub_time': r_hard.get('time', 0.0)})
        if easy:
            print(f"  [radapt-router] LIGHT path (LUV) on {len(easy)} classes")
            r_easy = self.light.unlearn(easy, retain_train_loader, forget_train_loader, **kw)
            history.append({'path': 'LIGHT', 'classes': easy,
                            'sub_history': r_easy.get('history', []),
                            'sub_time': r_easy.get('time', 0.0)})

        return {
            'history': history,
            'time': time.time() - start,
            'method': 'radapt_router',
            'routing_decisions': decisions,
            'threshold': thresh,
        }
