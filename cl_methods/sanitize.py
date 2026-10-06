"""Protocol-mandated CL-state sanitization at unlearning events.

Every stateful CL method caches knowledge OUTSIDE the live model: prototype
stores, Gaussian class stats, frozen teacher snapshots (distillation), EWC
parameter anchors, generative-replay conditioning, raw-exemplar buffers. An
unlearning operator that only edits the live model leaves the forgotten class
alive in these caches, and the NEXT CL event replays/distills/anchors it right
back into the model ("relapse"). Two mechanisms:

  (a) class-keyed caches (prototypes / gaussians / buffers / replay class lists)
      still contain the forget class -> it is literally re-trained;
  (b) frozen teachers and EWC anchors still point at the PRE-unlearning model
      -> distillation / the EWC penalty actively pull the weights back toward
      a model that knows the forgotten class.

The benchmark protocol therefore requires, at every UL event, AFTER the UL
operator has run:
  1. purge class-keyed state of the forget classes;
  2. re-snapshot every frozen teacher/anchor from the CURRENT (post-unlearning)
     model — keeps each method's semantics (pin to the latest reference state)
     while making that reference the unlearned one;
  3. ban forget classes from generative-replay sampling (functional deletion;
     the generator's weights are handled by the UL operator's scope);
  4. drop forget-class rows from raw-exemplar buffers.

Centralized attribute-based implementation so all methods share one audited
code path; disable with --sanitize_cl_state 0 for the relapse ablation.
"""
from copy import deepcopy

# dicts keyed by (int) class id
CLASS_KEYED_ATTRS = ('protos', 'global_protos', 'prev_protos', 'gaussians',
                     '_proto_anchor')
# (bottoms_attr, top_attr) frozen-teacher snapshot pairs
TEACHER_ATTRS = (('old_bottoms', 'old_top'), ('_old_bottoms', '_old_top'))


def _frozen_copy_bottoms(trainer):
    nb = [deepcopy(b).eval() for b in trainer.bottoms]
    for ob in nb:
        for p in ob.parameters():
            p.requires_grad = False
    return nb


def _frozen_copy_top(trainer):
    nt = deepcopy(trainer.top_model).eval()
    for p in nt.parameters():
        p.requires_grad = False
    return nt


def sanitize_cl_state(cl_method, trainer, forget_classes):
    """Apply the protocol to any CL method; returns a report list for logging."""
    fc = {int(c) for c in forget_classes}
    report = []

    # 1. class-keyed caches
    for attr in CLASS_KEYED_ATTRS:
        d = getattr(cl_method, attr, None)
        if isinstance(d, dict) and d:
            kept = {c: v for c, v in d.items() if int(c) not in fc}
            if len(kept) != len(d):
                setattr(cl_method, attr, kept)
                report.append(f'{attr}:-{len(d) - len(kept)}cls')

    # 2. frozen teachers / drift anchors -> re-snapshot post-unlearning model
    for battr, tattr in TEACHER_ATTRS:
        if getattr(cl_method, battr, None) is not None:
            setattr(cl_method, battr, _frozen_copy_bottoms(trainer))
            report.append(f'{battr}:refreshed')
        if getattr(cl_method, tattr, None) is not None:
            setattr(cl_method, tattr, _frozen_copy_top(trainer))
            report.append(f'{tattr}:refreshed')

    # 2b. EWC-style parameter anchors -> re-anchor to current params (otherwise
    # the quadratic penalty pulls weights back toward the pre-unlearning model).
    # The Fisher magnitudes are kept: they are a curvature estimate, and
    # re-estimating them is the UL operator's cost to pay, not the protocol's.
    op = getattr(cl_method, 'old_params', None)
    if isinstance(op, list) and any(op):
        cl_method.old_params = [
            {n: p.data.clone().cpu() for n, p in b.named_parameters()}
            for b in trainer.bottoms]
        report.append('old_params:re-anchored')

    # 3. generative replay ban (TARGET reads self.forgotten in its replay loop;
    # class ids stay in task_classes because they index the generator's
    # conditioning table — they are just never sampled again)
    if hasattr(cl_method, 'generators'):
        banned = set(getattr(cl_method, 'forgotten', set())) | fc
        cl_method.forgotten = banned
        report.append(f'replay-ban:+{sorted(fc)}')

    # 4. raw-exemplar buffers
    buf = getattr(cl_method, 'buffer', None)
    if buf is not None:
        purge = getattr(buf, 'purge_classes', None)
        if callable(purge):                         # buffer-owned atomic contract
            removed = purge(fc)
            if removed:
                report.append(f'buffer:-{removed}ex')
        elif hasattr(buf, 'lb'):                     # der_pp flat reservoir
            keep = [i for i, y in enumerate(buf.lb) if int(y) not in fc]
            removed = len(buf.lb) - len(keep)
            if removed:
                buf.ex = [buf.ex[i] for i in keep]
                buf.lb = [buf.lb[i] for i in keep]
                buf.lg = [buf.lg[i] for i in keep]
                report.append(f'buffer:-{removed}ex')
        elif hasattr(buf, 'data') and isinstance(buf.data, dict):  # er per-class
            removed = sum(len(buf.data.get(c, [])) for c in list(buf.data) if int(c) in fc)
            for c in list(buf.data):
                if int(c) in fc:
                    buf.data.pop(c, None)
                    if hasattr(buf, 'seen_count'):
                        buf.seen_count.pop(c, None)
            if removed:
                report.append(f'buffer:-{removed}ex')

    return report
