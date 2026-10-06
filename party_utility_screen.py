"""Read-only replay utility screen for saved vector VFL checkpoints."""

import torch
import torch.nn.functional as F

from data_utils import split_features


@torch.no_grad()
def replay_scores(old_trainer, new_trainer, replay, seen, frozen_weights, args):
    """Aggregate fixed-model party utility and next-task logit change by class."""
    seen = [int(c) for c in seen]
    parties = int(args.num_parties)
    if (args.aggregation != 'concat' or parties < 2 or not seen
            or len(set(seen)) != len(seen) or set(replay) != set(seen)
            or set(frozen_weights) != set(seen)):
        raise ValueError('screen requires exact seen-class replay and concat parties')
    if any(len(trainer.bottoms) != parties for trainer in (old_trainer, new_trainer)):
        raise ValueError('party count differs from model')

    rows = []
    for class_id in seen:
        inputs = replay[class_id]
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2
                or inputs.size(0) == 0 or not torch.isfinite(inputs).all()):
            raise ValueError('replay is empty or non-finite')
        parts = split_features(inputs.to(args.device), args)
        if len(parts) != parties:
            raise ValueError('replay party split is malformed')

        outputs = []
        for trainer in (old_trainer, new_trainer):
            embeddings = [bottom(part) for bottom, part in zip(trainer.bottoms, parts)]
            top = trainer.top_model
            weight = top.classifier.weight
            if (weight.size(1) % parties or top.cosine
                    or bool(top._adaptive_enabled)):
                raise ValueError('screen requires an ordinary linear concat head')
            width = weight.size(1) // parties
            per_party = [F.linear(embedding,
                                  weight[:, p * width:(p + 1) * width], None)
                         for p, embedding in enumerate(embeddings)]
            full = sum(per_party)
            if top.classifier.bias is not None:
                full = full + top.classifier.bias
            if (not torch.isfinite(full).all()
                    or not all(torch.isfinite(x).all() for x in per_party)
                    or not torch.allclose(full, top(trainer._aggregate(embeddings)),
                                          atol=1e-5, rtol=1e-5)):
                raise ValueError('party logits do not reconstruct the full head')
            outputs.append((per_party, full))

        old_party, old_full = outputs[0]
        new_party, _ = outputs[1]
        position = seen.index(class_id)
        target = torch.full((inputs.size(0),), position,
                            dtype=torch.long, device=old_full.device)
        baseline = F.cross_entropy(old_full[:, seen].double(), target,
                                   reduction='none')
        utilities = torch.stack([
            (F.cross_entropy((old_full - logits)[:, seen].double(), target,
                             reduction='none') - baseline).mean()
            for logits in old_party
        ])
        positive = utilities.clamp_min(0)
        fallback = bool(positive.sum() == 0)
        utility_weights = (torch.full_like(positive, 1 / parties) if fallback
                           else positive / positive.sum())
        frozen = torch.as_tensor(frozen_weights[class_id], dtype=torch.float64,
                                 device=old_full.device)
        if (frozen.shape != (parties,) or not torch.isfinite(frozen).all()
                or (frozen < 0).any() or frozen.sum() <= 0):
            raise ValueError('frozen contribution weights are malformed')
        frozen = frozen / frozen.sum()
        drift = torch.stack([
            (new_logits[:, class_id].double()
             - old_logits[:, class_id].double()).square().mean()
            for old_logits, new_logits in zip(old_party, new_party)
        ])
        if not torch.isfinite(drift).all() or not torch.isfinite(utilities).all():
            raise ValueError('party utility or drift is non-finite')
        uniform = torch.full_like(drift, 1 / parties)
        weights = {
            'utility': utility_weights,
            'uniform': uniform,
            'frozen': frozen,
            'shuffled': utility_weights.roll(1),
        }
        rows.append({
            'class_id': class_id,
            'replay_count': int(inputs.size(0)),
            'utility_fallback': fallback,
            'marginal_utility': utilities.tolist(),
            'utility_weights': utility_weights.tolist(),
            'party_logit_drift': drift.tolist(),
            **{name: float((weight * drift).sum())
               for name, weight in weights.items()},
        })
    return rows
