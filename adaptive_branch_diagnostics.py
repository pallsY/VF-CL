"""Read-only validation comparison of four heads in an adaptive checkpoint."""

import argparse
import json

import torch

from adaptive_head_consolidation import (
    FrozenAdaptiveCandidates,
    _diagnostic_accuracy,
    adaptive_candidate_log_probabilities,
    mix_log_probabilities,
)
from models import TopModel


@torch.no_grad()
def diagnose_checkpoint(path):
    """Inspect a trusted VF-CL checkpoint without changing its audit bundle."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    bundle = checkpoint['cl_state']['adaptive_audit_bundle']
    result = bundle['result']
    classes = tuple(result['ordered_classes'])
    states = result['candidate_hashes']
    pre_state = bundle['pre_state']
    weight = pre_state['classifier.weight']
    pre = TopModel(weight.shape[1], weight.shape[0], cosine='scale' in pre_state)
    pre.load_state_dict(pre_state, strict=True)
    pre.eval()
    candidates = FrozenAdaptiveCandidates(
        pre_head_sha256=result['pre_head_sha256'],
        full_state=bundle['full_state'],
        bias_state=bundle['bias_state'],
        full_head_sha256=states['full'],
        bias_head_sha256=states['bias'],
        full_audit=None,
        bias_audit=None,
        ordered_classes=classes,
    )
    embeddings = bundle['validation_embeddings']
    labels = bundle['validation_labels']
    selected = torch.tensor(classes, dtype=torch.long)
    pre_p = torch.log_softmax(
        pre(embeddings).index_select(1, selected).to(torch.float64), dim=1
    )
    full_p, bias_p = adaptive_candidate_log_probabilities(
        pre, candidates, embeddings
    )
    probabilities = {
        'pre': pre_p,
        'full': full_p,
        'bias': bias_p,
        'mixed': mix_log_probabilities(full_p, bias_p, result['gate']['g']),
    }
    heads = {}
    for name, log_p in probabilities.items():
        task_accuracy, within_accuracy = _diagnostic_accuracy(
            log_p, labels, classes, bundle['task_classes']
        )
        predictions = selected.index_select(0, log_p.argmax(dim=1))
        heads[name] = {
            'class_il_accuracy': float((predictions == labels).to(torch.float64).mean()),
            'task_id_accuracy': task_accuracy,
            'within_task_class_accuracy': within_accuracy,
        }
    return {
        'dataset': result['validation_manifest']['dataset'],
        'split': 'frozen_training_validation',
        'count': int(labels.numel()),
        'heads': heads,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Trusted VF-CL adaptive checkpoint')
    args = parser.parse_args()
    print(json.dumps(diagnose_checkpoint(args.checkpoint), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
