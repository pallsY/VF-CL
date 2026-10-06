"""Read-only factorized-head evaluation on a frozen validation checkpoint."""

import argparse
import json

import torch

from adaptive_head_consolidation import _diagnostic_accuracy
from models import TopModel


@torch.no_grad()
def diagnose_checkpoint(path):
    """Compare Mixed and Factorized using a trusted VF-CL checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    bundle = checkpoint['cl_state']['adaptive_audit_bundle']
    state = bundle['installed_state']
    live_state = checkpoint['trainer_state']['top_model']
    if (set(state) != set(live_state)
            or any(not torch.equal(state[key], live_state[key]) for key in state)):
        raise ValueError('installed head differs from checkpoint head')
    weight = state['classifier.weight']
    model = TopModel(weight.shape[1], weight.shape[0], cosine='scale' in state)
    model.load_state_dict(state, strict=True)
    model.eval()
    embeddings = bundle['validation_embeddings']
    labels = bundle['validation_labels']
    classes = tuple(bundle['result']['ordered_classes'])
    columns = torch.tensor(classes, dtype=torch.long)
    heads = {}
    for name, log_p in (
            ('mixed', model(embeddings)),
            ('factorized', model.factorized_log_probabilities(embeddings))):
        task_accuracy, within_accuracy = _diagnostic_accuracy(
            log_p, labels, classes, bundle['task_classes']
        )
        predictions = columns.index_select(0, log_p.argmax(dim=1).cpu())
        heads[name] = {
            'class_il_accuracy': float((predictions == labels).double().mean()),
            'task_id_accuracy': task_accuracy,
            'within_task_class_accuracy': within_accuracy,
        }
    return {
        'dataset': bundle['result']['validation_manifest']['dataset'],
        'split': 'frozen_training_validation',
        'count': int(labels.numel()),
        **heads,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Trusted VF-CL adaptive checkpoint')
    args = parser.parse_args()
    print(json.dumps(diagnose_checkpoint(args.checkpoint), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
