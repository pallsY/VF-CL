"""Development-only synthetic proxy for final dual-head fitting FLOPs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.flop_counter import FlopCounterMode

from resource_flop_probe import (parse_task_classes,
                                 verify_profile_inputs)


def synthetic_replay_embeddings(classes, per_class, dimension):
    if (not classes or len(classes) != len(set(classes))
            or any(type(label) is not int or label < 0 for label in classes)
            or type(per_class) is not int or per_class < 1
            or type(dimension) is not int or dimension < 1):
        raise ValueError('synthetic replay class dimensions differ')
    generator = torch.Generator().manual_seed(0)
    return {label: torch.randn(per_class, dimension, generator=generator)
            for label in sorted(classes)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_head(job, source):
    config, _, commit = verify_profile_inputs(job, source, 1)
    if (config.get('cl_method') != 'proto_evolve'
            or config.get('head_consolidation_mode') != 'adaptive_dual_branch'
            or config.get('sample_replay_only') != 1
            or config.get('head_consolidation_samples_per_class') != 20
            or config.get('head_full_steps') != 500
            or config.get('head_bias_steps') != 600):
        raise ValueError('sample-only final head protocol differs')
    sys.path.insert(0, str(Path(source).resolve()))
    from adaptive_head_consolidation import fit_adaptive_candidates
    from models import build_models

    options = SimpleNamespace(**config)
    options.device = 'cpu'
    if config['data'] == 'tabvfl':
        options.party_col_ranges = (
            [(0, 256), (256, 512), (512, 1024), (1024, 1536)]
            if config['num_classes'] == 101 else
            [(0, 155), (155, 309), (309, 463), (463, 617)]
        )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    _bottoms, pre_top = build_models(options)
    classes_by_task = parse_task_classes(config)
    classes = [label for group in classes_by_task for label in group]
    replay = synthetic_replay_embeddings(
        classes, 20, pre_top.classifier.in_features)
    task_classes = {task: group for task, group
                    in enumerate(classes_by_task)}
    with FlopCounterMode(display=False) as counter:
        fit_adaptive_candidates(
            pre_top, replay, {}, task_classes, len(classes) * 20,
            config['seed'], 'cpu')
    return {
        'schema_version': 1,
        'job': '/'.join(Path(job).parts[-3:]),
        'producer_commit': commit,
        'probe_script_sha256': _sha(__file__),
        'torch_version': torch.__version__,
        'classes': len(classes),
        'per_class': 20,
        'embedding_dimension': pre_top.classifier.in_features,
        'full_steps': 500, 'bias_steps': 600,
        'head_fit_flop_proxy': counter.get_total_flops(),
        'coverage': ('torch FlopCounterMode supported ops for synthetic '
                     'Full/Bias fitting; validation encoding and gate solver excluded'),
        'input_origin': 'synthetic embeddings; no validation or test data',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = profile_head(args.job, args.source)
    body = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(body)
    print('RESOURCE_HEAD_FLOP_JSON=' + json.dumps(result, sort_keys=True))
