"""Preflight gate for the one permitted follow-up formal stage."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from party_drift_manifest import DATASETS
from replay_fim_manifest import ALPHA, EPOCHS, POLICIES, RESULT_PREFIX


def formal_job_keys(policy):
    if policy not in POLICIES:
        raise ValueError('formal stage has no selected FIM policy')
    return {f'{dataset}/{policy}/0.25/{seed}'
            for dataset in DATASETS for seed in (42, 43, 44)}


def authorized_policy(stage_a_decision, stage_b_decision):
    """Return the sole selected policy only after both development gates pass."""
    a_path = Path(stage_a_decision)
    a_bytes = a_path.read_bytes()
    a = json.loads(a_bytes)
    b = json.loads(Path(stage_b_decision).read_bytes())
    policy = a.get('selected_policy')
    if (a.get('stage') != 'A' or a.get('status') != 'selection_passed'
            or b.get('stage') != 'B' or b.get('status') != 'party_kd_supported'
            or policy not in POLICIES or b.get('selected_policy') != policy
            or a.get('official_test_accessed') is not False
            or b.get('official_test_accessed') is not False
            or b.get('stage_a_decision_sha256') != hashlib.sha256(a_bytes).hexdigest()):
        raise ValueError('formal stage requires sealed, linked A and B passes')
    return policy


def build_formal_config(source, dataset, policy, seed, root, physical_gpu):
    """Keep the corresponding formal seed's full-data source configuration."""
    if (dataset not in DATASETS or policy not in POLICIES
            or seed not in (42, 43, 44) or physical_gpu not in (0, 1)
            or PurePosixPath(root) != RESULT_PREFIX / 'formal'
            or source.get('seed') != seed
            or source.get('cl_method') != 'proto_evolve'
            or source.get('formal_deferred_evaluation') is not True
            or source.get('epochs_per_task') != EPOCHS[dataset]
            or source.get('sample_replay_only') != 1
            or source.get('fim_freeze_frac') != 0.0
            or source.get('party_kd_enabled') != 1
            or source.get('party_kd_lambda') != 1.0
            or source.get('head_consolidation_samples_per_class') != 20
            or source.get('proto_replay_gradient_enabled') != 0
            or source.get('replay_mode') != 'prototype'):
        raise ValueError('formal config differs from locked full-data protocol')
    run = PurePosixPath(root) / dataset / policy / '0.25' / str(seed)
    config = {key: value for key, value in source.items()
              if not key.startswith('development_')
              and key != 'formal_source_provenance'}
    config.update({
        'device': 'cuda:0', 'fim_mask_policy': POLICIES[policy],
        'party_drift_enabled': 1, 'party_drift_preflight': 0,
        'development_nonfinite_guard': 1,
        'party_replay_arm': 'all_party', 'party_replay_alpha': ALPHA,
        'study_physical_gpu': physical_gpu,
        'output_dir': str(run), 'results_dir': str(run.parent),
        'exp_name': run.name,
    })
    return config
