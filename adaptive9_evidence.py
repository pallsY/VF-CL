"""Independent, read-only evidence checks for the proposed method."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import replay11_evidence as shared
import replay11_matrix as baseline
from replay11_numeric import _producer_sources


TASKS = {'cifar100': 10, 'isolet': 13, 'upmc_food101': 10}
EPOCHS = {'cifar100': 50, 'isolet': 50, 'upmc_food101': 20}
APPROVED_PRODUCER_COMMIT = 'e2a559dc4945729848225cf21fdc95451762c231'
RUNTIME = {
    'python': '3.11.15', 'torch': '2.5.1+cu124',
    'torchvision': '0.20.1+cu124', 'numpy': '1.24.4',
    'scikit_learn': '1.9.0', 'pillow': '11.3.0',
    'cuda_runtime': '12.4',
}
HOST_PYTHON = {
    'c3080.upcbrfed.xyz': '/home/c3080/miniconda3/envs/hy3d/bin/python',
    'c5090.upcbrfed.xyz': '/var/tmp/vfcl-adaptive9-yangxx/venv/bin/python',
}
HOST_GPU_COUNT = {'c3080.upcbrfed.xyz': 2, 'c5090.upcbrfed.xyz': 1}


def verify_sample_only_state(checkpoint, config):
    """Require bounded raw memory and no persistent feature replay."""
    import torch
    from runner import _decode_checkpoint_value

    required = {
        'sample_replay_only': 1,
        'proto_replay_gradient_enabled': 0,
        'proto_sdc': False,
        'feat_distill_weight': 0.0,
        'head_consolidation_samples_per_class': 20,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('sample-only configuration differs')
    if type(checkpoint) is not dict:
        raise ValueError('sample-only checkpoint differs')
    state = _decode_checkpoint_value(checkpoint.get('cl_state'))
    tasks = checkpoint.get('seen_task_classes')
    if type(state) is not dict or type(tasks) is not dict:
        raise ValueError('sample-only checkpoint state differs')
    classes = [class_id for group in tasks.values() for class_id in group]
    raw = state.get('head_raw_replay')
    if (state.get('global_protos') != {}
            or state.get('prev_protos') != {}
            or type(raw) is not dict or len(classes) != len(set(classes))
            or set(raw) != set(classes)
            or any(not isinstance(value, torch.Tensor)
                   or value.ndim < 2 or value.shape[0] != 20
                   for value in raw.values())):
        raise ValueError('sample-only replay state differs')
    return len(classes)


def validate_runtime(runtime, interpreter):
    if (type(runtime) is not dict or set(runtime) != set(RUNTIME) | {'host'}
            or runtime.get('host') not in HOST_PYTHON
            or interpreter != HOST_PYTHON[runtime['host']]
            or any(runtime.get(key) != value for key, value in RUNTIME.items())):
        raise ValueError('adaptive training runtime differs')
    return runtime


def verify_adaptive_access_flow(run, expected_tasks, num_classes):
    """Bind the adaptive validation and test access markers to the flow log."""
    run = Path(run)
    access_dir = run / 'formal_access'
    if (access_dir.is_symlink() or not access_dir.is_dir()
            or {path.name for path in access_dir.iterdir()}
            != {'validation.consumed.json', 'test.consumed.json'}):
        raise ValueError('adaptive access inventory differs')
    classes = list(range(num_classes))
    common = {
        'event': 'first_iteration', 'event_idx': expected_tasks - 1,
        'task_id': expected_tasks - 1,
        'timeline_step': f'event_{expected_tasks - 1}_CIL',
        'classes': classes,
    }
    expected = {
        'validation': {
            **common,
            'loader_key': repr(('lambda_validation', tuple(classes))),
            'split': 'validation', 'phase': 'final_validation_pre_install',
        },
        'test': {
            **common, 'loader_key': repr(('test', tuple(classes))),
            'split': 'test', 'phase': 'final_test_post_install',
        },
    }
    for name, record in expected.items():
        marker = shared._json(run / 'formal_access'
                              / f'{name}.consumed.json')
        if (set(marker) != {'schema_version', 'status', 'access'}
                or marker['schema_version'] != 1
                or marker['status'] != 'consumed'
                or marker['access'] != record):
            raise ValueError(f'adaptive {name} access marker differs')
    accesses = []
    with shared._file(run / 'data_flow_audit.jsonl').open(
            encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError('adaptive validation access flow is malformed')
            if row.get('split') is not None:
                accesses.append(row)
    if accesses != [expected['validation'], expected['test']]:
        raise ValueError('adaptive validation access flow differs')
    return expected


def verify_run(run_dir, *, dataset, seed, expected_epochs,
               producer_commit, producer_root, data_root):
    """Recompute source, data, flow, checkpoint, and result identities."""
    if (dataset not in TASKS or seed not in baseline.SEEDS
            or type(expected_epochs) is not int or expected_epochs < 1):
        raise ValueError('adaptive audit cell is invalid')
    run = Path(run_dir)
    if run.is_symlink() or not run.is_dir():
        raise ValueError('adaptive run directory is missing or linked')
    config = shared._json(run / 'config.json')
    result = shared._json(run / 'results.json')
    expected = {
        'cl_method': 'proto_evolve', 'seed': seed,
        'num_tasks': TASKS[dataset], 'num_parties': 4,
        'epochs_per_task': expected_epochs,
        'head_consolidation_enabled': 1,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'formal_deferred_evaluation': True,
        'sample_replay_only': 1,
        'proto_replay_gradient_enabled': 0,
        'proto_sdc': False,
        'feat_distill_weight': 0.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('adaptive configuration differs')
    if any(result.get('config', {}).get(key) != config[key]
           for key in ('cl_method', 'seed', 'num_tasks')):
        raise ValueError('adaptive result configuration differs')
    if Path(config['data_path']) != Path(data_root):
        raise ValueError('adaptive data root differs')
    relative_vector = {
        'isolet': 'isolet/isolet_vfl.npz',
        'upmc_food101': baseline.UPMC_FOUR_CACHE,
    }.get(dataset)
    if (relative_vector is not None and Path(config.get('vector_npz', ''))
            != Path(data_root) / relative_vector):
        raise ValueError('adaptive vector cache differs')
    verified_data = shared._verify_authoritative_data(
        data_root, baseline.expected_data(dataset))
    frozen = shared._json(run / 'FORMAL_STATE_FROZEN.json')
    published = shared._json(run / 'FORMAL_EVALUATION_PUBLISHED.json')
    complete = shared._json(run / 'FORMAL_EVALUATION_COMPLETE.json')
    access = shared._json(run / 'formal_access/test.consumed.json')
    provenance = result.get('source_provenance')
    if (type(provenance) is not dict
            or provenance.get('source_commit') != producer_commit
            or frozen['identity'].get('source_commit') != producer_commit
            or frozen['identity'].get('source_provenance') != provenance):
        raise ValueError('adaptive source provenance differs')
    hashes = provenance.get('source_sha256')
    if type(hashes) is not dict or set(hashes) != set(shared.FORMAL_SOURCE_FILES):
        raise ValueError('adaptive source inventory differs')
    _producer_sources(producer_root, producer_commit, provenance)
    shared._verify_frozen_task_classes(frozen, config, TASKS[dataset])
    expected_protocol = shared._expected_protocol(config, provenance)
    if (frozen['identity'].get('protocol') != expected_protocol
            or frozen['identity'].get('protocol_sha256')
            != shared._canonical_sha(expected_protocol)):
        raise ValueError('adaptive frozen protocol differs')
    shared._verify_transaction(run, frozen, published, provenance, result)
    snapshots = frozen['identity'].get('snapshots')
    if type(snapshots) is not list or len(snapshots) != TASKS[dataset]:
        raise ValueError('adaptive snapshot inventory differs')
    for task, entry in enumerate(snapshots):
        for name, expected_path in (
                ('file', f'formal_snapshots/event_{task}_CIL.pt'),
                ('source_file', f'checkpoints/event_{task}_CIL.pt')):
            record = entry.get(name, {})
            if (record.get('path') != expected_path
                    or baseline.sha256_file(shared._file(run / expected_path))
                    != record.get('sha256')):
                raise ValueError('adaptive snapshot bytes differ')
        from adaptive_consolidation_audit import _safe_torch_load
        snapshot = _safe_torch_load(
            shared._file(run / f'formal_snapshots/event_{task}_CIL.pt'))
        verify_sample_only_state(snapshot, config)
    checkpoint = shared._file(run / 'checkpoints/formal_final.pt')
    result_path = shared._file(run / 'results.json')
    if (published.get('checkpoint', {}).get('sha256')
            != baseline.sha256_file(checkpoint)
            or published.get('results', {}).get('sha256')
            != baseline.sha256_file(result_path)):
        raise ValueError('adaptive publication bytes differ')
    flow_hash = shared._verify_data_flow(
        run, access.get('access'), TASKS[dataset])
    verify_adaptive_access_flow(
        run, TASKS[dataset], config['num_classes'])
    if (access.get('status') != 'consumed'
            or access.get('access', {}).get('split') != 'test'
            or access['access'].get('phase') != 'final_test_post_install'):
        raise ValueError('adaptive deferred test access differs')
    flow = shared._verify_training_flow_strict(
        run, config, data_root, TASKS[dataset])
    state = shared._verify_checkpoint_method_state(
        checkpoint, config, TASKS[dataset],
        frozen_final=frozen['identity']['final_checkpoint'],
        evaluation=complete['evaluation'],
        party_col_ranges=flow['party_col_ranges'],
        producer_root=producer_root)
    verify_sample_only_state(_safe_torch_load(checkpoint), config)
    metrics = baseline.metrics_from_history(
        result['task_acc_history'], TASKS[dataset], deferred=True)
    return {
        'admission': ('formal' if expected_epochs == EPOCHS[dataset]
                      else 'pilot'),
        'job': f'{dataset}/adaptive/{seed}',
        'producer_commit': producer_commit,
        'data_hashes': verified_data,
        'data_flow_sha256': flow_hash,
        'flow': flow, 'checkpoint_state': state,
        'results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'checkpoint_sha256': baseline.sha256_file(checkpoint),
        'metrics': metrics,
    }


def verify_pinned_job(job_dir, *, results_root, data_root,
                      producer_root, producer_commit):
    """Admit only one of the nine planned originals at its fixed locator."""
    job_dir = Path(os.path.abspath(job_dir))
    try:
        dataset, method, seed_text = job_dir.parts[-3:]
        seed = int(seed_text)
    except (ValueError, IndexError) as error:
        raise ValueError('pinned adaptive job path differs') from error
    expected = Path(os.path.abspath(
        Path(results_root) / dataset / 'adaptive' / str(seed)))
    if (dataset not in TASKS or method != 'adaptive'
            or seed not in baseline.SEEDS or job_dir != expected
            or any(part.is_symlink() for part in (job_dir, *job_dir.parents))):
        raise ValueError('pinned adaptive job path differs')
    if producer_commit != APPROVED_PRODUCER_COMMIT:
        raise ValueError('adaptive approved producer commit differs')
    plan = shared._json(job_dir / 'JOB_PLAN.json')
    success = shared._json(job_dir / 'JOB_SUCCESS.json')
    expected_plan_keys = {
        'dataset', 'method', 'seed', 'code_commit', 'command',
        'data_hashes', 'smoke', 'runtime',
    }
    if (set(plan) != expected_plan_keys or plan['dataset'] != dataset
            or plan['method'] != 'adaptive' or plan['seed'] != seed
            or plan['code_commit'] != producer_commit
            or plan['data_hashes'] != baseline.expected_data(dataset)
            or plan['smoke'] is not False
            or set(success) != expected_plan_keys | {
                'metrics', 'results_sha256', 'checkpoint_sha256'}
            or any(success.get(key) != value for key, value in plan.items())):
        raise ValueError('adaptive plan or success identity differs')
    command = plan['command']
    wrapper = ("from three_dataset_formal_runtime import "
               "execute_variant_main; execute_variant_main('adaptive')")
    if (type(command) is not list or len(command) < 5
            or command[1:3] != ['-c', wrapper]):
        raise ValueError('adaptive training entrypoint differs')
    validate_runtime(plan['runtime'], command[0])
    producer_root = Path(producer_root)
    head = subprocess.check_output(
        ['git', '-C', str(producer_root), 'rev-parse', 'HEAD'],
        text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(producer_root), 'status', '--porcelain'],
        text=True).strip()
    committed = subprocess.check_output([
        'git', '-C', str(producer_root), 'show',
        f'{producer_commit}:adaptive9_matrix.py'])
    if (head != producer_commit or dirty
            or hashlib.sha256(committed).hexdigest()
            != hashlib.sha256(
                (producer_root / 'adaptive9_matrix.py').read_bytes()).hexdigest()):
        raise ValueError('adaptive command generator source differs')
    script = (
        'import json,sys;from pathlib import Path;'
        'from adaptive9_matrix import command_for;'
        'print(json.dumps(command_for('
        'sys.argv[1],int(sys.argv[2]),python=Path(sys.argv[3]),'
        'worktree=Path(sys.argv[4]),results_dir=Path(sys.argv[5]),'
        'data_root=Path(sys.argv[6]),device=sys.argv[7])))'
    )
    generated = json.loads(subprocess.check_output([
        sys.executable, '-c', script, dataset, str(seed), command[0],
        str(producer_root), str(job_dir), str(data_root),
        command[command.index('--device') + 1],
    ], cwd=producer_root, text=True))
    if command != generated:
        raise ValueError('adaptive scientific command differs')
    checked = verify_run(
        job_dir / 'run', dataset=dataset, seed=seed,
        expected_epochs=EPOCHS[dataset],
        producer_commit=producer_commit, producer_root=producer_root,
        data_root=data_root)
    if (checked['admission'] != 'formal'
            or success['metrics'] != checked['metrics']
            or success['results_sha256'] != checked['results_sha256']
            or success['checkpoint_sha256'] != checked['checkpoint_sha256']):
        raise ValueError('adaptive formal success record differs')
    return checked
