"""Commands for the separately audited nine-cell proposed-method comparison."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from itertools import product
from types import SimpleNamespace

import replay11_matrix as baseline
import three_dataset_formal_registry as registry


def jobs():
    return tuple(product(baseline.DATASETS, ('adaptive',), baseline.SEEDS))


def runtime_identity():
    """Record the interpreter and packages that actually launch training."""
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision

    return {
        'host': platform.node(), 'python': platform.python_version(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': numpy.__version__, 'scikit_learn': sklearn.__version__,
        'pillow': PIL.__version__, 'cuda_runtime': str(torch.version.cuda),
    }


def command_for(dataset, seed, *, python, worktree, results_dir,
                data_root, device):
    if dataset not in baseline.DATASETS or seed not in baseline.SEEDS:
        raise ValueError('adaptive comparison cell is outside the frozen matrix')
    spec = registry.FormalSpec(dataset, 'adaptive', seed)
    options = dict(registry.protocol_for(spec)['base_options'])
    options.update(
        data_path=str(data_root), results_dir=str(results_dir),
        device=str(device), exp_name='run', formal_deferred_evaluation=True,
        sample_replay_only=1, proto_replay_gradient_enabled=0,
        proto_sdc=False, feat_distill_weight=0.0,
    )
    if dataset == 'upmc_food101':
        options['num_parties'] = 4
        options['vector_npz'] = str(Path(data_root) / baseline.UPMC_FOUR_CACHE)
    elif dataset == 'isolet':
        options['vector_npz'] = str(Path(data_root) / 'isolet/isolet_vfl.npz')
    command = [
        str(python), '-c',
        "from three_dataset_formal_runtime import execute_variant_main; "
        "execute_variant_main('adaptive')",
    ]
    for key, value in options.items():
        command.extend(('--' + key, registry._option_token(key, value)))
    return command


def validate_completed_run(dataset, seed, run_dir):
    run_dir = Path(run_dir)
    expected_tasks = {'cifar100': 10, 'isolet': 13,
                      'upmc_food101': 10}[dataset]
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    result = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    expected = {
        'seed': seed, 'num_tasks': expected_tasks, 'num_parties': 4,
        'cl_method': 'proto_evolve', 'head_consolidation_enabled': 1,
        'head_consolidation_mode': 'adaptive_dual_branch',
        'formal_deferred_evaluation': True,
        'sample_replay_only': 1,
        'proto_replay_gradient_enabled': 0,
        'proto_sdc': False,
        'feat_distill_weight': 0.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('adaptive run identity differs from frozen comparison')
    if dataset == 'upmc_food101' and not str(
            config.get('vector_npz', '')).endswith(baseline.UPMC_FOUR_CACHE):
        raise ValueError('adaptive UPMC run did not use the four-party cache')
    checkpoint = run_dir / 'checkpoints/formal_final.pt'
    from adaptive_consolidation_audit import prepare_formal_deferred_evaluation
    from runner import _safe_torch_load

    payload = _safe_torch_load(str(checkpoint))
    if payload.get('schema_version') != 4 or payload.get('protocol', {}).get(
            'formal_deferred_evaluation') is not True:
        raise ValueError('adaptive formal checkpoint protocol differs')
    state = payload.get('cl_state')
    if (type(state) is not dict or state.get('global_protos')
            or state.get('prev_protos')):
        raise ValueError('sample-only final checkpoint contains prototypes')
    raw = state.get('head_raw_replay')
    if (type(raw) is not dict or len(raw) != int(config['num_classes'])
            or any(getattr(samples, 'shape', (0,))[0] != 20
                   for samples in raw.values())):
        raise ValueError('sample-only final checkpoint raw memory differs')
    snapshots = [run_dir / 'formal_snapshots' / f'event_{task}_CIL.pt'
                 for task in range(expected_tasks)]
    transaction = prepare_formal_deferred_evaluation(
        args=SimpleNamespace(**config), snapshot_paths=snapshots,
        final_checkpoint=checkpoint,
        task_classes=payload['seen_task_classes'], output_dir=run_dir)
    if transaction['status'] != 'published':
        raise ValueError('adaptive deferred evaluation was not published')
    return baseline.metrics_from_history(
        result['task_acc_history'], expected_tasks, deferred=True)


def run_one(dataset, seed, *, results_root, data_root, device):
    if (dataset, 'adaptive', seed) not in jobs():
        raise ValueError('adaptive comparison cell is outside the nine jobs')
    results_root = Path(results_root)
    job_dir = results_root / dataset / 'adaptive' / str(seed)
    if job_dir.exists():
        raise FileExistsError(f'adaptive job already exists: {job_dir}')
    worktree = Path(__file__).resolve().parent
    commit = baseline._git_commit(worktree)
    baseline.verify_data(dataset, data_root)
    results_root.mkdir(parents=True, exist_ok=True)
    minimum_free = 2 * 1024 ** 3 if dataset == 'cifar100' else 512 * 1024 ** 2
    if shutil.disk_usage(results_root).free < minimum_free:
        raise RuntimeError('insufficient adaptive result storage')
    command = command_for(
        dataset, seed, python=Path(sys.executable), worktree=worktree,
        results_dir=job_dir, data_root=data_root, device=device)
    job_dir.mkdir(parents=True)
    plan = {
        'dataset': dataset, 'method': 'adaptive', 'seed': seed,
        'code_commit': commit, 'command': command,
        'data_hashes': baseline.expected_data(dataset), 'smoke': False,
        'runtime': runtime_identity(),
    }
    with (job_dir / 'JOB_PLAN.json').open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write('\n')
    env = dict(os.environ)
    env.update(VFCL_EXPERIMENT_PROFILE='formal', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONHASHSEED=str(seed),
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (job_dir / 'training.log').open('x', encoding='utf-8') as log:
        subprocess.run(command, cwd=worktree, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    run_dir = job_dir / 'run'
    metrics = validate_completed_run(dataset, seed, run_dir)
    success = {
        **plan, 'metrics': metrics,
        'results_sha256': baseline.sha256_file(run_dir / 'results.json'),
        'checkpoint_sha256': baseline.sha256_file(
            run_dir / 'checkpoints/formal_final.pt'),
    }
    with (job_dir / 'JOB_SUCCESS.json').open('x', encoding='utf-8') as handle:
        json.dump(success, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'run-one'))
    parser.add_argument('--dataset', choices=baseline.DATASETS)
    parser.add_argument('--seed', type=int, choices=baseline.SEEDS)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'check':
        for dataset in baseline.DATASETS:
            baseline.verify_data(dataset, args.data_root)
        print('adaptive9_data_verified')
    else:
        if (args.dataset is None or args.seed is None
                or args.results_root is None):
            parser.error('run-one needs dataset, seed, and results-root')
        print(json.dumps(run_one(
            args.dataset, args.seed, results_root=args.results_root,
            data_root=args.data_root, device=args.device), sort_keys=True))
