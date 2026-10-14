"""Commands for the separately audited nine-cell proposed-method comparison."""

from pathlib import Path

import replay11_matrix as baseline
import three_dataset_formal_registry as registry


def command_for(dataset, seed, *, python, worktree, results_dir,
                data_root, device):
    if dataset not in baseline.DATASETS or seed not in baseline.SEEDS:
        raise ValueError('adaptive comparison cell is outside the frozen matrix')
    spec = registry.FormalSpec(dataset, 'adaptive', seed)
    options = dict(registry.protocol_for(spec)['base_options'])
    options.update(
        data_path=str(data_root), results_dir=str(results_dir),
        device=str(device), exp_name='run', formal_deferred_evaluation=True,
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
