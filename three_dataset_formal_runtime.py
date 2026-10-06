import contextlib
import runpy


@contextlib.contextmanager
def variant_runtime(variant: str):
    fixed = {
        'fixed_full': 'full',
        'fixed_bias': 'bias',
        'no_consolidation': 'no_consolidation',
    }
    ablations = {
        'fixed_half': 'fixed_half_ablation',
        'sample_mean_nll': 'sample_mean_nll',
    }
    if variant in fixed:
        from adaptive_tinyimagenet_heldout import _fixed_branch_runtime
        with _fixed_branch_runtime(fixed[variant]):
            yield
        return
    if variant in ablations:
        from adaptive_dual_branch_validation import _ablation_runtime
        with _ablation_runtime(ablations[variant]):
            yield
        return
    if variant != 'adaptive':
        raise ValueError(f'unknown internal variant: {variant}')
    yield


def execute_variant_main(variant: str) -> None:
    import config  # Bind canonical parser defaults before runtime substitutions.
    with variant_runtime(variant):
        runpy.run_path('main.py', run_name='__main__')
