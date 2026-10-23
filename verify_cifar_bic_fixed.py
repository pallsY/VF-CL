"""Compare the unchanged CIFAR BiC fit on fixed calibration-only logits."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    from bic_calibration import fit_final_calibrator

    module = sys.modules['bic_calibration']
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError('BiC imported from the wrong checkout')
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(730042)
    logits = torch.randn(2500, 100, generator=generator)
    labels = torch.arange(2500) % 100
    task_classes = {task: list(range(task * 10, (task + 1) * 10))
                    for task in range(10)}
    calibrator, fit = fit_final_calibrator(
        logits, labels, task_classes, 'joint_alpha_beta', lr=0.05,
        steps=1000,
    )
    result = {
        'checkout': str(checkout),
        'bic_source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'calibrator_state': calibrator.state_dict(),
        'fit': fit,
        'calibrated_logits_sha256': hashlib.sha256(
            calibrator.apply(logits).contiguous().numpy().tobytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
