#!/usr/bin/env python3
"""Frozen TinyImageNet P3 seed gate and final report."""
import argparse
import json
from pathlib import Path


def seed_gate(results):
    bic = results.get('bic_final') or {}
    paired = bic.get('paired') or {}
    raw, calibrated = paired.get('raw', {}), paired.get('calibrated', {})
    fractions_raw = raw.get('task_prediction_fraction', {})
    fractions_cal = calibrated.get('task_prediction_fraction', {})
    task_keys = sorted(fractions_raw, key=lambda key: int(key.split('_')[-1]))
    raw_accuracy = float(raw.get('overall_accuracy', float('nan')))
    calibrated_accuracy = float(calibrated.get('overall_accuracy', float('nan')))
    delta = calibrated_accuracy - raw_accuracy
    last_task = task_keys[-1] if task_keys else None
    checks = {
        'selected_method_frozen': bic.get('fit', {}).get('mode') == 'joint_alpha_beta',
        'aa_final_gain_at_least_0_005': delta + 1e-12 >= 0.005,
        'task_il_invariant': float(bic.get('task_il_max_abs_delta', float('inf'))) <= 1e-6,
        'last_task_fraction_reduced': bool(last_task) and
            float(fractions_cal.get(last_task, float('inf'))) <
            float(fractions_raw.get(last_task, float('-inf'))),
        'calibration_audit': bool(bic.get('calibration_audit', {}).get('passed')) and
            not bool(bic.get('calibration_audit', {}).get('test_used_for_fit', True)),
        'privacy_audit': bool(bic.get('privacy_audit', {}).get('passed')) and
            not bool(bic.get('privacy_audit', {}).get('test_used_for_fit', True)),
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'raw_aa_final': raw_accuracy,
        'calibrated_aa_final': calibrated_accuracy,
        'aa_final_delta': delta,
        'last_task': last_task,
        'last_task_fraction_raw': fractions_raw.get(last_task) if last_task else None,
        'last_task_fraction_calibrated': fractions_cal.get(last_task) if last_task else None,
    }


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pilot')
    parser.add_argument('--formal', nargs='*')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        gate = seed_gate(_read(args.pilot))
        (output / 'P3_SEED42_GATE.json').write_text(
            json.dumps(gate, indent=2, sort_keys=True), encoding='utf-8'
        )
        print(json.dumps(gate, sort_keys=True))
        return
    paths = dict(item.split('=', 1) for item in args.formal or [])
    gates = {key: seed_gate(_read(path)) for key, path in sorted(paths.items())}
    success = set(gates) == {'seed42', 'seed43'} and all(
        gate['passed'] for gate in gates.values()
    )
    report = {
        'status': 'P3_SUCCESS' if success else 'P3_STOPPED',
        'selected_method': 'joint_alpha_beta',
        'seeds': gates,
    }
    (output / 'P3_FINAL_RESULTS.json').write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding='utf-8'
    )
    lines = ['# P3 TinyImageNet Confirmation', '',
             f"Status: `{report['status']}`", '',
             'Frozen method: `joint_alpha_beta`', '',
             '| Seed | Raw AA_final | Calibrated AA_final | Gain | Passed |',
             '|---:|---:|---:|---:|---:|']
    for key, gate in gates.items():
        lines.append('| {} | {:.4f} | {:.4f} | {:+.4f} | {} |'.format(
            key.replace('seed', ''), gate['raw_aa_final'],
            gate['calibrated_aa_final'], gate['aa_final_delta'], gate['passed']))
    (output / 'P3_FINAL_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / report['status']).touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
