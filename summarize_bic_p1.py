"""Frozen acceptance gate for the formal P1 BiC pilot."""
import argparse
import json
from pathlib import Path


def evaluate_pilot(result):
    final = result['bic_final']
    raw = final['paired']['raw']
    calibrated = final['paired']['calibrated']
    delta = calibrated['overall_accuracy'] - raw['overall_accuracy']
    task_il_losses = {
        key: calibrated['task_il'][key] - value
        for key, value in raw['task_il'].items()
    }
    task9_delta = (
        calibrated['task_prediction_fraction']['task_9']
        - raw['task_prediction_fraction']['task_9']
    )
    audit = final['calibration_audit']
    checks = {
        'delta_aa_final_at_least_0_005': delta >= 0.005 - 1e-12,
        'task_il_not_lower': min(task_il_losses.values()) >= -1e-6,
        'task9_prediction_fraction_lower': task9_delta < 0.0,
        'calibration_audit_passed': bool(audit.get('passed')),
        'test_not_used_for_fit': not bool(audit.get('test_used_for_fit')),
        'disabled_calibration_is_identity': final['disabled_identity_max_abs_diff'] == 0.0,
        'task_il_invariance_audit': final['task_il_max_abs_delta'] <= 1e-6,
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'delta_aa_final': delta,
        'task_il_deltas': task_il_losses,
        'task9_prediction_fraction_delta': task9_delta,
    }


def summarize_formal(results):
    runs = {}
    for label, result in sorted(results.items()):
        paired = result['bic_final']['paired']
        raw = paired['raw']['overall_accuracy']
        calibrated = paired['calibrated']['overall_accuracy']
        runs[label] = {
            'raw_aa_final': raw,
            'calibrated_aa_final': calibrated,
            'delta_aa_final': round(calibrated - raw, 12),
            'raw_task9_prediction_fraction': paired['raw']['task_prediction_fraction']['task_9'],
            'calibrated_task9_prediction_fraction': paired['calibrated']['task_prediction_fraction']['task_9'],
        }
    return {
        'runs': runs,
        'positive_runs': sum(item['delta_aa_final'] > 0 for item in runs.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot-results')
    mode.add_argument('--formal-results', nargs='+', metavar='LABEL=PATH')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.formal_results:
        results = {}
        for item in args.formal_results:
            label, path = item.split('=', 1)
            results[label] = json.loads(Path(path).read_text(encoding='utf-8'))
        summary = summarize_formal(results)
        (output / 'P1_FINAL.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8'
        )
        rows = '\n'.join(
            f"| {label} | {item['raw_aa_final']:.4f} | {item['calibrated_aa_final']:.4f} | {item['delta_aa_final']:+.4f} |"
            for label, item in summary['runs'].items()
        )
        (output / 'P1_FINAL_REPORT.md').write_text(
            '# Formal P1 BiC Results\n\n'
            '| Run | Raw AA_final | Calibrated AA_final | Delta |\n'
            '|---|---:|---:|---:|\n' + rows + '\n', encoding='utf-8'
        )
        (output / 'P1_SUCCESS').touch()
        return
    result = json.loads(Path(args.pilot_results).read_text(encoding='utf-8'))
    decision = evaluate_pilot(result)
    (output / 'P1_GATE.json').write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding='utf-8'
    )
    rows = '\n'.join(
        f"- {name}: {value}" for name, value in decision['checks'].items()
    )
    (output / 'P1_GATE_REPORT.md').write_text(
        '# P1 BiC Pilot Gate\n\n'
        f"- Passed: {decision['passed']}\n"
        f"- AA_final delta: {decision['delta_aa_final']:.6f}\n"
        f"- Task-9 prediction fraction delta: {decision['task9_prediction_fraction_delta']:.6f}\n\n"
        f"## Checks\n\n{rows}\n",
        encoding='utf-8',
    )
    (output / ('P1_GATE_PASS' if decision['passed'] else 'P1_STOPPED')).touch()


if __name__ == '__main__':
    main()
