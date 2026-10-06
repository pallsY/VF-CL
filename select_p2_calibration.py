#!/usr/bin/env python3
"""Apply the frozen P2 simple-first calibration selection rule."""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


METHODS = (
    'beta_only', 'alpha_only', 'joint_alpha_beta', 'sequential_alpha_beta'
)


def _mean(values):
    return sum(values) / len(values)


def _method_summary(rows):
    reasons = []
    if len(rows) != 4 or len({row['run_key'] for row in rows}) != 4:
        reasons.append('requires exactly four unique runs')
    finite_values = []
    positive = True
    task_il_ok = True
    audits_ok = True
    for row in rows:
        raw, calibrated = row['raw'], row['calibrated']
        values = (
            raw['overall_accuracy'], calibrated['overall_accuracy'],
            raw['nll'], calibrated['nll'], raw['ece'], calibrated['ece'],
        )
        finite_values.extend(values)
        positive &= calibrated['overall_accuracy'] > raw['overall_accuracy']
        task_il_ok &= all(
            calibrated['task_il'][key] >= raw['task_il'][key] - 1e-6
            for key in raw['task_il']
        )
        audits_ok &= bool(row.get('audit', {}).get('passed', False))
    if not all(math.isfinite(float(value)) for value in finite_values):
        reasons.append('non-finite metric')
    if not positive:
        reasons.append('AA_final must improve in all four runs')
    if not task_il_ok:
        reasons.append('Task-IL invariant failed')
    if not audits_ok:
        reasons.append('audit failed')
    result = {'eligible': not reasons, 'reasons': reasons, 'run_count': len(rows)}
    if rows:
        result.update(
            mean_accuracy=_mean([r['calibrated']['overall_accuracy'] for r in rows]),
            mean_nll=_mean([r['calibrated']['nll'] for r in rows]),
            mean_ece=_mean([r['calibrated']['ece'] for r in rows]),
        )
    return result


def select_method(records):
    grouped = defaultdict(list)
    for row in records:
        if int(row.get('budget', -1)) == 25 and row.get('method') in METHODS:
            grouped[row['method']].append(row)
    summaries = {
        method: _method_summary(grouped.get(method, [])) for method in METHODS
    }
    eligible = [method for method in METHODS if summaries[method]['eligible']]
    if not eligible:
        return {
            'selected_method': None, 'status': 'P2_STOPPED',
            'reason': 'no method passed the frozen eligibility gate',
            'methods': summaries,
        }
    best = max(eligible, key=lambda method: summaries[method]['mean_accuracy'])
    best_mean = summaries[best]['mean_accuracy']
    if ('beta_only' in eligible and
            best_mean - summaries['beta_only']['mean_accuracy'] <= 0.002 + 1e-12):
        selected = 'beta_only'
        reason = 'beta-only is within 0.002 AA_final of the best eligible method'
    else:
        selected = 'sequential_alpha_beta' if 'sequential_alpha_beta' in eligible else best
        reason = 'retained saved sequential alpha+beta reference'
        for candidate in ('joint_alpha_beta', 'alpha_only'):
            if candidate != best or candidate not in eligible:
                continue
            if 'sequential_alpha_beta' not in eligible:
                selected, reason = candidate, 'best eligible method; sequential reference ineligible'
                break
            candidate_rows = {r['run_key']: r for r in grouped[candidate]}
            sequential_rows = {r['run_key']: r for r in grouped['sequential_alpha_beta']}
            all_positive = all(
                candidate_rows[key]['calibrated']['overall_accuracy'] >
                sequential_rows[key]['calibrated']['overall_accuracy']
                for key in sequential_rows
            )
            gain = (summaries[candidate]['mean_accuracy'] -
                    summaries['sequential_alpha_beta']['mean_accuracy'])
            quality_ok = (
                summaries[candidate]['mean_nll'] <=
                summaries['sequential_alpha_beta']['mean_nll'] + 1e-12 and
                summaries[candidate]['mean_ece'] <=
                summaries['sequential_alpha_beta']['mean_ece'] + 1e-12
            )
            if gain + 1e-12 >= 0.005 and all_positive and quality_ok:
                selected = candidate
                reason = (
                    f'{candidate} beats sequential by at least 0.005 in mean, '
                    'wins all four runs, and does not worsen NLL/ECE'
                )
            break
    return {
        'selected_method': selected,
        'status': 'P2_SUCCESS',
        'reason': reason,
        'best_eligible_method': best,
        'methods': summaries,
    }


def finalize_root(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob('*/ablation.json')):
        with open(path, encoding='utf-8') as handle:
            records.extend(json.load(handle))
    with open(root / 'P2_ALL_RESULTS.json', 'w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
    selection = select_method(records)
    with open(root / 'P2_SELECTION.json', 'w', encoding='utf-8') as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
    lines = [
        '# P2 Offline Calibration Ablation', '',
        f"Status: `{selection['status']}`", '',
        f"Selected method: `{selection['selected_method']}`", '',
        f"Reason: {selection['reason']}", '',
        '| Method | Eligible | Mean AA_final | Mean NLL | Mean ECE |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        item = selection['methods'][method]
        lines.append('| {} | {} | {} | {} | {} |'.format(
            method, item['eligible'],
            f"{item.get('mean_accuracy', float('nan')):.4f}",
            f"{item.get('mean_nll', float('nan')):.4f}",
            f"{item.get('mean_ece', float('nan')):.4f}",
        ))
    (root / 'P2_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    marker = root / selection['status']
    marker.touch()
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_json', nargs='?')
    parser.add_argument('output_json', nargs='?')
    parser.add_argument('--root')
    args = parser.parse_args()
    if args.root:
        print(json.dumps(finalize_root(args.root), indent=2, sort_keys=True))
        return
    if not args.results_json or not args.output_json:
        parser.error('results_json and output_json are required without --root')
    with open(args.results_json, encoding='utf-8') as handle:
        records = json.load(handle)
    selection = select_method(records)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
