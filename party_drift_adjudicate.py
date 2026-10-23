"""Embargoed adjudication of the complete development matrix."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from party_drift_manifest import (
    ALPHAS, BASELINES, DATASETS, REPLAY_ARMS, SEEDS,
    expected_job_keys, load_approved_job,
)
from party_drift_audit_gate import verify_sealed_job


def validate_queue_complete(root, manifest_sha256):
    root = Path(root)
    gate = root / 'QUEUE_COMPLETE.json'
    if not gate.is_file():
        raise ValueError('development queue has no clean completion gate')
    record = json.loads(gate.read_text(encoding='utf-8'))
    events = root / 'ATTEMPTS.jsonl'
    digest = hashlib.sha256(events.read_bytes()).hexdigest()
    if (record.get('status') != 'complete' or
            record.get('accounted_for') != 183 or
            record.get('manifest_sha256') != manifest_sha256 or
            record.get('attempts_sha256') != digest):
        raise ValueError('development queue completion evidence differs')
    return record


def select_alpha(grid):
    if set(grid) != set(ALPHAS):
        raise ValueError('alpha grid differs from locked protocol')
    eligible = []
    for alpha, seeds in grid.items():
        if len(seeds) != 3 or any(item is None for item in seeds):
            continue
        counts = [int(total) for _, total in seeds]
        if len(set(counts)) != 1 or counts[0] <= 0 or any(
                not 0 <= int(correct) <= total for correct, total in seeds):
            raise ValueError('three-seed holdout counts differ')
        mean = Fraction(sum(int(correct) for correct, _ in seeds),
                        3 * counts[0])
        eligible.append((mean, -alpha, alpha))
    return max(eligible)[2] if eligible else None


def outcome_case(first_on_all_datasets, selector_on_all_datasets):
    if type(first_on_all_datasets) is not bool or type(selector_on_all_datasets) is not bool:
        raise ValueError('outcome flags must be booleans')
    return {
        (True, True): 'advance_to_confirmation',
        (True, False): 'replay_gain_no_selector',
        (False, True): 'selector_gain_no_first',
        (False, False): 'direction_closed',
    }[(first_on_all_datasets, selector_on_all_datasets)]


def _score(records, dataset, method, alpha):
    token = 'none' if alpha is None else str(alpha)
    scores = [records.get(f'{dataset}/{method}/{token}/{seed}')
              for seed in SEEDS]
    if any(item is None or item['status'] != 'success' for item in scores):
        return None
    counts = {item['count'] for item in scores}
    if len(counts) != 1:
        raise ValueError('paired holdout sample counts differ across seeds')
    return Fraction(sum(item['cil_correct'] for item in scores),
                    len(scores) * counts.pop())


def adjudicate(manifest_path, manifest_sha256, result_root, output):
    result_root, manifest_path, output = map(
        Path, (result_root, manifest_path, output))
    validate_queue_complete(result_root, manifest_sha256)
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError('manifest changed before adjudication')
    rows = [json.loads(line) for line in manifest_path.read_text(
        encoding='utf-8').splitlines()]
    if len(rows) != 183 or {row['job_key'] for row in rows} != expected_job_keys():
        raise ValueError('adjudication matrix is incomplete')
    commit = rows[0]['production_commit']
    events = [json.loads(line) for line in (result_root / 'ATTEMPTS.jsonl').read_text(
        encoding='utf-8').splitlines()]
    finished = {}
    for event in events:
        if event.get('kind') == 'finish':
            finished[event['job_key']] = event
    if set(finished) != expected_job_keys():
        raise ValueError('queue did not account for every listed job')
    readouts = {}
    for row in rows:
        key = row['job_key']
        load_approved_job(key, manifest_path, manifest_sha256,
                          production_commit=commit)
        config = json.loads((manifest_path.parent / row['config_relpath']).read_text(
            encoding='utf-8'))
        result = finished[key]
        status = result['classification']
        if status == 'training_anomaly':
            readouts[key] = {'status': status, 'cil_correct': None,
                             'count': None, 'cil': None}
            continue
        if status != 'success':
            raise ValueError('unresolved infrastructure or unknown job failure')
        run = Path(config['output_dir']) / f"attempt-{result['attempt']:02d}"
        gate = verify_sealed_job(
            run, key, manifest_sha256, replay=row['method'] in REPLAY_ARMS)
        if result.get('job_success_sha256') != hashlib.sha256(
                (run / 'JOB_SUCCESS.json').read_bytes()).hexdigest():
            raise ValueError('job success gate differs from append-only queue event')
        readout_path = run / 'HOLDOUT_READOUT.json'
        raw = readout_path.read_bytes()
        readout = json.loads(raw)
        metrics = readout['metrics']
        correct, count = metrics['cil_correct'], metrics['count']
        if (readout.get('job_key') != key or
                readout.get('test_accessed') is not False or
                type(correct) is not int or type(count) is not int or
                not 0 <= correct <= count or count !=
                (100 if row['dataset'] == 'cifar100' else
                 26 if row['dataset'] == 'isolet' else 101) * 30):
            raise ValueError('job holdout count or identity differs')
        readouts[key] = {
            'status': 'success', 'cil_correct': correct, 'count': count,
            'cil': metrics['cil'], 'readout_rule': readout['readout_rule'],
        }
        expected_rule = ('final_bic_joint_alpha_beta' if row['dataset'] ==
                         'cifar100' else 'icarl_nme' if row['method'] ==
                         'icarl' else 'model_head')
        if readout['readout_rule'] != expected_rule:
            raise ValueError('development readout differs from frozen method rule')
        if row['dataset'] == 'cifar100':
            details = readout.get('readout_details', {})
            raw_metrics = details.get('raw_metrics', {})
            expected_source = ('calibration' if row['method'] == 'er_ace'
                               else 'validation')
            if (details.get('bic_fit_source') != expected_source or
                    details.get('bic_fit_count') != 2500 or
                    raw_metrics.get('count') != count or
                    type(raw_metrics.get('cil_correct')) is not int or
                    not 0 <= raw_metrics['cil_correct'] <= count):
                raise ValueError('CIFAR raw/BiC paired development readout differs')
            readouts[key]['raw_cil_correct'] = raw_metrics['cil_correct']
            readouts[key]['raw_cil'] = raw_metrics['cil']

    selected = {}
    comparisons = {}
    inconclusive = False
    first_all = selector_all = True
    for dataset in DATASETS:
        selected[dataset] = {}
        for method in REPLAY_ARMS:
            grid = {alpha: [
                ((item['cil_correct'], item['count'])
                 if item['status'] == 'success' else None)
                for seed in SEEDS
                for item in [readouts[f'{dataset}/{method}/{alpha}/{seed}']]
            ] for alpha in ALPHAS}
            selected[dataset][method] = select_alpha(grid)
        alpha = selected[dataset]['drift']
        if alpha is None:
            inconclusive = True
            continue
        candidate = _score(readouts, dataset, 'drift', alpha)
        baseline = {method: _score(readouts, dataset, method, None)
                    for method in BASELINES[dataset]}
        random = _score(readouts, dataset, 'uniform', alpha)
        static = _score(readouts, dataset, 'static', alpha)
        if candidate is None or random is None or static is None or any(
                value is None for value in baseline.values()):
            inconclusive = True
            continue
        first = all(candidate > value for value in baseline.values())
        selector = candidate > random and candidate > static
        first_all = first_all and first
        selector_all = selector_all and selector
        comparisons[dataset] = {
            'candidate_alpha': alpha,
            'candidate_mean_exact': str(candidate),
            'baseline_mean_exact': {key: str(value)
                                    for key, value in baseline.items()},
            'candidate_minus_random_exact': str(candidate - random),
            'candidate_minus_static_exact': str(candidate - static),
            'first': first, 'selector_beats_both': selector,
        }
    decision = ('inconclusive_training_anomaly' if inconclusive else
                outcome_case(first_all, selector_all))
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'all_183_cells.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'dataset', 'method', 'alpha', 'seed', 'status',
            'cil_correct', 'count', 'cil', 'readout_rule',
            'raw_cil_correct', 'raw_cil',
        ])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item['job_key']):
            writer.writerow({
                key: value for key, value in {
                    **{key: row[key] for key in ('dataset', 'method', 'alpha', 'seed')},
                    **readouts[row['job_key']],
                }.items() if key in writer.fieldnames
            })
    result = {
        'status': 'complete_adjudication',
        'manifest_sha256': manifest_sha256,
        'all_cells_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'selected_alpha': selected, 'comparisons': comparisons,
        'decision': decision, 'official_test_accessed': False,
    }
    with (output / 'DECISION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(args.manifest, args.manifest_sha256,
                        args.result_root, args.output)
    print(json.dumps({'status': result['status'],
                      'decision': result['decision']}, sort_keys=True))


if __name__ == '__main__':
    main()
