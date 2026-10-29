"""Read-only, count-exact adjudication of the locked FIM study."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from party_drift_manifest import BASELINES, DATASETS
from replay_fim_manifest import POLICIES, SEEDS, expected_job_keys, load_approved_job
from replay_fim_queue import verify_job_gate
from run_party_drift_development import _sha256


def choose_policy(scores, baselines):
    if set(scores) != set(POLICIES) or set(baselines) != set(DATASETS):
        raise ValueError('stage-A scores are incomplete')
    eligible = [policy for policy in POLICIES if (
        set(scores[policy]) == set(DATASETS) and
        all(scores[policy][dataset] > baselines[dataset]
            for dataset in DATASETS))]
    if not eligible:
        return None
    return max(eligible, key=lambda policy: (
        min(scores[policy][dataset] - baselines[dataset]
            for dataset in DATASETS),
        -list(POLICIES).index(policy)))


def party_kd_supported(full, without):
    if set(full) != set(DATASETS) or set(without) != set(DATASETS):
        raise ValueError('stage-B PartyKD scores are incomplete')
    deltas = [full[dataset] - without[dataset] for dataset in DATASETS]
    return (sum(delta >= Fraction(1, 200) for delta in deltas) >= 2 and
            all(delta >= Fraction(-1, 200) for delta in deltas))


def _records(manifest, digest, result_root, stage, selected_policy):
    from run_replay_fim_development import validate_job_sources

    manifest, result_root = Path(manifest).resolve(), Path(result_root).resolve()
    expected = expected_job_keys(stage, selected_policy)
    queue = json.loads((result_root / 'QUEUE_COMPLETE.json').read_text(
        encoding='utf-8'))
    events_path = result_root / 'ATTEMPTS.jsonl'
    if ((result_root / 'QUEUE_FAILURE.json').exists() or
            queue.get('status') != 'complete' or
            queue.get('stage') != stage or
            queue.get('manifest_sha256') != digest or
            queue.get('accounted_for') != len(expected) or
            queue.get('successes') != len(expected) or
            queue.get('attempts_sha256') != _sha256(events_path)):
        raise ValueError('study queue is not completely sealed')
    events = [json.loads(line) for line in events_path.read_text(
        encoding='utf-8').splitlines()]
    if len(events) != len(expected) * 2:
        raise ValueError('study attempt ledger has missing or extra events')
    by_key = {key: [] for key in expected}
    for event in events:
        if event.get('job_key') not in by_key:
            raise ValueError('study attempt ledger contains unlisted job')
        by_key[event['job_key']].append(event)
    for key, pair in by_key.items():
        if (len(pair) != 2 or pair[0].get('kind') != 'start' or
                pair[1].get('kind') != 'finish' or
                pair[0].get('attempt') != 1 or
                pair[1].get('attempt') != 1 or
                pair[1].get('classification') != 'success' or
                pair[1].get('returncode') != 0):
            raise ValueError(f'study job is not a single success: {key}')
    checkout = Path(__file__).resolve().parent
    import subprocess
    code_commit = subprocess.check_output([
        'git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        text=True).strip()
    if queue.get('code_commit') != code_commit:
        raise ValueError('study adjudicator code differs from queue')
    rows = [json.loads(line) for line in manifest.read_text(
        encoding='utf-8').splitlines()]
    records = {}
    for row in rows:
        key = row['job_key']
        approved = load_approved_job(
            key, manifest, digest, code_commit=code_commit,
            stage=stage, selected_policy=selected_policy)
        config_path = (manifest.parent / row['config_relpath']).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8'))
        validate_job_sources(config, approved)
        run = Path(config['output_dir']) / 'attempt-01'
        gate_path = run / 'JOB_SUCCESS.json'
        if (by_key[key][0].get('config_sha256') != row['config_sha256'] or
                by_key[key][1].get('log_sha256') != _sha256(
                    Path(config['output_dir']) / 'attempt-01.log') or
                by_key[key][1].get('job_success_sha256') != _sha256(gate_path)):
            raise ValueError(f'study attempt hashes differ: {key}')
        verify_job_gate(run, key, digest, bool(config['party_drift_enabled']))
        readout = json.loads((run / 'HOLDOUT_READOUT.json').read_text(
            encoding='utf-8'))
        metrics = readout['metrics']
        correct, total = metrics['cil_correct'], metrics['count']
        if (readout.get('test_accessed') is not False or
                readout.get('job_key') != key or
                type(correct) is not int or type(total) is not int or
                not 0 <= correct <= total or
                total != config['num_classes'] * 30 or
                abs(float(metrics['cil']) - correct / total) > 1e-12):
            raise ValueError(f'study holdout score is invalid: {key}')
        expected_rule = ('final_bic_joint_alpha_beta'
                         if row['dataset'] == 'cifar100' else
                         'icarl_nme' if row['method'] == 'icarl' else
                         'model_head')
        if readout['readout_rule'] != expected_rule:
            raise ValueError('study readout rule differs from preregistration')
        records[key] = (correct, total)
    if set(records) != expected:
        raise ValueError('study records are incomplete')
    return records


def _mean(records, dataset, method, alpha):
    values = [records[f'{dataset}/{method}/{alpha}/{seed}'] for seed in SEEDS]
    totals = {total for _, total in values}
    if len(totals) != 1:
        raise ValueError('paired development holdout counts differ')
    return Fraction(sum(correct for correct, _ in values),
                    sum(total for _, total in values))


def _encode(value):
    return {'exact': str(value), 'pct': float(value * 100)}


def adjudicate(stage, manifest, digest, result_root, output, *,
               selected_policy=None, stage_a_decision=None):
    records = _records(manifest, digest, result_root, stage, selected_policy)
    if stage == 'A':
        baseline_methods = {
            dataset: {method: _mean(records, dataset, method, 'none')
                      for method in BASELINES[dataset]}
            for dataset in DATASETS
        }
        best = {dataset: max(values.values())
                for dataset, values in baseline_methods.items()}
        scores = {
            policy: {dataset: _mean(records, dataset, policy, '0.25')
                     for dataset in DATASETS}
            for policy in POLICIES
        }
        selected = choose_policy(scores, best)
        decision = {
            'status': 'selection_passed' if selected else 'direction_closed',
            'stage': stage, 'selected_policy': selected,
            'baseline_scores': {
                dataset: {method: _encode(value)
                          for method, value in values.items()}
                for dataset, values in baseline_methods.items()},
            'best_baseline': {dataset: _encode(value)
                              for dataset, value in best.items()},
            'scores': {policy: {dataset: _encode(value)
                                for dataset, value in values.items()}
                       for policy, values in scores.items()},
        }
    elif stage == 'B' and selected_policy in POLICIES and stage_a_decision:
        anchor = Path(stage_a_decision)
        prior = json.loads(anchor.read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in Path(manifest).read_text(
            encoding='utf-8').splitlines()]
        if (prior.get('status') != 'selection_passed' or
                prior.get('selected_policy') != selected_policy or
                any(row.get('stage_a_decision_sha256') != _sha256(anchor)
                    for row in rows)):
            raise ValueError('stage-B prior decision anchor differs')
        full = {dataset: Fraction(prior['scores'][selected_policy][dataset]['exact'])
                for dataset in DATASETS}
        without = {
            dataset: _mean(records, dataset,
                           f'no_partykd_{selected_policy}', '0.25')
            for dataset in DATASETS
        }
        supported = party_kd_supported(full, without)
        decision = {
            'status': ('party_kd_supported' if supported else
                       'direction_closed'),
            'stage': stage, 'selected_policy': selected_policy,
            'stage_a_decision_sha256': _sha256(anchor),
            'full_scores': {dataset: _encode(value)
                            for dataset, value in full.items()},
            'no_party_kd_scores': {dataset: _encode(value)
                                   for dataset, value in without.items()},
            'party_kd_delta': {dataset: _encode(full[dataset] - without[dataset])
                               for dataset in DATASETS},
        }
    else:
        raise ValueError('invalid adjudication stage or prior decision')
    decision.update({
        'manifest_sha256': digest,
        'result_root': str(Path(result_root).resolve()),
        'ledger_sha256': _sha256(Path(result_root) / 'ATTEMPTS.jsonl'),
        'official_test_accessed': False,
    })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(decision, stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write('\n')
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('A', 'B'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selected-policy', choices=tuple(POLICIES))
    parser.add_argument('--stage-a-decision', type=Path)
    args = parser.parse_args()
    print(json.dumps(adjudicate(args.stage, args.manifest,
                                args.manifest_sha256, args.result_root,
                                args.output,
                                selected_policy=args.selected_policy,
                                stage_a_decision=args.stage_a_decision),
                     sort_keys=True))


if __name__ == '__main__':
    main()
