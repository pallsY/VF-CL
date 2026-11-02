"""Independent hash audit before reading nine formal Class-IL results."""

import argparse
import json
import math
import statistics
from pathlib import Path

from f_off_formal_manifest import ROOT, STAGE_A_SHA, SOURCE_CONFIG_SHA256, SOURCE_ROOT, digest, expected_keys, pinned_checkout
from f_off_formal_queue import GPU_UUID, validate_rows


def formal_score(dataset, result):
    if dataset == 'cifar100':
        bic = result['bic_final']
        audit = bic['calibration_audit']
        if (bic['fit']['mode'] != 'joint_alpha_beta'
                or audit.get('passed') is not True
                or audit.get('overlap_count') != 0
                or audit.get('test_used_for_fit') is not False):
            raise ValueError('formal CIFAR BiC calibration is invalid')
        score = bic['paired']['calibrated']['overall_accuracy']
    else:
        score = result['cl_metrics']['AA_final']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('formal Class-IL score is invalid')
    return score


def summarize(values):
    if set(values) != {'cifar100', 'isolet', 'upmc_food101'}:
        raise ValueError('formal dataset results are incomplete')
    output = {}
    for dataset, seeds in values.items():
        if set(seeds) != {42, 43, 44} or any(
                type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1 for value in seeds.values()):
            raise ValueError('formal seed values are incomplete or invalid')
        ordered = [seeds[seed] for seed in (42, 43, 44)]
        output[dataset] = {
            'seeds': {str(seed): seeds[seed] for seed in (42, 43, 44)},
            'mean': statistics.mean(ordered),
            'sample_sd': statistics.stdev(ordered),
        }
    return output


def audit_and_summarize(manifest, manifest_sha, output):
    manifest = Path(manifest).resolve()
    root = Path(ROOT / 'formal')
    if (manifest != Path(ROOT / 'manifests/formal/manifest.jsonl')
            or digest(manifest) != manifest_sha):
        raise ValueError('formal manifest differs from sealed anchor')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    validate_rows(rows)
    queue_path = root / 'QUEUE_COMPLETE.json'
    ledger_path = root / 'ATTEMPTS.jsonl'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    pinned_checkout(queue.get('code_commit'))
    if ((root / 'QUEUE_FAILURE.json').exists()
            or queue.get('status') != 'complete'
            or queue.get('stage') != 'formal_followup'
            or queue.get('accounted_for') != 9
            or queue.get('successes') != 9
            or queue.get('manifest_sha256') != manifest_sha
            or queue.get('attempts_sha256') != digest(ledger_path)):
        raise ValueError('formal queue is not completely sealed')
    events = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    if len(events) != 18:
        raise ValueError('formal attempt ledger is incomplete')
    by_key = {key: [] for key in expected_keys()}
    for event in events:
        if event.get('job_key') not in by_key:
            raise ValueError('formal attempt ledger contains unlisted job')
        by_key[event['job_key']].append(event)

    # Validate every artifact and input before opening any results.json content.
    for row in rows:
        key = row['job_key']
        dataset, _, _, seed_text = key.split('/')
        if (row.get('code_commit') != queue['code_commit']
                or row.get('source_config_path') != str(
                    SOURCE_ROOT / 'full' / dataset / 'adaptive' / seed_text / 'run/config.json')
                or row.get('source_config_sha256') != SOURCE_CONFIG_SHA256[dataset][int(seed_text)]):
            raise ValueError(f'formal source/code identity differs: {key}')
        pair = by_key[key]
        if (len(pair) != 2 or pair[0].get('kind') != 'start'
                or pair[1].get('kind') != 'finish'
                or pair[0].get('attempt') != 1
                or pair[1].get('attempt') != 1
                or pair[0].get('gpu_index') != row['physical_gpu']
                or pair[0].get('gpu_uuid') != GPU_UUID[row['physical_gpu']]
                or pair[0].get('config_sha256') != row['config_sha256']
                or pair[1].get('classification') != 'success'
                or pair[1].get('returncode') != 0):
            raise ValueError(f'formal job attempt differs: {key}')
        config_path = (manifest.parent / row['config_relpath']).resolve()
        if (not config_path.is_relative_to((manifest.parent / 'configs').resolve())
                or digest(config_path) != row['config_sha256']):
            raise ValueError(f'formal config changed: {key}')
        for field, path_field in (
            ('source_config_sha256', 'source_config_path'),
            ('source_data_sha256', 'source_data_path'),
            ('validation_manifest_sha256', 'validation_manifest_path'),
        ):
            if digest(row[path_field]) != row[field]:
                raise ValueError(f'formal source changed: {key}/{field}')
        run = root / key / 'attempt-01'
        gate_path = run / 'JOB_SUCCESS.json'
        gate = json.loads(gate_path.read_text(encoding='utf-8'))
        if (digest(gate_path) != pair[1].get('job_success_sha256')
                or digest(root / key / 'attempt-01.log') != pair[1].get('log_sha256')
                or gate.get('status') != 'success' or gate.get('job_key') != key
                or gate.get('code_commit') != queue['code_commit']
                or gate.get('manifest_sha256') != manifest_sha):
            raise ValueError(f'formal success gate changed: {key}')
        for field, path in (
            ('results_sha256', run / 'results.json'),
            ('checkpoint_sha256', run / 'checkpoints/formal_final.pt'),
            ('published_sha256', run / 'FORMAL_EVALUATION_PUBLISHED.json'),
            ('complete_sha256', run / 'FORMAL_EVALUATION_COMPLETE.json'),
            ('sealed_sha256', run / 'FORMAL_EVALUATION_SEALED.json'),
            ('fim_mask_audit_sha256', run / 'FIM_MASK_AUDIT.json'),
            ('data_flow_sha256', run / 'data_flow_audit.jsonl'),
        ):
            if digest(path) != gate.get(field):
                raise ValueError(f'formal artifact changed: {key}/{field}')
        published = json.loads((run / 'FORMAL_EVALUATION_PUBLISHED.json').read_text())
        if (published.get('status') != 'published'
                or published.get('results', {}).get('sha256') != gate['results_sha256']
                or published.get('checkpoint', {}).get('sha256') != gate['checkpoint_sha256']):
            raise ValueError(f'formal publication differs: {key}')
        mask = json.loads((run / 'FIM_MASK_AUDIT.json').read_text())
        if (mask.get('policy') != 'off'
                or not mask.get('tasks')
                or any(party['frozen_names']
                       for task in mask['tasks'] for party in task['parties'])):
            raise ValueError(f'formal F-off mask differs: {key}')
        for task in mask['tasks']:
            snapshot = run / 'formal_snapshots' / f"event_{task['task_id']}_CIL.pt"
            if digest(snapshot) != task['snapshot_sha256']:
                raise ValueError(f'formal FIM snapshot changed: {key}')

    values = {'cifar100': {}, 'isolet': {}, 'upmc_food101': {}}
    for row in rows:
        result = json.loads((root / row['job_key'] / 'attempt-01/results.json').read_text())
        metric = formal_score(row['dataset'], result)
        values[row['dataset']][row['seed']] = metric
    summary = {
        'status': 'audited_formal_followup',
        'stage_a_decision_sha256': STAGE_A_SHA,
        'manifest_sha256': manifest_sha,
        'queue_sha256': digest(queue_path),
        'ledger_sha256': digest(ledger_path),
        'class_il_final': summarize(values),
        'interpretation': 'post_selection_followup_on_previously_seen_benchmark',
    }
    output = Path(output)
    if output != Path(ROOT / 'adjudication/FORMAL_SUMMARY.json'):
        raise ValueError('formal summary output differs from protocol')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_and_summarize(args.manifest,
                                         args.manifest_sha256, args.output),
                     sort_keys=True))
