"""Locked, inference-only ISOLET F0 versus iCaRL error diagnostic."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

STUDY = Path('/brf/users/yangxx/VFCL-results/allparty-fim-final-attempt-v1-20261029')
OUTPUT = Path('/brf/users/yangxx/VFCL-results/isolet-f0-icarl-error-diagnostic-v1-20261030')
MANIFEST_SHA = 'de1eabbecfcd1fcb7c1c38b5d61cc36329423931e06a781d9eae9f843ed39528'
CODE_COMMIT = 'd848b22889765e15264f3e1335481396402e93a8'
ESET = frozenset((1, 2, 3, 4, 6, 15, 19, 21, 25))
ERROR_TYPES = ('within_task', 'old_to_new', 'old_to_other_old_task', 'new_to_old')
SEEDS = (80, 81, 82, 83, 84)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_type(truth, prediction, tasks):
    if truth == prediction:
        raise ValueError('correct prediction has no error type')
    owner = {label: task for task, classes in tasks.items() for label in classes}
    if truth not in owner or prediction not in owner:
        raise ValueError('predicted or true class is outside the task schedule')
    true_task, predicted_task = owner[truth], owner[prediction]
    last = max(tasks)
    if true_task == predicted_task:
        return 'within_task'
    if true_task != last and predicted_task == last:
        return 'old_to_new'
    if true_task != last and predicted_task != last:
        return 'old_to_other_old_task'
    return 'new_to_old'


def paired_excess(truth, f0, icarl, positions):
    only_f0_wrong = sum(f0[i] != truth[i] and icarl[i] == truth[i]
                        for i in positions)
    only_icarl_wrong = sum(f0[i] == truth[i] and icarl[i] != truth[i]
                           for i in positions)
    return only_f0_wrong, only_icarl_wrong, only_f0_wrong - only_icarl_wrong


def sign_pattern(values):
    if len(values) != 5:
        raise ValueError('the diagnostic requires all five seeds')
    if all(value > 0 for value in values):
        return 'f0_deficit'
    if all(value < 0 for value in values):
        return 'f0_advantage'
    return 'mixed_or_zero'


def checked_jobs():
    manifest = STUDY / 'manifests/stage-a/manifest.jsonl'
    root = STUDY / 'development-a'
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError('stage-A manifest SHA changed')
    queue = json.loads((root / 'QUEUE_COMPLETE.json').read_text())
    attempts_path = root / 'ATTEMPTS.jsonl'
    if ((root / 'QUEUE_FAILURE.json').exists() or
            queue.get('status') != 'complete' or
            queue.get('successes') != 65 or
            queue.get('accounted_for') != 65 or
            queue.get('manifest_sha256') != MANIFEST_SHA or
            queue.get('code_commit') != CODE_COMMIT or
            queue.get('attempts_sha256') != digest(attempts_path)):
        raise ValueError('stage A was not completely sealed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 65 or len({r['job_key'] for r in rows}) != 65:
        raise ValueError('stage-A allowlist is incomplete')
    events = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    if len(events) != 130:
        raise ValueError('stage-A attempt ledger is incomplete')
    by_key = {row['job_key']: [] for row in rows}
    for event in events:
        if event.get('job_key') not in by_key:
            raise ValueError('stage-A ledger contains an unlisted job')
        by_key[event['job_key']].append(event)
    uuids = {0: 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
             1: 'GPU-6569e526-43ea-74e1-264e-4b66417181e1'}
    cache = {}
    def pinned_digest(path):
        path = Path(path)
        if path not in cache:
            cache[path] = digest(path)
        return cache[path]
    selected = {}
    for row in rows:
        key = row['job_key']
        pair = by_key[key]
        if (len(pair) != 2 or pair[0].get('kind') != 'start' or
                pair[1].get('kind') != 'finish' or
                pair[0].get('attempt') != 1 or
                pair[1].get('attempt') != 1 or
                pair[0].get('config_sha256') != row['config_sha256'] or
                pair[0].get('gpu_index') != row['physical_gpu'] or
                pair[0].get('gpu_uuid') != uuids[row['physical_gpu']] or
                pair[1].get('classification') != 'success' or
                pair[1].get('returncode') != 0):
            raise ValueError(f'stage-A attempt differs: {key}')
        config_path = manifest.parent / row['config_relpath']
        if digest(config_path) != row['config_sha256']:
            raise ValueError(f'config SHA changed: {key}')
        config = json.loads(config_path.read_text())
        job_root = Path(config['output_dir'])
        gate_path = job_root / 'attempt-01/JOB_SUCCESS.json'
        if (digest(gate_path) != pair[1].get('job_success_sha256') or
                digest(job_root / 'attempt-01.log') != pair[1].get('log_sha256')):
            raise ValueError(f'job gate or log differs from ledger: {key}')
        if key.startswith('isolet/f0/0.25/') or key.startswith('isolet/icarl/none/'):
            for field, config_field in (
                    ('source_config_sha256', 'development_source_config_path'),
                    ('source_data_sha256', 'development_source_data_path'),
                    ('holdout_sha256', 'development_holdout_manifest'),
                    ('validation_manifest_sha256', 'pilot_paired_validation_manifest')):
                if pinned_digest(config[config_field]) != row[field]:
                    raise ValueError(f'stage-A source differs: {key}, {field}')
            run = job_root / 'attempt-01'
            gate = json.loads(gate_path.read_text())
            checkpoint = run / 'checkpoints/formal_final.pt'
            readout_path = run / 'HOLDOUT_READOUT.json'
            if (gate.get('status') != 'success' or
                    gate.get('job_key') != key or
                    gate.get('stage') != 'A' or
                    gate.get('test_accessed') is not False or
                    gate.get('manifest_sha256') != MANIFEST_SHA or
                    digest(checkpoint) != gate.get('checkpoint_sha256') or
                    digest(readout_path) != gate.get('readout_sha256')):
                raise ValueError(f'sealed job evidence changed: {key}')
            selected[key] = (config, checkpoint, json.loads(readout_path.read_text()))
    expected = {f'isolet/{method}/{"0.25" if method == "f0" else "none"}/{seed}'
                for method in ('f0', 'icarl') for seed in SEEDS}
    if set(selected) != expected:
        raise ValueError('the ten paired ISOLET jobs are incomplete')
    return selected


def infer(config, checkpoint, scratch):
    import torch
    import runner
    from cl_methods import get_cl_method
    from models import build_models
    from run_resource_lite_pilot import PilotDataset
    from vfl_trainer import VFLTrainer

    args = SimpleNamespace(**{**config, 'device': 'cpu', 'output_dir': str(scratch)})
    dataset = PilotDataset(args)
    locked = json.loads(Path(args.development_holdout_manifest).read_text())
    if (dataset.holdout_manifest['sha256'] != locked['sha256'] or
            dataset.holdout_indices != set(locked['ordered_indices'])):
        raise ValueError('holdout identity changed')
    payload = runner._decode_checkpoint_value(torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    tasks = {int(task): list(classes)
             for task, classes in payload['seen_task_classes'].items()}
    if (len(tasks) != 13 or set(tasks) != set(range(13)) or
            sorted(label for classes in tasks.values() for label in classes)
            != list(range(26)) or
            '|'.join(','.join(str(label) for label in tasks[task])
                     for task in range(13)) != config['custom_tasks']):
        raise ValueError('ISOLET final task schedule differs')
    bottoms, top = build_models(args)
    trainer = VFLTrainer(bottoms, top, args)
    trainer.load_state(payload['trainer_state'])
    for model in (*trainer.bottoms, trainer.top_model):
        model.eval()
    view = getattr(dataset, 'validationset', dataset.trainset)
    ids = [i for i in dataset.get_class_indices(view, range(26))
           if i in dataset.holdout_indices]
    loader = dataset.holdout_loader(list(range(26)))
    with torch.no_grad():
        if config['cl_method'] == 'icarl':
            method = get_cl_method('icarl', trainer, args)
            method.load_state(payload['cl_state'])
            classes, means = method._class_means()
            if classes.tolist() != list(range(26)):
                raise ValueError('iCaRL NME does not cover 26 classes')
            scores, labels = [], []
            for examples, target in loader:
                features = method._features(examples, record_comm=False)
                distances = torch.cdist(features, means)
                scores.append((-distances).cpu())
                labels.extend(target.tolist())
            scores = torch.cat(scores)
        else:
            scores, target = trainer.collect_logits(loader)
            scores = scores.cpu()
            labels = target.tolist()
    predictions = scores.argmax(dim=1).tolist()
    if (len(ids) != 780 or len(labels) != len(ids) or
            any(int(view.targets[i]) != label for i, label in zip(ids, labels))):
        raise ValueError('sample IDs or labels differ from the locked holdout')
    return {'ids': ids, 'truth': labels, 'pred': predictions,
            'scores': scores.tolist() if config['cl_method'] != 'icarl' else None,
            'tasks': tasks, 'holdout_sha': dataset.holdout_manifest['sha256']}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'empty diagnostic table: {path.name}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(paired):
    tasks = paired[80]['f0']['tasks']
    reference_ids = paired[80]['f0']['ids']
    reference_truth = paired[80]['f0']['truth']
    last = max(tasks)
    owner = {label: task for task, classes in tasks.items() for label in classes}
    output = {name: [] for name in ('task', 'error_type', 'class', 'sample', 'bias')}
    group_rows = {name: [] for name in
                    ('old', 'final', *ERROR_TYPES, 'e_set', 'non_e_set')}
    sample = {}
    for seed in SEEDS:
        f, i = paired[seed]['f0'], paired[seed]['icarl']
        if (f['ids'] != i['ids'] or f['truth'] != i['truth'] or
                f['ids'] != reference_ids or f['truth'] != reference_truth or
                f['tasks'] != i['tasks'] or f['tasks'] != tasks or
                f['holdout_sha'] != i['holdout_sha']):
            raise ValueError('paired method/sample/task alignment differs')
        ids, truth, fp, ip = f['ids'], f['truth'], f['pred'], i['pred']
        for task, classes in tasks.items():
            positions = [j for j, label in enumerate(truth) if label in classes]
            a, b, d = paired_excess(truth, fp, ip, positions)
            fc = sum(fp[j] == truth[j] for j in positions)
            ic = sum(ip[j] == truth[j] for j in positions)
            output['task'].append(dict(seed=seed, task=task, classes=','.join(map(str, classes)),
                n=len(positions), f0_correct=fc, icarl_correct=ic,
                f0_accuracy=fc / len(positions), icarl_accuracy=ic / len(positions),
                delta_pp=100 * (fc - ic) / len(positions),
                f0_only_wrong=a, icarl_only_wrong=b, net_f0_excess=d))
        for label in range(26):
            positions = [j for j, value in enumerate(truth) if value == label]
            a, b, d = paired_excess(truth, fp, ip, positions)
            fc = sum(fp[j] == label for j in positions)
            ic = sum(ip[j] == label for j in positions)
            output['class'].append(dict(seed=seed, class_id=label, letter=chr(65 + label),
                group='e_set' if label in ESET else 'non_e_set', n=len(positions),
                f0_correct=fc, icarl_correct=ic,
                f0_accuracy=fc / len(positions), icarl_accuracy=ic / len(positions),
                delta_pp=100 * (fc - ic) / len(positions),
                f0_only_wrong=a, icarl_only_wrong=b, net_f0_excess=d))
        for name, positions in (
            ('old', [j for j, y in enumerate(truth) if owner[y] != last]),
            ('final', [j for j, y in enumerate(truth) if owner[y] == last]),
            ('e_set', [j for j, y in enumerate(truth) if y in ESET]),
            ('non_e_set', [j for j, y in enumerate(truth) if y not in ESET]),
        ):
            a, b, d = paired_excess(truth, fp, ip, positions)
            group_rows[name].append(dict(seed=seed,
                f0_only_wrong=a, icarl_only_wrong=b, net=d,
                f0_errors=sum(fp[j] != truth[j] for j in positions),
                icarl_errors=sum(ip[j] != truth[j] for j in positions)))
        for name in ERROR_TYPES:
            f_errors = [j for j, y in enumerate(truth) if fp[j] != y and
                        error_type(y, fp[j], tasks) == name]
            i_errors = [j for j, y in enumerate(truth) if ip[j] != y and
                        error_type(y, ip[j], tasks) == name]
            f_only = sum(ip[j] == truth[j] for j in f_errors)
            i_only = sum(fp[j] == truth[j] for j in i_errors)
            d = f_only - i_only
            group_rows[name].append(dict(seed=seed,
                f0_only_wrong=f_only, icarl_only_wrong=i_only, net=d,
                f0_errors=len(f_errors), icarl_errors=len(i_errors)))
            output['error_type'].append(dict(seed=seed, error_type=name,
                f0_errors=len(f_errors), icarl_errors=len(i_errors),
                f0_only_wrong=f_only, icarl_only_wrong=i_only, net_f0_excess=d))
        for j, source_id in enumerate(ids):
            rec = sample.setdefault(source_id, {'truth': truth[j], 'f0_only_wrong': 0,
                'icarl_only_wrong': 0, 'both_wrong': 0, 'both_correct': 0})
            if rec['truth'] != truth[j]:
                raise ValueError('sample label changed across seeds')
            state = ('f0_only_wrong' if fp[j] != truth[j] and ip[j] == truth[j] else
                     'icarl_only_wrong' if fp[j] == truth[j] and ip[j] != truth[j] else
                     'both_wrong' if fp[j] != truth[j] else 'both_correct')
            rec[state] += 1
        for method, pred in (('f0', fp), ('icarl', ip)):
            n = len(truth)
            row = dict(seed=seed, method=method, n=n,
                true_old=sum(owner[y] != last for y in truth) / n,
                predicted_old=sum(owner[y] != last for y in pred) / n,
                true_new=sum(owner[y] == last for y in truth) / n,
                predicted_new=sum(owner[y] == last for y in pred) / n,
                mean_old_logit='', mean_new_logit='')
            if method == 'f0':
                scores = f['scores']
                old = [label for label in range(26) if owner[label] != last]
                new = tasks[last]
                row['mean_old_logit'] = sum(scores[j][label]
                    for j in range(n) for label in old) / (n * len(old))
                row['mean_new_logit'] = sum(scores[j][label]
                    for j in range(n) for label in new) / (n * len(new))
            output['bias'].append(row)
    if any(sum(rec[key] for key in ('f0_only_wrong', 'icarl_only_wrong',
                                   'both_wrong', 'both_correct')) != 5
           for rec in sample.values()):
        raise ValueError('sample recurrence is incomplete')
    output['sample'] = [dict(source_index=source_id, class_id=rec['truth'],
        letter=chr(65 + rec['truth']), **{k: rec[k] for k in
        ('f0_only_wrong', 'icarl_only_wrong', 'both_wrong', 'both_correct')})
        for source_id, rec in sorted(sample.items())]
    groups = {}
    for name, rows in group_rows.items():
        values = [row['net'] for row in rows]
        groups[name] = {'per_seed': rows, 'per_seed_net': values,
            'pooled_net': sum(values), 'pattern': sign_pattern(values),
            **{f'pooled_{field}': sum(row[field] for row in rows)
               for field in ('f0_only_wrong', 'icarl_only_wrong',
                             'f0_errors', 'icarl_errors')}}
    return output, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.output_root != OUTPUT or args.output_root.exists():
        raise ValueError('analysis output must be the new locked empty root')
    jobs = checked_jobs()
    if args.preflight_only:
        print(json.dumps({'status': 'preflight_passed', 'paired_jobs': len(jobs),
                          'sample_predictions_opened': False}))
        return
    paired, prediction_rows = {}, []
    with tempfile.TemporaryDirectory(prefix='isolet-error-inference-') as temporary:
        for seed in SEEDS:
            paired[seed] = {}
            for method in ('f0', 'icarl'):
                key = f'isolet/{method}/{"0.25" if method == "f0" else "none"}/{seed}'
                config, checkpoint, readout = jobs[key]
                result = infer(config, checkpoint, Path(temporary) / f'{method}-{seed}')
                correct = sum(y == p for y, p in zip(result['truth'], result['pred']))
                if (correct != readout['metrics']['cil_correct'] or
                        len(result['truth']) != readout['metrics']['count']):
                    raise ValueError(f'checkpoint inference did not reproduce {key}')
                paired[seed][method] = result
                prediction_rows.extend(dict(seed=seed, method=method,
                    source_index=source_id, truth=y, prediction=p)
                    for source_id, y, p in zip(result['ids'], result['truth'], result['pred']))
    if (sum(r['truth'] == r['prediction'] for r in prediction_rows if r['method'] == 'f0')
            != 3561 or
            sum(r['truth'] == r['prediction'] for r in prediction_rows if r['method'] == 'icarl')
            != 3567):
        raise ValueError('pooled correct counts differ from sealed stage A')
    tables, groups = analyze(paired)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        write_csv(args.output_root / f'{name}.csv', rows)
    with (args.output_root / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    summary = {'status': 'descriptive_diagnostic', 'manifest_sha256': MANIFEST_SHA,
        'source_commit': CODE_COMMIT, 'seed_count': 5, 'sample_count_per_seed': 780,
        'groups': groups, 'official_test_accessed': False,
        'new_training_authorized': False,
        'shared_holdout_warning': 'Seeds reuse the same holdout; 5/5 signs are descriptive.'}
    with (args.output_root / 'SUMMARY.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'output_root': str(args.output_root),
                      'groups': groups}, sort_keys=True))


if __name__ == '__main__':
    main()
