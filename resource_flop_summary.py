"""Weight development-only operator probes by audited optimizer steps."""

import argparse
import ast
import csv
import json
from pathlib import Path

from resource_flop_probe import parse_task_classes
from resource_ledger import file_sha256


SELECTED = (
    ('cifar100', 'sample', 'cifar100/adaptive/42',
     'e2a559dc4945729848225cf21fdc95451762c231',
     'adaptive-sample-only-fourparty-v1-20261017/full/cifar100/adaptive/42', 10),
    ('cifar100', 'erace', 'cifar100/er_ace/42',
     '467611a603c9604ad57861920495a3ad05612037',
     'replay11-paper-v3-20261013/full-3080/cifar100/er_ace/42', 10),
    ('isolet', 'sample', 'isolet/adaptive/42',
     'e2a559dc4945729848225cf21fdc95451762c231',
     'adaptive-sample-only-fourparty-v1-20261017/full/isolet/adaptive/42', 13),
    ('isolet', 'icarl', 'isolet/icarl/42',
     '467611a603c9604ad57861920495a3ad05612037',
     'replay11-paper-v3-20261013/full-3080/isolet/icarl/42', 13),
    ('upmc', 'sample', 'upmc_food101/adaptive/42',
     'e2a559dc4945729848225cf21fdc95451762c231',
     'adaptive-sample-only-fourparty-v1-20261017/full/upmc_food101/adaptive/42', 10),
    ('upmc', 'derpp', 'upmc_food101/der_pp/42',
     '467611a603c9604ad57861920495a3ad05612037',
     'replay11-paper-v3-20261013/full-3080/upmc_food101/der_pp/42', 10),
)
RESULTS = Path('/brf/users/yangxx/VFCL-results')


def profile_name(dataset, method, task):
    prefix = 'cifar' if dataset == 'cifar100' else dataset
    return f'{prefix}_{method}_t{task}.json'


def nominal_steps_per_task(flow_path, config):
    tasks = parse_task_classes(config)
    by_classes = {tuple(group): index for index, group in enumerate(tasks)}
    steps = {task: 0 for task in range(len(tasks))}
    seen = set()
    with Path(flow_path).open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('split') is not None:
                continue
            key = ast.literal_eval(record['loader_key'])
            if key[0] != 'train' or tuple(key[1]) not in by_classes:
                raise ValueError('profile training task loader differs')
            if record['loader_iteration'] != 0:
                continue
            task = by_classes[tuple(key[1])]
            identity = task, record['batch']
            if identity in seen:
                raise ValueError('duplicate first-pass training batch')
            seen.add(identity)
            steps[task] += 1
    if any(not count for count in steps.values()):
        raise ValueError('profile task first-pass batches are missing')
    return {task: count * config['epochs_per_task']
            for task, count in steps.items()}


def weighted_profile(profiles, steps):
    by_task = {row['profile_task']: row for row in profiles}
    if (len(by_task) != len(profiles) or set(by_task) != set(steps)
            or any(type(count) is not int or count < 1
                   for count in steps.values())):
        raise ValueError('profile task set differs from optimizer steps')
    scripts = {row['probe_script_sha256'] for row in profiles}
    if len(scripts) != 1:
        raise ValueError('profile script identity differs')
    flops = [row['flop_counter_total'] for row in profiles]
    if any(type(value) is not int or value < 1 for value in flops):
        raise ValueError('profile operator count differs')
    return {
        'nominal_optimizer_steps': sum(steps.values()),
        'model_train_flop_proxy': sum(
            steps[task] * by_task[task]['flop_counter_total']
            for task in steps),
        'one_batch_flops_min': min(flops),
        'one_batch_flops_max': max(flops),
        'probe_script_sha256': scripts.pop(),
    }


def summarize(profiles_root, access_csv, output_root):
    profiles_root, output_root = Path(profiles_root), Path(output_root)
    with Path(access_csv).open(newline='', encoding='utf-8') as handle:
        access = {row['job']: row for row in csv.DictReader(handle)}
    rows = []
    for dataset, method, job_name, commit, relative, tasks in SELECTED:
        job = RESULTS / relative
        config = json.loads((job / 'run/config.json').read_text(encoding='utf-8'))
        steps = nominal_steps_per_task(
            job / 'run/data_flow_audit.jsonl', config)
        profiles = [json.loads((profiles_root / profile_name(dataset, method, task))
                               .read_text(encoding='utf-8'))
                    for task in range(tasks)]
        if any(row.get('job') != job_name
               or row.get('producer_commit') != commit
               or row.get('input_origin')
               != 'synthetic_current_task_shape; no test data'
               or row.get('coverage')
               != 'torch FlopCounterMode supported ops only; no final-head fit'
               for row in profiles):
            raise ValueError(f'profile authority differs: {job_name}')
        counted = weighted_profile(profiles, steps)
        access_row = access[job_name]
        if counted['nominal_optimizer_steps'] != int(
                access_row['nominal_optimizer_steps']):
            raise ValueError(f'profile optimizer steps differ: {job_name}')
        rows.append({
            'job': job_name, 'producer_commit': commit,
            **counted,
            'head_candidate_steps_unprofiled': int(
                access_row['head_candidate_steps']),
            'validation_examples': int(access_row['validation_examples']),
            'calibration_examples': int(access_row['calibration_examples']),
            'flops_measured_complete': False,
        })
    output_root.mkdir(parents=True, exist_ok=False)
    json_path = output_root / 'selected_model_flop_proxy.json'
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
    csv_path = output_root / 'selected_model_flop_proxy.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {'rows': len(rows), 'json_sha256': file_sha256(json_path),
            'csv_sha256': file_sha256(csv_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles-root', type=Path, required=True)
    parser.add_argument('--access-csv', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.profiles_root, args.access_csv,
                               args.output_root), sort_keys=True))
