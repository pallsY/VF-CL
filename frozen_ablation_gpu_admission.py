"""Isolated CIFAR GPU admission; never launches a formal worker or test loader."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import frozen_ablation_smoke as smoke
import frozen_ablation_manifest as manifest

PROFILES = ('base', 'default_on', 'off', 'no_partykd')
THRESHOLDS = {'reference': 3584, 'candidate': 8192}
GPUS = {'reference': ('GPU-6569e526-43ea-74e1-264e-4b66417181e1', 'NVIDIA GeForce RTX 3080'),
        'candidate': ('GPU-950d97cf-5ce8-44a6-d79a-12d7d6c0bd22', 'NVIDIA GeForce RTX 3090')}
BUDGET_MIB = 2048
NUMERIC_FIELDS = ('states', 'losses', 'gradient_steps', 'fim_scores',
                  'fim_tensor_values', 'sample_orders', 'applied_freeze')
ORIGINAL_CONFIG = smoke.smoke_config
ALLOWED_ROOTS = tuple(map(Path, ('/home/nlp/YangXX/vfcl-env-setup/queue-admission',
    '/tmp/vfcl-3090-admission', '/brf/users/yangxx/VFCL-admission/3090-20261101')))


def independent_path(path):
    path = Path(path).absolute()
    resolved = path.resolve()
    forbidden = (Path('/brf/users/yangxx/VFCL-results').resolve(),
                 smoke.SOURCE_ROOT.resolve(), Path(__file__).resolve().parent)
    if (any(candidate == root or candidate.is_relative_to(root)
            for candidate in (path, resolved) for root in forbidden) or
            not any(path.is_relative_to(root.absolute()) and resolved.is_relative_to(root.resolve())
                    for root in ALLOWED_ROOTS)):
        raise ValueError('admission path is outside independent allowlist: ' + str(path))
    return resolved


def current_commit():
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent),
                                    'rev-parse', 'HEAD'], text=True).strip()


def check_no_test(output):
    run_dir = independent_path(Path(output) / 'run')
    access_dir = run_dir / 'formal_access'
    if access_dir.is_symlink():
        raise ValueError('no-test evidence violated: symlink access directory')
    independent_path(access_dir)
    required_audit = independent_path(run_dir / 'data_flow_audit.jsonl')
    if not required_audit.is_file():
        raise ValueError('no-test evidence violated: data-flow audit is absent')
    if list((run_dir / 'formal_access').glob('test.*.json')) or (run_dir / 'results.json').exists():
        raise ValueError('no-test evidence violated: test authorization/result marker')
    def has_test(value):
        if isinstance(value, dict):
            return value.get('split') == 'test' or any(has_test(item) for item in value.values())
        return isinstance(value, list) and any(has_test(item) for item in value)
    for audit in run_dir.glob('data_flow_audit*'):
        independent_path(audit)
        for line in audit.read_text(encoding='utf-8').splitlines():
            if line.strip() and has_test(json.loads(line)):
                raise ValueError('no-test evidence violated: test data-flow audit')
    access_files = []
    for path in sorted(access_dir.rglob('*')):
        if path.is_symlink() or path.name.startswith('test.'):
            raise ValueError('no-test evidence violated: test marker or symlink')
        independent_path(path)
        if path.is_file():
            access_files.append(path.relative_to(run_dir / 'formal_access').as_posix())
    return dict(data_flow_audit_sha256=sha(required_audit), formal_access_files=access_files)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with independent_path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def anchored_source(root):
    """Authenticate current CIFAR source against the independent frozen publication."""
    root = Path(root).resolve()
    anchor_path = root / 'audit/ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json'
    if sha(anchor_path) != manifest.SOURCE_ANCHOR_SHA:
        raise ValueError('anchored source: original formal anchor changed')
    anchor = json.loads(anchor_path.read_text(encoding='utf-8'))
    if (anchor.get('producer_commit') != smoke.BASE_COMMIT or
            anchor.get('basic_passed') != 9 or anchor.get('verified_original_trees') != 9):
        raise ValueError('anchored source: original nine jobs are not anchored')
    gate_path = (root / anchor['final_gate']['relative_path']).resolve()
    if not gate_path.is_relative_to(root) or sha(gate_path) != anchor['final_gate']['sha256']:
        raise ValueError('anchored source: final gate changed')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    source_job = root / 'full/cifar100/adaptive/42'
    tree_sha = manifest.original_tree_sha256(source_job)
    if gate['original_tree_sha256'].get('cifar100/adaptive/42') != tree_sha:
        raise ValueError('anchored source: original CIFAR tree changed')
    return dict(anchor_sha256=manifest.SOURCE_ANCHOR_SHA,
                final_gate_sha256=sha(gate_path), original_tree_sha256=tree_sha,
                source_config_sha256=sha(source_job / 'run/config.json'))


def gpu_config(source, profile, output):
    if profile not in PROFILES:
        raise ValueError('unknown GPU profile')
    config = ORIGINAL_CONFIG(source, 'base' if profile == 'no_partykd' else profile, output)
    config['device'] = 'cuda:0'
    if profile == 'no_partykd':
        config.update(party_kd_enabled=0, dep_tracking_enabled=0,
                      expected_party_kd_variant='base')
    return config


def query_gpu(uuid):
    raw = subprocess.check_output([
        'nvidia-smi', '--id=' + uuid,
        '--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu,driver_version',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(raw) != 1:
        raise ValueError('GPU UUID did not resolve uniquely')
    actual, name, free, total, utilization, driver = [v.strip() for v in raw[0].split(',')]
    if actual != uuid:
        raise ValueError('GPU UUID differs')
    return dict(uuid=actual, name=name, free_mib=int(free), total_mib=int(total),
                utilization_percent=int(utilization), driver_version=driver)


def validate_gate(rows, role, uuid):
    if role not in THRESHOLDS or len(rows) < 2:
        raise ValueError('resource gate missing')
    times = [row['elapsed_seconds'] for row in rows]
    if (times[0] != 0 or times[-1] < 120 or
            any(not 0 < b - a <= 1.5 for a, b in zip(times, times[1:])) or
            any(row['uuid'] != uuid or row.get('name') != GPUS[role][1] or
                row['free_mib'] < THRESHOLDS[role]
                for row in rows)):
        raise ValueError('resource gate requires continuous 120 seconds')


def wait_gate(role, uuid, journal):
    rows, start = [], time.monotonic()
    while True:
        row = query_gpu(uuid)
        if not rows:
            start = time.monotonic()
        elapsed = time.monotonic() - start
        row['elapsed_seconds'] = 0 if not rows else elapsed
        journal.append(dict(phase='gate', **row))
        if row['free_mib'] < THRESHOLDS[role]:
            rows, start = [], time.monotonic()
        else:
            rows.append(row)
            if rows[-1]['elapsed_seconds'] >= 120:
                validate_gate(rows, role, uuid)
                return rows
        time.sleep(1)


def compare_numeric(left, right):
    for field in NUMERIC_FIELDS:
        if (not left.get(field) or not right.get(field) or
                json.dumps(left[field], sort_keys=True, allow_nan=False) !=
                json.dumps(right[field], sort_keys=True, allow_nan=False)):
            raise ValueError('bitwise evidence differs or absent: ' + field)


def run(profile, role, uuid, code_root, output):
    # One fresh process per profile: run_smoke intentionally instruments imported training classes.
    if profile not in PROFILES or role not in THRESHOLDS or uuid != GPUS[role][0]:
        raise ValueError('invalid admission identity')
    output = independent_path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal, stop = [], threading.Event()
    errors = []
    def monitor():
        while not stop.wait(1):
            try:
                journal.append(dict(phase='training', timestamp=time.time(), **query_gpu(uuid)))
            except Exception as exc:
                errors.append(str(exc))
                return
    try:
        source_provenance = anchored_source(smoke.SOURCE_ROOT)
        source_path = smoke.SOURCE_ROOT / 'full/cifar100/adaptive/42/run/config.json'
        with independent_path(output / 'SOURCE_CONFIG.json').open('xb') as stream:
            stream.write(source_path.read_bytes())
        if sha(output / 'SOURCE_CONFIG.json') != source_provenance['source_config_sha256']:
            raise ValueError('anchored source config changed before GPU precheck')
        producer_commit = current_commit()
        if query_gpu(uuid)['name'] != GPUS[role][1]:
            raise ValueError('physical GPU model differs from pinned role')
        gate = wait_gate(role, uuid, journal)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('precheck must see exactly one GPU')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(BUDGET_MIB * 1024 ** 2 / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        launch = query_gpu(uuid)
        journal.append(dict(phase='launch_recheck', timestamp=time.time(), **launch))
        if launch['free_mib'] < THRESHOLDS[role]:
            raise ValueError('launch resource recheck failed')
        if sha(source_path) != source_provenance['source_config_sha256']:
            raise ValueError('anchored source config changed before training launch')
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        mapped = 'base' if profile == 'no_partykd' else profile
        with patch.object(smoke, 'smoke_config', lambda source, mode, target:
                          gpu_config(source, profile, target)):
            record = smoke.run_smoke('cifar100', mapped, code_root.resolve(), output / 'smoke')
        no_test_inventory = check_no_test(output / 'smoke')
        if record.get('orchestration_commit') != producer_commit:
            raise ValueError('GPU smoke orchestration commit differs from current harness')
        torch.cuda.synchronize()
        stop.set()
        worker.join()
        if errors:
            raise ValueError('GPU telemetry failed: ' + repr(errors))
        peak = torch.cuda.max_memory_reserved(0)
        if peak > BUDGET_MIB * 1024 ** 2:
            raise ValueError('allocator peak exceeds admission budget')
        config = json.loads((output / 'smoke/SMOKE_CONFIG.json').read_text())
        if (sha(source_path) != source_provenance['source_config_sha256'] or
                sha(output / 'SOURCE_CONFIG.json') != source_provenance['source_config_sha256'] or
                record['source_config_sha256'] != source_provenance['source_config_sha256']):
            raise ValueError('anchored source config changed during GPU precheck')
        result = dict(schema_version=1, profile=profile, role=role, hostname=socket.gethostname(),
                      gpu_uuid=uuid, allocator_budget_mib=BUDGET_MIB,
                      allocator_peak_reserved_bytes=peak,
                      allocator_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      harness_sha256=sha(__file__), gpu_config_override={'device': 'cuda:0'},
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      cudnn_version=torch.backends.cudnn.version(),
                      gpu_name=torch.cuda.get_device_name(0),
                      gate=gate, launch_recheck=launch,
                      source_config_file_sha256=sha(output / 'SOURCE_CONFIG.json'),
                      source_provenance=source_provenance,
                      no_test_inventory=no_test_inventory,
                      config=config, smoke_result_sha256=sha(output / 'smoke/SMOKE_RESULT.json'))
        write(output / 'GPU_ADMISSION.json', result)
        return result
    except Exception as exc:
        write(output / 'GPU_ADMISSION_FAILURE.json', {'error': str(exc), 'profile': profile})
        raise
    finally:
        stop.set()
        write(output / 'GPU_RESOURCE_SAMPLES.json', journal)


def _hash_metadata(value):
    """Match smoke.value_hash for the JSON-only complete FIM inventory."""
    digest = hashlib.sha256()
    def feed(item):
        if isinstance(item, dict):
            digest.update(b'D')
            for key in sorted(item, key=repr):
                feed(key)
                feed(item[key])
        elif isinstance(item, list):
            digest.update(b'L' + str(len(item)).encode())
            for part in item:
                feed(part)
        else:
            digest.update(b'S' + repr(item).encode())
    feed(value)
    return digest.hexdigest()


def load_profile(root, profile, role, source_provenance=None, producer_commit=None):
    directory = independent_path(Path(root) / profile)
    # Validate every existing descendant before reading evidence through it.
    for path in directory.rglob('*'):
        independent_path(path)
    meta = json.loads((directory / 'GPU_ADMISSION.json').read_text())
    output = directory / 'smoke'
    no_test_inventory = check_no_test(output)
    record = json.loads((output / 'SMOKE_RESULT.json').read_text())
    config = json.loads((output / 'SMOKE_CONFIG.json').read_text())
    source = json.loads((directory / 'SOURCE_CONFIG.json').read_text())
    if source_provenance is None:
        source_provenance = anchored_source(smoke.SOURCE_ROOT)
    if (sha(directory / 'SOURCE_CONFIG.json') != source_provenance['source_config_sha256'] or
            meta.get('source_provenance') != source_provenance):
        raise ValueError('anchored source config/provenance differs: ' + profile)
    no_test = json.loads((output / 'NO_TEST_ACCESS.json').read_text())
    mapped = 'base' if profile == 'no_partykd' else profile
    producer_output = Path(record.get('smoke_output_dir', ''))
    expected = smoke.BASE_COMMIT if mapped == 'base' else smoke.PATCH_COMMIT
    runtime = record.get('runtime', {})
    if producer_commit is None:
        producer_commit = current_commit()
    if (meta.get('schema_version') != 1 or meta.get('profile') != profile or
            meta.get('role') != role or meta.get('harness_sha256') != sha(__file__) or
            meta.get('no_test_inventory') != no_test_inventory or
            meta.get('gpu_uuid') != GPUS[role][0] or meta.get('gpu_name') != GPUS[role][1] or
            meta.get('compute_capability') != [8, 6] or not meta.get('cudnn_version') or
            meta.get('allocator_budget_mib') != BUDGET_MIB or
            not 0 < meta.get('allocator_peak_allocated_bytes', 0) <=
            meta.get('allocator_peak_reserved_bytes', 0) <= BUDGET_MIB * 1024 ** 2 or
            meta.get('gpu_config_override') != {'device': 'cuda:0'} or
            meta.get('smoke_result_sha256') != sha(output / 'SMOKE_RESULT.json') or
            meta.get('source_config_file_sha256') != sha(directory / 'SOURCE_CONFIG.json') or
            record.get('source_config_sha256') != sha(directory / 'SOURCE_CONFIG.json') or
            meta.get('config') != config or config != gpu_config(source, profile, producer_output) or
            source.get('seed') != 42 or config.get('epochs_per_task') != 1 or
            record.get('dataset') != 'cifar100' or record.get('mode') != mapped or
            record.get('code_commit') != expected or
            not all(runtime.get(key) for key in ('cuda_runtime', 'host', 'numpy', 'pillow',
                'python', 'scikit_learn', 'torch', 'torchvision')) or
            runtime.get('host') != meta.get('hostname') or
            runtime.get('python') != '3.11.15' or runtime.get('torch') != '2.5.1+cu124' or
            runtime.get('cuda_runtime') != '12.4' or
            record.get('orchestration_commit') != producer_commit or
            record.get('smoke_script_sha256') != sha(smoke.__file__) or
            record.get('smoke_config_sha256') != sha(output / 'SMOKE_CONFIG.json') or
            record.get('no_test_access_sha256') != sha(output / 'NO_TEST_ACCESS.json') or
            record.get('test_accessed') is not False or
            record.get('stopped_after_two_tasks') is not True or
            record.get('train_per_class') != 64 or
            no_test != dict(schema_version=1, status='no_test_access',
                smoke_output_dir=str(producer_output), test_accessed=False,
                stopped_after_two_tasks=True, test_marker_present=False,
                formal_results_present=False, smoke_config_sha256=sha(output / 'SMOKE_CONFIG.json')) or
            (output / 'run/formal_access/test.consumed.json').exists() or
            (output / 'run/results.json').exists()):
        raise ValueError('profile/config/provenance/no-test evidence invalid: ' + profile)
    validate_gate(meta.get('gate', []), role, meta['gpu_uuid'])
    launch = meta.get('launch_recheck', {})
    if launch.get('uuid') != meta['gpu_uuid'] or launch.get('free_mib', 0) < THRESHOLDS[role]:
        raise ValueError('launch resource recheck invalid')
    samples = json.loads((directory / 'GPU_RESOURCE_SAMPLES.json').read_text())
    if (not samples or any(row.get('uuid') != meta['gpu_uuid'] for row in samples) or
            not any(row.get('phase') == 'launch_recheck' for row in samples) or
            not any(row.get('phase') == 'training' for row in samples)):
        raise ValueError('resource samples absent')
    logged_gate = [{k: v for k, v in row.items() if k != 'phase'}
                   for row in samples if row.get('phase') == 'gate']
    logged_launch = [{k: v for k, v in row.items() if k not in ('phase', 'timestamp')}
                     for row in samples if row.get('phase') == 'launch_recheck']
    if logged_gate[-len(meta['gate']):] != meta['gate'] or logged_launch != [launch]:
        raise ValueError('resource gate/recheck differs from raw samples')
    for field in NUMERIC_FIELDS:
        rows = record.get(field, [])
        tasks = [row.get('task') for row in rows]
        if set(tasks) != {0, 1} or (field in ('states', 'sample_orders', 'fim_tensor_values',
                                           'applied_freeze') and tasks != [0, 1]):
            raise ValueError('task boundary evidence incomplete: ' + field)
    for task, state in enumerate(record['states']):
        for field in ('model_sha256', 'memory_mask_sha256', 'rng_sha256',
                      'fim_masks_sha256', 'fim_tensor_values_sha256', 'sample_order_sha256'):
            if len(state.get(field, '')) != 64:
                raise ValueError('state hash absent: ' + field)
        tensors = record['fim_tensor_values'][task].get('tensors', [])
        names = [row.get('name') for row in tensors]
        if (not names or len(names) != len(set(names)) or
                any(not row.get('dtype') or not isinstance(row.get('shape'), list) or
                    len(row.get('sha256', '')) != 64 for row in tensors) or
                state['fim_tensor_values_sha256'] != _hash_metadata([record['fim_tensor_values'][task]])):
            raise ValueError('complete FIM evidence invalid')
    inventories = [[(row['name'], row['shape'], row['dtype']) for row in item['tensors']]
                   for item in record['fim_tensor_values']]
    if inventories[0] != inventories[1]:
        raise ValueError('complete FIM inventory changed between tasks')
    for row in record['gradient_steps']:
        if len(row.get('sha256', '')) != 64:
            raise ValueError('gradient hash absent')
    for task in record['sample_orders']:
        if not task.get('loaders') or any(not item.get('loader_key') or
                not item.get('epoch_orders') or any(not order for order in item['epoch_orders'])
                for item in task['loaders']):
            raise ValueError('raw sample order absent')
    smoke.validate_sample_orders(record, record, record)
    return meta, record, source


def compare(reference, candidate, report):
    reference, candidate, report = map(independent_path, (reference, candidate, report))
    result = {'status': 'rejected', 'errors': [], 'profiles': {}, 'test_accessed': False}
    try:
        # One independent source-tree verification for this comparison, shared by all eight loads.
        source_provenance = anchored_source(smoke.SOURCE_ROOT)
        producer_commit = current_commit()
        result['source_provenance'] = source_provenance
        hosts = {}
        for role, root in (('reference', reference), ('candidate', candidate)):
            entries = {profile: load_profile(root, profile, role, source_provenance, producer_commit)
                       for profile in PROFILES}
            hosts[role] = entries
            base, default, off = [entries[p][1] for p in PROFILES[:3]]
            compare_numeric(base, default)
            for field in NUMERIC_FIELDS:
                if ([row for row in base[field] if row['task'] == 0] !=
                        [row for row in off[field] if row['task'] == 0]):
                    raise ValueError(role + ': off task0 differs: ' + field)
            smoke.validate_sample_orders(base, default, off)
            if (any(row['frozen_elements'] != 0 for row in off['applied_freeze']) or
                    default['applied_freeze'][1]['frozen_elements'] <= 0 or
                    base['states'][1]['model_sha256'] == off['states'][1]['model_sha256']):
                raise ValueError(role + ': off freeze/divergence check failed')
            identities = {(entry[0]['hostname'], entry[0]['gpu_uuid']) for entry in entries.values()}
            if len(identities) != 1 or len({entry[1]['orchestration_commit'] for entry in entries.values()}) != 1:
                raise ValueError(role + ': profile producer identity differs')
            if any(entry[2] != entries['base'][2] for entry in entries.values()):
                raise ValueError(role + ': source config differs across profiles')
        for profile in PROFILES:
            left, right = hosts['reference'][profile], hosts['candidate'][profile]
            if (left[0]['hostname'] == right[0]['hostname'] or
                    left[0]['gpu_uuid'] == right[0]['gpu_uuid'] or left[2] != right[2] or
                    any(left[1].get(key) != right[1].get(key) for key in
                        ('orchestration_commit', 'source_config_sha256')) or
                    {k: v for k, v in left[1].get('runtime', {}).items() if k != 'host'} !=
                    {k: v for k, v in right[1].get('runtime', {}).items() if k != 'host'} or
                    left[0].get('cudnn_version') != right[0].get('cudnn_version')):
                raise ValueError('cross-host provenance differs: ' + profile)
            compare_numeric(left[1], right[1])
            result['profiles'][profile] = dict(bitwise_identical=True, **{
                role + '_evidence_sha256': {
                    name: sha(Path(root) / profile / name) for name in
                    ('GPU_ADMISSION.json', 'GPU_RESOURCE_SAMPLES.json', 'SOURCE_CONFIG.json',
                     'smoke/SMOKE_RESULT.json', 'smoke/SMOKE_CONFIG.json', 'smoke/NO_TEST_ACCESS.json',
                     'smoke/run/data_flow_audit.jsonl')}
                for role, root in (('reference', reference), ('candidate', candidate))})
        result['status'] = 'admitted'
    except Exception as exc:
        result['errors'].append(str(exc))
    write(report, result)
    if result['status'] != 'admitted':
        raise ValueError('admission rejected: ' + '; '.join(result['errors']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--profile', choices=PROFILES, required=True)
    run_parser.add_argument('--role', choices=THRESHOLDS, required=True)
    run_parser.add_argument('--gpu-uuid', required=True)
    run_parser.add_argument('--code-root', type=Path, required=True)
    run_parser.add_argument('--output-root', type=Path, required=True)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('--reference-root', type=Path, required=True)
    compare_parser.add_argument('--candidate-root', type=Path, required=True)
    compare_parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.profile, args.role, args.gpu_uuid, args.code_root, args.output_root / args.profile)
    else:
        compare(args.reference_root, args.candidate_root, args.report)


if __name__ == '__main__':
    main()
