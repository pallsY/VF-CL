"""Content-bound, independent numeric replay for Replay-11 seed-42 jobs."""

import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch


def unique_correct(accuracy, total):
    """Invert a producer's four-decimal accuracy only when the integer is unique."""
    if (type(total) is not int or total < 1 or type(accuracy) not in (int, float)
            or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
        raise ValueError('invalid rounded accuracy or total')
    candidates = [correct for correct in range(total + 1)
                  if round(correct / total, 4) == accuracy]
    if len(candidates) != 1:
        raise ValueError('rounded accuracy has no unique integer (ambiguous or invalid)')
    return candidates[0]


def _record_path(root, record):
    if type(record) is not dict or type(record.get('path')) is not str:
        raise ValueError('file record is invalid')
    logical = PurePosixPath(record['path'])
    if (logical.is_absolute() or not logical.parts
            or any(part in ('', '.', '..') for part in logical.parts)
            or logical.as_posix() != record['path']):
        raise ValueError('file path escapes run')
    path = Path(root).joinpath(*logical.parts)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('symlink in evidence path')
    return path


def _verify_handle(handle, record):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
        size += len(chunk)
    if digest.hexdigest() != record.get('sha256'):
        raise ValueError('file sha256 differs')
    if 'size' in record and size != record['size']:
        raise ValueError('file size differs')
    handle.seek(0)


def verified_path(root, record):
    """Stream a relative file's hash; ignore historical inode and mtime."""
    path = _record_path(root, record)
    with path.open('rb') as handle:
        _verify_handle(handle, record)
    return path


def verified_bytes(root, record):
    with _record_path(root, record).open('rb') as handle:
        _verify_handle(handle, record)
        content = handle.read()
    return content


def load_checkpoint(root, record):
    """Hash and restricted-load one checkpoint through the same open handle."""
    with _record_path(root, record).open('rb') as handle:
        _verify_handle(handle, record)
        return torch.load(handle, map_location='cpu', weights_only=True)


def read_job_evidence(job_dir):
    """Read frozen bytes and result expectations without producer inode checks."""
    run = Path(job_dir) / 'run'
    def read_json(name):
        return json.loads((run / name).read_text(encoding='utf-8'))
    frozen = read_json('FORMAL_STATE_FROZEN.json')
    complete = read_json('FORMAL_EVALUATION_COMPLETE.json')
    published = read_json('FORMAL_EVALUATION_PUBLISHED.json')
    identity = frozen['identity']
    snapshots = []
    for stage in identity['snapshots']:
        verified_path(run, stage['file'])
        verified_path(run, stage['source_file'])
        snapshots.append(stage)
    final = published['checkpoint']
    if final['path'] != identity['final_checkpoint']['file']['path']:
        raise ValueError('published final checkpoint path differs from freeze')
    verified_path(run, final)
    return {
        'config': read_json('config.json'),
        'results': read_json('results.json'),
        'identity': identity,
        'cache_identity': complete['cache_identity'],
        'evaluation': complete.get('evaluation'),
        'snapshots': snapshots,
        'final': final,
    }


def _tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(repr(tuple(value.shape)).encode('ascii'))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_identity(batches):
    records = []
    count = 0
    for x, y in batches:
        if (x.device.type != 'cpu' or y.device.type != 'cpu' or y.ndim != 1
                or x.ndim < 1 or x.shape[0] != y.shape[0] or not y.numel()):
            raise ValueError('invalid CPU cache batch')
        count += y.numel()
        records.append({
            'input_sha256': _tensor_sha256(x),
            'label_sha256': _tensor_sha256(y),
            'input_shape': list(x.shape), 'label_shape': list(y.shape),
            'input_dtype': str(x.dtype), 'label_dtype': str(y.dtype),
        })
    if not records:
        raise ValueError('empty test cache')
    return {'batch_count': len(records), 'sample_count': count,
            'batches': records}


def verify_cache(batches, expected):
    if cache_identity(batches) != expected:
        raise ValueError('test cache identity differs')
    labels = Counter()
    for _, y in batches:
        labels.update(int(value) for value in y.tolist())
    return dict(labels)


def validate_task_classes(recorded, planned, num_classes):
    """Bind frozen test partition to the complete configured class schedule."""
    pairs = planned.items() if type(planned) is dict else enumerate(planned)
    expected = {task: list(classes) for task, classes in pairs}
    flattened = [label for classes in expected.values() for label in classes]
    if (recorded != expected or type(num_classes) is not int
            or sorted(flattened) != list(range(num_classes))):
        raise ValueError('frozen task classes differ from full planned partition')
    return expected


@torch.no_grad()
def _direct_method_task_il(trainer, method, batches, task_classes):
    """Count oracle-task predictions without inverting a rounded readout."""
    answer = {}
    icarl = getattr(method, 'name', None) == 'iCaRL_VFL'
    if icarl:
        modules = [*trainer.bottoms, trainer.top_model]
        modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            class_ids, means = method._class_means()
            for task, classes in task_classes.items():
                allowed = torch.tensor(
                    [int(label) in classes for label in class_ids.tolist()],
                    dtype=torch.bool, device=class_ids.device)
                selected = torch.where(allowed)[0]
                if not len(selected):
                    raise ValueError('iCaRL oracle task has no exemplar means')
                correct = 0
                for x, y in batches:
                    mask = torch.zeros_like(y, dtype=torch.bool)
                    for label in classes:
                        mask |= y == int(label)
                    if bool(mask.any()):
                        features = method._features(x[mask], record_comm=False)
                        distances = torch.cdist(features, means)
                        predictions = class_ids[selected[distances[:, allowed].argmin(1)]]
                        correct += int(predictions.eq(y[mask].to(predictions.device)).sum())
                answer[f'task_{task}'] = correct
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)
        return answer
    for task, classes in task_classes.items():
        selected = []
        for x, y in batches:
            mask = torch.zeros_like(y, dtype=torch.bool)
            for label in classes:
                mask |= y == int(label)
            if bool(mask.any()):
                selected.append((x[mask], y[mask]))
        if not selected:
            raise ValueError('oracle task has no test examples')
        _, probabilities, labels = trainer.evaluate(tuple(selected))
        labels = labels.cpu().long()
        probabilities = probabilities.cpu()
        allowed = torch.tensor(classes, dtype=torch.long)
        predictions = allowed[probabilities.index_select(1, allowed).argmax(1)]
        answer[f'task_{task}'] = int(predictions.eq(labels).sum())
    return answer


def score_state(trainer, method, batches, task_classes):
    """Count fresh Class-IL and Task-IL predictions on selected CPU batches."""
    if hasattr(method, 'evaluate_class_il_readouts_cached'):
        readouts = method.evaluate_class_il_readouts_cached(batches, task_classes)
        primary = readouts['class_il_pred_task_counts']
        direct_task_il = _direct_method_task_il(
            trainer, method, batches, task_classes)
        answer = {'primary': {}, 'task_il': {}}
        for task, classes in task_classes.items():
            key = f'task_{task}'
            total = sum(int(y.eq(int(label)).sum()) for _, y in batches
                        for label in classes)
            counts = primary[key]
            if counts['total'] != total or not 0 <= counts['correct'] <= total:
                raise ValueError('method exact readout total differs from cache')
            if round(counts['correct'] / total, 4) != readouts['class_il_pred_task'][key]:
                raise ValueError('method exact readout ratio differs')
            answer['primary'][key] = counts['correct']
            answer['task_il'][key] = direct_task_il[key]
            if (round(direct_task_il[key] / total, 4)
                    != readouts['task_il_oracle'][key]):
                raise ValueError('Task-IL direct count differs from method readout')
        return answer
    answer = {'primary': {}, 'task_il': {}}
    for task, classes in task_classes.items():
        selected = []
        for x, y in batches:
            mask = torch.zeros_like(y, dtype=torch.bool)
            for label in classes:
                mask |= y == int(label)
            if bool(mask.any()):
                selected.append((x[mask], y[mask]))
        if not selected:
            raise ValueError('task has no test examples')
        _, probabilities, labels = trainer.evaluate(tuple(selected))
        labels = labels.cpu().long()
        probabilities = probabilities.cpu()
        if probabilities.ndim != 2 or probabilities.shape[0] != labels.numel():
            raise ValueError('trainer readout shape differs')
        allowed = torch.tensor(classes, dtype=torch.long)
        key = f'task_{task}'
        answer['primary'][key] = int(probabilities.argmax(1).eq(labels).sum())
        predicted_in_task = allowed[probabilities.index_select(1, allowed).argmax(1)]
        answer['task_il'][key] = int(predicted_in_task.eq(labels).sum())
    return answer


def compare_trajectory(history, totals, observed):
    """Compare independent integers to every saved diagonal and final readout."""
    count = len(history)
    if count < 1 or set(totals) != {f'task_{i}' for i in range(count)}:
        raise ValueError('task totals differ')
    expected_keys = {'diagonal', 'final', 'task_il'}
    if set(observed) != expected_keys or any(
            set(observed[name]) != set(totals) for name in expected_keys):
        raise ValueError('observed task keys differ')
    for i, row in enumerate(history):
        key = f'task_{i}'
        if row.get('step') != f'event_{i}_CIL':
            raise ValueError('trajectory step differs')
        accuracy = row['per_task_accs'][key]
        if i == count - 1:
            if set(row['per_task_accs']) != set(totals):
                raise ValueError('final task keys differ')
        elif set(row['per_task_accs']) != {key}:
            raise ValueError('diagonal task keys differ')
        if unique_correct(accuracy, totals[key]) != observed['diagonal'][key]:
            raise ValueError('diagonal integer differs')
    final = history[-1]
    if set(final['per_task_accs_taskil']) != set(totals):
        raise ValueError('Task-IL task keys differ')
    for key, total in totals.items():
        if unique_correct(final['per_task_accs'][key], total) != observed['final'][key]:
            raise ValueError('final Class-IL integer differs')
        if unique_correct(final['per_task_accs_taskil'][key], total) != observed['task_il'][key]:
            raise ValueError('final Task-IL integer differs')
    diagonal = [round(observed['diagonal'][f'task_{i}'] / totals[f'task_{i}'], 4)
                for i in range(count)]
    final_scores = [round(observed['final'][f'task_{i}'] / totals[f'task_{i}'], 4)
                    for i in range(count)]
    task_il = [round(observed['task_il'][f'task_{i}'] / totals[f'task_{i}'], 4)
               for i in range(count)]
    return {
        'aa_final_cil': statistics.fmean(final_scores),
        'bwt_cil': (statistics.fmean(final_scores[i] - diagonal[i]
                    for i in range(count - 1)) if count > 1 else 0.0),
        'task_il_final': statistics.fmean(task_il),
    }


def _producer_sources(root, commit, provenance):
    """Refuse to execute source bytes different from the recorded Git blobs."""
    root = Path(root)
    current = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if current != commit or provenance.get('source_commit') != commit:
        raise ValueError('producer commit differs')
    if subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('producer checkout is dirty')
    hashes = provenance.get('source_sha256')
    if type(hashes) is not dict or not hashes:
        raise ValueError('producer source inventory is missing')
    for logical, declared in hashes.items():
        path = PurePosixPath(logical)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != logical:
            raise ValueError('producer source path differs')
        committed = subprocess.check_output(
            ['git', '-C', str(root), 'show', f'{commit}:{logical}'])
        if (hashlib.sha256(committed).hexdigest() != declared
                or hashlib.sha256(root.joinpath(*path.parts).read_bytes()).hexdigest()
                != declared):
            raise ValueError('producer source blob differs')


def _numeric_worker(job_dir, producer_commit, producer_root, data_root):
    started = time.perf_counter()
    torch.set_num_threads(1)
    job_dir = Path(job_dir)
    evidence = read_job_evidence(job_dir)
    identity, config = evidence['identity'], evidence['config']
    if (identity.get('source_commit') != producer_commit
            or config.get('seed') != 42):
        raise ValueError('numeric replay source commit or seed differs')
    _producer_sources(producer_root, producer_commit,
                      identity['source_provenance'])
    sys.path.insert(0, str(Path(producer_root).resolve()))
    from adaptive_consolidation_audit import _formal_state_hash, _fresh_formal_state
    from data_utils import TaskManager, VFLDataset
    from metrics import cache_formal_batches

    task_classes = {
        int(task): list(classes)
        for task, classes in identity['task_classes'].items()
    }
    if sorted(task_classes) != list(range(len(task_classes))):
        raise ValueError('frozen task sequence differs')
    classes = [label for task in sorted(task_classes)
               for label in task_classes[task]]
    expected_cache = evidence['cache_identity']
    expected_test = expected_cache.get('test', expected_cache)
    run = job_dir / 'run'
    for marker in ('FORMAL_EVALUATION_CONSUMING.json',
                   'FORMAL_EVALUATION_SEALED.json'):
        recorded = json.loads((run / marker).read_text(encoding='utf-8'))
        if recorded.get('cache_identity') != expected_cache:
            raise ValueError('formal cache transaction identity differs')
    with tempfile.TemporaryDirectory(prefix='replay11_numeric_') as scratch:
        args = SimpleNamespace(**config)
        args.device = 'cpu'
        args.num_workers = 0
        args.output_dir = scratch
        args.results_dir = scratch
        args.data_path = str(Path(data_root))
        vector = config.get('vector_npz')
        if vector:
            try:
                relative = PurePosixPath(vector.replace('\\', '/')).relative_to(
                    PurePosixPath(config['data_path'].replace('\\', '/')))
            except ValueError as error:
                raise ValueError('vector data path differs from data root') from error
            if '..' in relative.parts:
                raise ValueError('vector data path escapes data root')
            args.vector_npz = str(Path(data_root).joinpath(*relative.parts))
        validate_task_classes(
            task_classes, TaskManager(args).task_classes, args.num_classes)
        dataset = VFLDataset(args)
        indices = dataset.get_class_indices(dataset.testset, classes)
        loader = dataset._loader(
            dataset.testset, indices, ('test', tuple(classes)), False)
        batches = cache_formal_batches(loader)
        label_counts = verify_cache(batches, expected_test)
        totals = {
            f'task_{task}': sum(label_counts.get(int(label), 0)
                                for label in labels)
            for task, labels in task_classes.items()
        }
        if any(total < 1 for total in totals.values()):
            raise ValueError('authoritative test task is empty')
        observed = {'diagonal': {}, 'final': {}, 'task_il': {}}
        for task, stage in enumerate(evidence['snapshots']):
            payload = load_checkpoint(run, stage['file'])
            if (payload['task_id'] != task
                    or payload['protocol'] != identity['protocol']):
                raise ValueError('frozen snapshot protocol differs')
            trainer, method, _ = _fresh_formal_state(payload, args)
            readout = score_state(
                trainer, method, batches,
                {key: task_classes[key] for key in range(task + 1)})
            observed['diagonal'][f'task_{task}'] = readout['primary'][f'task_{task}']
            del trainer, method, payload, readout
            gc.collect()
        final_payload = load_checkpoint(run, evidence['final'])
        if final_payload['protocol'] != identity['protocol']:
            raise ValueError('final checkpoint protocol differs')
        frozen_final = identity['final_checkpoint']
        for key, recorded_key in (
                ('trainer_state', 'trainer_state_sha256'),
                ('cl_state', 'method_state_sha256'),
                ('rng_state', 'rng_state_sha256')):
            if _formal_state_hash(final_payload[key]) != frozen_final[recorded_key]:
                raise ValueError('published final state differs from freeze')
        if (evidence['evaluation'] is not None
                and _formal_state_hash(final_payload['tracker_state'])
                != _formal_state_hash(evidence['evaluation'])):
            raise ValueError('published tracker differs from sealed evaluation')
        trainer, method, _ = _fresh_formal_state(final_payload, args)
        final_readout = score_state(trainer, method, batches, task_classes)
        observed['final'] = final_readout['primary']
        observed['task_il'] = final_readout['task_il']
    metrics = compare_trajectory(
        evidence['results']['task_acc_history'], totals, observed)
    success = json.loads((job_dir / 'JOB_SUCCESS.json').read_text(encoding='utf-8'))
    if metrics != success.get('metrics'):
        raise ValueError('AA/BWT/Task-IL formula differs')
    peak_rss_bytes = None
    if sys.platform.startswith('linux'):
        import resource
        peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        'job': '/'.join(job_dir.parts[-3:]),
        'producer_commit': producer_commit,
        'task_totals': totals,
        'correct': observed,
        'metrics': metrics,
        'seconds': round(time.perf_counter() - started, 3),
        'peak_rss_bytes': peak_rss_bytes,
        'environment': {'python': platform.python_version(),
                        'torch': torch.__version__, 'device': 'cpu',
                        'host': platform.node(), 'peak_cuda_bytes': 0},
    }


def audit_job(job_dir, producer_commit, producer_root, data_root):
    """Launch one read-only numeric replay in a fresh producer-code process."""
    job_dir = Path(job_dir)
    if job_dir.name != '42':
        raise ValueError('numeric replay is preselected for seed 42 only')
    if (type(producer_commit) is not str or len(producer_commit) != 40
            or any(letter not in '0123456789abcdef' for letter in producer_commit)):
        raise ValueError('invalid producer commit')
    import replay11_evidence as basic_audit
    if (producer_commit != basic_audit.CORRECTED_COMMIT
            or not basic_audit._same_path(data_root,
                                          basic_audit.FROZEN_DATA_ROOT)):
        raise ValueError('numeric replay source or data root differs from fixed protocol')
    basic = basic_audit.verify_v3_formal_job(
        job_dir, producer_root=producer_root)
    if basic.get('admission') != 'formal':
        raise ValueError('numeric replay lacks formal basic admission')
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--worker',
         str(job_dir.resolve()), producer_commit, str(Path(producer_root).resolve()),
         str(Path(data_root).resolve())],
        capture_output=True, text=True, check=False)
    if completed.returncode:
        raise ValueError('numeric worker failed: ' + completed.stderr[-4000:])
    marker = 'REPLAY11_NUMERIC_JSON='
    lines = [line[len(marker):] for line in completed.stdout.splitlines()
             if line.startswith(marker)]
    if len(lines) != 1:
        raise ValueError('numeric worker produced no single result')
    result = json.loads(lines[0])
    result['basic_admission'] = 'formal'
    return result


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--worker':
        result = _numeric_worker(*sys.argv[2:])
        print('REPLAY11_NUMERIC_JSON=' + json.dumps(result, sort_keys=True))
    elif len(sys.argv) == 5:
        print(json.dumps(audit_job(*sys.argv[1:]), indent=2, sort_keys=True))
    else:
        raise SystemExit(
            'usage: replay11_numeric.py JOB_DIR PRODUCER_COMMIT PRODUCER_ROOT DATA_ROOT')
