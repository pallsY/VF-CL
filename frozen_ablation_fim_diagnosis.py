"""Bounded two-task FIM diagnosis. A supported diagnosis never grants GPU admission."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

import frozen_ablation_gpu_admission as admission
import frozen_ablation_smoke as smoke

ATOL, RTOL = 1e-12, 1e-6
HISTORICAL_PRODUCER = '7c2cc3602526e2d1fa3d82512e6290ce6f003889'
SCOPE = ('CIFAR seed42 base, 64 training examples/class, 1 epoch/task, tasks0/1, '
         'BiC retained; no full-task, other-seed, or ablation inference; no GPU admission. '
         'Cross-host reduction reproduction cannot isolate GPU hardware from driver differences')
check_no_test = admission.check_no_test
HASH = smoke.value_hash


def require_environment():
    for name, value in dict(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                            MKL_NUM_THREADS='1', PYTHONHASHSEED='42').items():
        if os.environ.get(name) != value:
            raise ValueError(f'diagnosis requires startup environment {name}={value}')


def fresh_output(output):
    output = admission.independent_path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def inventory(root, excluded=()):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        admission.independent_path(path)
        if path.is_symlink():
            raise ValueError('sealed evidence cannot contain symlinks')
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            result[path.relative_to(root).as_posix()] = admission.sha(path)
    return result


def code_hashes():
    return {Path(path).name: admission.sha(path)
            for path in (__file__, admission.__file__, smoke.__file__, admission.manifest.__file__,
                         Path(__file__).with_name('run_frozen_fim_diagnosis.sh'))}


def require_clean_checkout():
    if subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent),
                                'status', '--porcelain'], text=True).strip():
        raise ValueError('diagnosis requires a clean committed checkout')


def seal(root, metadata):
    admission.write(root / 'SEALED.json', dict(metadata, files=inventory(root),
                    schema_version=1, utc=datetime.now(timezone.utc).isoformat(),
                    code_sha256=code_hashes(), scope=SCOPE, atol=ATOL, rtol=RTOL))
    admission.write(root / 'SEAL_SHA256.json', {'SEALED.json': admission.sha(root / 'SEALED.json')})


def load_seal(root):
    root = admission.independent_path(root)
    expected = json.loads((root / 'SEAL_SHA256.json').read_text())
    record = json.loads((root / 'SEALED.json').read_text())
    if (expected != {'SEALED.json': admission.sha(root / 'SEALED.json')} or
            record.get('files') != inventory(root, ('SEALED.json', 'SEAL_SHA256.json')) or
            record.get('code_sha256') != code_hashes() or
            record.get('schema_version') != 1 or record.get('scope') != SCOPE or
            record.get('atol') != ATOL or record.get('rtol') != RTOL or
            record.get('test_accessed') is not False):
        raise ValueError('diagnostic seal/hash/provenance differs')
    return record


def compare_historical(candidate, reference):
    """Only the pre-existing score-hash difference may differ between old hosts."""
    for field in admission.NUMERIC_FIELDS:
        if field != 'fim_scores' and (not candidate.get(field) or
                                     candidate[field] != reference.get(field)):
            raise ValueError('historical non-score evidence differs: ' + field)
    if (candidate.get('orchestration_commit') != HISTORICAL_PRODUCER or
            reference.get('orchestration_commit') != HISTORICAL_PRODUCER or
            not candidate.get('source_config_sha256') or
            candidate['source_config_sha256'] != reference.get('source_config_sha256') or
            not candidate.get('runtime') or
            not isinstance(reference.get('runtime'), dict) or
            {k: v for k, v in candidate['runtime'].items() if k != 'host'} !=
            {k: v for k, v in reference.get('runtime', {}).items() if k != 'host'}):
        raise ValueError('historical source/runtime/producer provenance differs')


def compare_rows(candidate, reference):
    if not candidate or len(candidate) != len(reference):
        raise ValueError('missing numeric rows')
    rows, passed = [], True
    for left, right in zip(candidate, reference):
        if (any(left.get(k) != right.get(k) for k in ('task', 'name', 'kind', 'shape', 'dtype')) or
                left.get('kind') not in ('mean', 'quantile') or
                (left['kind'] == 'mean' and any(type(row.get(k)) is not bool
                    for row in (left, right) for k in ('selected', 'accumulated'))) or
                not isinstance(left.get('shape'), list) or not left.get('dtype') or
                not left.get('values') or len(left['values']) != len(right.get('values', [])) or
                len(left['values']) != math.prod(left['shape']) or
                any(not math.isfinite(v) for row in (left, right) for v in row['values'])):
            raise ValueError('nonfinite/missing/shape/dtype numeric evidence')
        differences = [abs(a - b) for a, b in zip(left['values'], right['values'])]
        outside = sum(d > ATOL + RTOL * abs(b)
                      for d, b in zip(differences, right['values']))
        relative = [d / abs(b) if b else (0.0 if d == 0 else float('inf'))
                    for d, b in zip(differences, right['values'])]
        exact = all(left.get(k) == right.get(k) for k in ('selected', 'accumulated'))
        passed &= outside == 0 and exact
        rows.append(dict(candidate=left, reference=right,
                         max_absolute_difference=max(differences),
                         max_relative_difference=max(relative) if all(map(math.isfinite, relative))
                         else 'infinity', outside_tolerance=outside,
                         selection_and_accumulated_exact=exact))
    return dict(status='diagnosis_supported' if passed else 'diagnosis_rejected',
                atol=ATOL, rtol=RTOL, rows=rows, scope=SCOPE, test_accessed=False)


def historical(root, role):
    return admission.load_profile(root, 'base', role,
                                  producer_commit=HISTORICAL_PRODUCER)


def require_driver(actual, historical_meta):
    if (not actual.get('driver_version') or actual['driver_version'] !=
            historical_meta['launch_recheck'].get('driver_version')):
        raise ValueError('historical host driver differs')


def numeric_row(task, name, tensor, kind='mean', **extra):
    return dict(task=task, name=name, kind=kind, shape=list(tensor.shape), dtype=str(tensor.dtype),
                values=tensor.reshape(-1).tolist(), **extra)


def validate_payload(payload, record, config):
    import numpy as np
    import torch
    tensors, quantiles = payload['tensors'], payload['quantiles']
    parties = config['num_parties']
    if parties != 4 or len(quantiles) != 8 or len(record['fim_scores']) != 8:
        raise ValueError('diagnosis requires the historical eight quantile records')
    expected = [(row['task'], tensor) for row in record['fim_tensor_values']
                for tensor in row['tensors']]
    if len(tensors) != len(expected):
        raise ValueError('raw FIM inventory is incomplete')
    for item, (task, old) in zip(tensors, expected):
        tensor, mean = item['tensor'], item['native_mean']
        if (item['task'] != task or item['name'] != old['name'] or
                item['shape'] != old['shape'] or list(tensor.shape) != item['shape'] or
                item['dtype'] != old['dtype'] or str(tensor.dtype) != item['dtype'] or
                list(tensor.stride()) != item['stride'] or item['device'] != 'cuda:0' or
                tensor.dtype != torch.float32 or
                tensor.device.type != 'cpu' or mean.device.type != 'cpu' or
                mean.shape != torch.Size([]) or mean.dtype != tensor.dtype or
                not torch.isfinite(tensor).all().item() or not torch.isfinite(mean).item() or
                HASH(tensor, torch, np) != old['sha256']):
            raise ValueError('raw FIM hash/metadata/native mean binding differs')
    rows, masks, position = [], [{} for _ in range(parties)], 0
    for task in (0, 1):
        for party in range(parties):
            group = [item for item in tensors if item['task'] == task and
                     item['name'].startswith(f'party{party}/')]
            quantile, score = quantiles[position], record['fim_scores'][position]
            position += 1
            values, threshold = quantile['values'], quantile['threshold']
            native = torch.tensor([item['native_mean'].item() for item in group], dtype=torch.float32)
            if (not group or quantile['task'] != task or score['task'] != task or
                    quantile['q'] != max(0.0, 1.0 - config.get('fim_freeze_frac', .25)) or
                    values.device.type != 'cpu' or threshold.device.type != 'cpu' or
                    values.dtype != torch.float32 or threshold.dtype != values.dtype or
                    list(values.shape) != [len(group)] or list(threshold.shape) != [] or
                    not torch.isfinite(values).all().item() or not torch.isfinite(threshold).item() or
                    values.sum().item() == 0 or not torch.equal(values, native) or
                    HASH(values, torch, np) != score['values_sha256'] or
                    HASH(threshold, torch, np) != score['threshold_sha256']):
                raise ValueError('historical quantile/native mean mapping/hash differs')
            for item in group:
                name = item['name'].split('/', 1)[1]
                selected = item['native_mean'].item() >= threshold.item()
                masks[party][name] = masks[party].get(name, False) or selected
                rows.append(numeric_row(task, item['name'], item['native_mean'],
                    selected=selected, accumulated=masks[party][name],
                    cpu_float64_mean=item['tensor'].to(torch.float64).mean().item(),
                    raw_shape=item['shape'], raw_stride=item['stride']))
            rows.append(numeric_row(task, f'party{party}/quantile', threshold, kind='quantile'))
        if HASH(masks, torch, np) != record['states'][task]['fim_masks_sha256']:
            raise ValueError('historical accumulated mask hash differs')
    return rows


def reconstruct_reference(payload, reference, config, device='cuda:0', raw_output=None):
    import torch
    torch.set_num_threads(1)
    result = dict(tensors=[], quantiles=[])
    for item in payload['tensors']:
        tensor = torch.empty_strided(item['shape'], item['stride'],
                                     dtype=item['tensor'].dtype, device=device)
        tensor.copy_(item['tensor'])
        if list(tensor.stride()) != item['stride']:
            raise ValueError('replay tensor layout differs')
        result['tensors'].append(dict(item, native_mean=tensor.mean().detach().cpu()))
        del tensor
    q = max(0.0, 1.0 - config.get('fim_freeze_frac', .25))
    for task in (0, 1):
        for party in range(config['num_parties']):
            values = torch.tensor([item['native_mean'].item() for item in result['tensors']
                if item['task'] == task and item['name'].startswith(f'party{party}/')],
                dtype=torch.float32)
            result['quantiles'].append(dict(task=task, q=q, values=values,
                                            threshold=torch.quantile(values, q)))
    if raw_output is not None:
        with admission.independent_path(raw_output).open('xb') as stream:
            torch.save(result, stream)
    return validate_payload(result, reference, config), result


def capture_observer(payload):
    def observed(value, torch, np):
        frame = inspect.currentframe().f_back
        local = frame.f_locals
        try:
            if frame.f_code.co_name == 'tracked_tensor_mean' and value is local.get('self'):
                context, index = local['fim_capture'], local['index']
                tensor = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                             device='cpu')
                tensor.copy_(value.detach())
                payload['tensors'].append(dict(task=context['task'], name=context['names'][index],
                    tensor=tensor, shape=list(value.shape), stride=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device),
                    native_mean=local['result'].detach().cpu().clone()))
            elif frame.f_code.co_name == 'quantile' and value is local.get('values'):
                args, kwargs = local['args'], local['kwargs']
                if len(args) != 1 or kwargs:
                    raise ValueError('unexpected quantile call signature')
                payload['quantiles'].append(dict(task=local['task']['id'], q=float(args[0]),
                    values=value.detach().cpu().clone(),
                    threshold=local['result'].detach().cpu().clone()))
        finally:
            del frame
        return HASH(value, torch, np)
    return observed


def capture(code_root, output, historical_candidate):
    output = fresh_output(output)
    payload = dict(tensors=[], quantiles=[])
    try:
        require_environment()
        require_clean_checkout()
        old_files = inventory(Path(historical_candidate) / 'base')
        old_meta, old, _ = historical(historical_candidate, 'candidate')
        with patch.object(smoke, 'value_hash', capture_observer(payload)):
            admission.run('base', 'candidate', admission.GPUS['candidate'][0],
                          Path(code_root), output / 'base')
        import torch
        with admission.independent_path(output / 'RAW_FIM.pt').open('xb') as stream:
            torch.save(payload, stream)
        meta, fresh, _ = admission.load_profile(output, 'base', 'candidate')
        admission.compare_numeric(fresh, old)
        require_driver(meta['launch_recheck'], old_meta)
        validate_payload(payload, old, meta['config'])
        if old_files != inventory(Path(historical_candidate) / 'base'):
            raise ValueError('historical evidence changed during capture')
        admission.write(output / 'CAPTURE.json', dict(status='capture_bound', test_accessed=False,
            producer_commit=admission.current_commit(), historical_producer=HISTORICAL_PRODUCER,
            historical_candidate_files=old_files,
            historical_no_test_inventory=old_meta['no_test_inventory'],
            historical_driver=old_meta['launch_recheck']['driver_version'],
            capture_driver=meta['launch_recheck']['driver_version'],
            tensor_count=len(payload['tensors']), quantile_count=len(payload['quantiles'])))
        seal(output, dict(status='capture_bound', test_accessed=False))
        return json.loads((output / 'CAPTURE.json').read_text())
    except Exception as exc:
        if payload['tensors'] and not (output / 'RAW_FIM.pt').exists():
            import torch
            with admission.independent_path(output / 'RAW_FIM.pt').open('xb') as stream:
                torch.save(payload, stream)
        admission.write(output / 'DIAGNOSIS_FAILURE.json',
                        dict(status='diagnosis_rejected', error=str(exc), test_accessed=False, scope=SCOPE))
        seal(output, dict(status='diagnosis_rejected', test_accessed=False))
        raise


def replay(capture_root, historical_reference, historical_candidate, output):
    output = fresh_output(output)
    journal = []
    try:
        capture_root = admission.independent_path(capture_root)
        sealed = load_seal(capture_root)
        require_environment()
        require_clean_checkout()
        provenance = json.loads((capture_root / 'CAPTURE.json').read_text())
        if (sealed.get('status') != 'capture_bound' or
                provenance.get('producer_commit') != admission.current_commit() or
                provenance.get('historical_producer') != HISTORICAL_PRODUCER or
                provenance.get('test_accessed') is not False or
                provenance.get('historical_candidate_files') !=
                inventory(Path(historical_candidate) / 'base')):
            raise ValueError('capture historical hash/producer binding differs')
        candidate_meta, candidate, candidate_source = historical(historical_candidate, 'candidate')
        reference_meta, reference, reference_source = historical(historical_reference, 'reference')
        reference_files = inventory(Path(historical_reference) / 'base')
        fresh_meta, fresh, _ = admission.load_profile(capture_root, 'base', 'candidate')
        admission.compare_numeric(fresh, candidate)
        compare_historical(candidate, reference)
        if (candidate_source != reference_source or
                candidate_meta['hostname'] == reference_meta['hostname'] or
                candidate_meta['cudnn_version'] != reference_meta['cudnn_version'] or
                provenance.get('historical_no_test_inventory') != candidate_meta['no_test_inventory']):
            raise ValueError('historical host/source/no-test provenance differs')
        # Only weights-only tensor evidence is deserialized; no model or dataset is loaded.
        import torch
        payload = torch.load(capture_root / 'RAW_FIM.pt', map_location='cpu', weights_only=True)
        candidate_rows = validate_payload(payload, candidate, fresh_meta['config'])
        uuid = admission.GPUS['reference'][0]
        gate = admission.wait_gate('reference', uuid, journal)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if torch.cuda.device_count() != 1:
            raise ValueError('replay must see exactly one reference GPU')
        if (socket.gethostname() != reference_meta['hostname'] or
                smoke.runtime_identity() != reference['runtime'] or
                list(torch.cuda.get_device_capability(0)) != reference_meta['compute_capability'] or
                torch.backends.cudnn.version() != reference_meta['cudnn_version'] or
                torch.cuda.get_device_name(0) != admission.GPUS['reference'][1]):
            raise ValueError('replay reference host/runtime/device differs')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(admission.BUDGET_MIB * 1024**2 / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        launch = admission.query_gpu(uuid)
        journal.append(dict(phase='launch_recheck', **launch))
        require_driver(launch, reference_meta)
        if launch['free_mib'] < admission.THRESHOLDS['reference'] or launch['name'] != admission.GPUS['reference'][1]:
            raise ValueError('replay launch resource recheck failed')
        reference_rows, reference_payload = reconstruct_reference(
            payload, reference, reference_meta['config'],
            raw_output=output / 'REFERENCE_REDUCTIONS.pt')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved(0)
        if peak > admission.BUDGET_MIB * 1024**2:
            raise ValueError('replay allocator budget exceeded')
        if (reference_files != inventory(Path(historical_reference) / 'base') or
                provenance['historical_candidate_files'] != inventory(Path(historical_candidate) / 'base')):
            raise ValueError('historical evidence changed during replay')
        result = compare_rows(candidate_rows, reference_rows)
        result.update(raw_inputs_bitwise_identical=True, historical_scores_reconstructed=8,
            historical_task_masks_reconstructed=2, all_other_historical_numeric_fields_exact=True,
            historical_candidate_files=inventory(Path(historical_candidate) / 'base'),
            historical_reference_files=reference_files,
            capture_seal_sha256=admission.sha(capture_root / 'SEALED.json'),
            historical_candidate_no_test=candidate_meta['no_test_inventory'],
            historical_reference_no_test=reference_meta['no_test_inventory'],
            candidate_driver=provenance['capture_driver'],
            reference_driver=launch['driver_version'],
            historical_candidate_driver=candidate_meta['launch_recheck']['driver_version'],
            historical_reference_driver=reference_meta['launch_recheck']['driver_version'],
            historical_task_mask_sha256=[row['fim_masks_sha256'] for row in reference['states']],
            allocator_budget_mib=admission.BUDGET_MIB, allocator_peak_reserved_bytes=peak,
            allocator_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            gpu_uuid=uuid, hostname=socket.gethostname(), gate=gate, launch_recheck=launch,
            runtime=smoke.runtime_identity(), producer_commit=admission.current_commit())
        admission.write(output / 'DIAGNOSIS.json', result)
        admission.write(output / 'GPU_RESOURCE_SAMPLES.json', journal)
        seal(output, dict(status=result['status'], test_accessed=False))
        return result
    except Exception as exc:
        admission.write(output / 'DIAGNOSIS_FAILURE.json',
                        dict(status='diagnosis_rejected', error=str(exc), test_accessed=False, scope=SCOPE))
        admission.write(output / 'GPU_RESOURCE_SAMPLES.json', journal)
        seal(output, dict(status='diagnosis_rejected', test_accessed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check-environment')
    capturing = commands.add_parser('capture')
    capturing.add_argument('--code-root', type=Path, required=True)
    replaying = commands.add_parser('replay')
    replaying.add_argument('--capture', type=Path, required=True)
    replaying.add_argument('--historical-reference', type=Path, required=True)
    for sub in (capturing, replaying):
        sub.add_argument('--historical-candidate', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check-environment':
        require_environment()
        return
    if args.command == 'capture':
        result = capture(args.code_root, args.output, args.historical_candidate)
    else:
        result = replay(args.capture, args.historical_reference, args.historical_candidate, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
