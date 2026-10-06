"""Immutable audit evidence and post-freeze adaptive evaluation."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import secrets
from types import SimpleNamespace
from collections.abc import Mapping

import torch

from adaptive_head_consolidation import (
    ADAPTIVE_METHOD_VERSION,
    BIAS_BRANCH_CONFIG,
    FULL_BRANCH_CONFIG,
    AdaptiveConsolidationResult,
    FrozenAdaptiveCandidates,
    adaptive_candidate_log_probabilities,
    build_adaptive_diagnostics,
    install_and_reload_verify,
    solve_global_mixture_weight,
)
from head_consolidation import freeze_state, hash_top_state
from metrics import MetricsTracker, evaluate_per_task_full_cached
from models import TopModel, build_models
from vfl_trainer import VFLTrainer


AUDIT_VERSION = 1
SOURCE_FILES = (
    'adaptive_consolidation_audit.py',
    'adaptive_head_consolidation.py',
    'cl_methods/proto_evolve.py',
    'runner.py',
    'metrics.py',
    'models.py',
    'head_consolidation.py',
    'data_utils.py',
    'vfl_trainer.py',
    'determinism.py',
    'calibration_split.py',
    'config.py',
    'cl_methods/sanitize.py',
    'cl_methods/__init__.py',
    'cl_methods/proto_evolve_radapt.py',
    'ul_methods/__init__.py',
    'ul_methods/fedau.py',
    'ul_methods/fedosd.py',
    'ul_methods/fedup.py',
    'ul_methods/fucrt.py',
    'ul_methods/fudp.py',
    'ul_methods/gradient_ascent.py',
    'ul_methods/luv.py',
    'ul_methods/mode.py',
    'ul_methods/radapt_router.py',
    'ul_methods/retrain.py',
    'ul_methods/roar.py',
)
FORMAL_SOURCE_FILES = (
    'adaptive_consolidation_audit.py', 'adaptive_dual_branch_validation.py',
    'adaptive_head_consolidation.py', 'adaptive_tinyimagenet_heldout.py',
    'bic_calibration.py', 'calibration_split.py', 'cl_methods/__init__.py',
    'cl_methods/adagauss.py', 'cl_methods/afc.py', 'cl_methods/der_pp.py',
    'cl_methods/er.py', 'cl_methods/er_ace.py', 'cl_methods/ewc.py',
    'cl_methods/fedprotip_vfl.py', 'cl_methods/finetune.py',
    'cl_methods/gpm.py', 'cl_methods/lwf.py', 'cl_methods/lwf_wa.py',
    'cl_methods/proto_evolve.py', 'cl_methods/proto_fedspace.py',
    'cl_methods/target.py', 'config.py', 'data_utils.py', 'determinism.py',
    'head_consolidation.py', 'main.py', 'metrics.py', 'models.py', 'runner.py',
    'three_dataset_formal_runtime.py', 'ul_methods/__init__.py',
    'ul_methods/retrain.py', 'vfl_trainer.py',
)
_FORMAL_GIT_SOURCE_CACHE = {}
_DATA_FLOW = {
    'candidates_frozen_before_validation': True,
    'validation_before_freeze': False,
    'test_before_install': False,
    'test_used_for_diagnostics': False,
    'solver_input': 'class_balanced_validation_nll',
}


def _strict_json(value):
    return json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
        separators=(',', ': '),
    ) + '\n'


def _formal_source_provenance(root=None):
    """Hash the exact producer bytes and require equality with current Git."""
    root = Path(root or Path(__file__).resolve().parent).resolve()
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.PIPE,
        ).strip()
        common = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--git-common-dir'],
            text=True, stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('formal source Git identity is unavailable') from error
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('formal source commit is invalid')
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (root / common_path).resolve()
    cache_key = (str(common_path), commit)
    committed = _FORMAL_GIT_SOURCE_CACHE.get(cache_key)
    if committed is None:
        committed = {}
        for logical in FORMAL_SOURCE_FILES:
            try:
                content = subprocess.check_output(
                    ['git', '-C', str(root), 'show', f'{commit}:{logical}'],
                    stderr=subprocess.PIPE,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    f'formal source Git blob is unavailable: {logical}'
                ) from error
            committed[logical] = hashlib.sha256(content).hexdigest()
        _FORMAL_GIT_SOURCE_CACHE[cache_key] = committed
    current = {}
    for logical in FORMAL_SOURCE_FILES:
        try:
            digest = hashlib.sha256((root / logical).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f'formal source is unavailable: {logical}') from error
        if digest != committed[logical]:
            raise ValueError(
                f'formal source differs from Git commit: {logical}'
            )
        current[logical] = digest
    return {
        'schema_version': 1,
        'source_commit': commit,
        'source_sha256': current,
    }


def _validated_formal_source_provenance(value, root=None):
    current = _formal_source_provenance(root)
    if value != current:
        raise ValueError('formal source provenance changed')
    return current


@contextlib.contextmanager
def _trusted_dir(path):
    path = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(f'unsafe directory component in {path}') from error
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _atomic_write_new_bytes(path, content):
    raw_path = Path(os.fspath(path))
    if '..' in raw_path.parts:
        raise ValueError('immutable evidence path traversal is not allowed')
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.name or path.name in ('.', '..'):
        raise ValueError('immutable evidence target is invalid')
    temporary = f'.{path.name}.{secrets.token_hex(8)}.tmp'
    with _trusted_dir(path.parent) as directory:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise ValueError('symlinked evidence target is not allowed')
            raise FileExistsError(path)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
            0o600,
            dir_fd=directory,
        )
        try:
            with os.fdopen(descriptor, 'wb', closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
            written = os.fstat(descriptor)
            current = os.stat(
                temporary, dir_fd=directory, follow_symlinks=False
            )
            if (not stat.S_ISREG(current.st_mode)
                    or not _same_file_identity(written, current)):
                raise RuntimeError('immutable evidence temp changed before install')
            try:
                os.link(
                    temporary, path.name,
                    src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise FileExistsError(path) from None
            installed = os.stat(
                path.name, dir_fd=directory, follow_symlinks=False
            )
            if not _same_file_identity(os.fstat(descriptor), installed):
                raise RuntimeError('immutable evidence installed wrong inode')
            os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                current = os.stat(
                    temporary, dir_fd=directory, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if not _same_file_identity(os.fstat(descriptor), current):
                    raise RuntimeError(
                        'immutable evidence temp changed during cleanup'
                    )
                os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
            finally:
                os.close(descriptor)
    return path


def _ensure_child_dir(root, name):
    if not name or os.path.sep in name or name in ('.', '..'):
        raise ValueError('adaptive child directory name is invalid')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with _trusted_dir(root) as directory:
        try:
            os.mkdir(name, mode=0o755, dir_fd=directory)
            os.fsync(directory)
        except FileExistsError:
            pass
        try:
            child = os.open(name, flags, dir_fd=directory)
        except OSError as error:
            raise ValueError('unsafe adaptive child directory') from error
        os.close(child)
    return Path(root) / name


def atomic_write_new_json(path, payload):
    """Durably create one immutable JSON record without following symlinks."""
    return _atomic_write_new_bytes(path, _strict_json(payload).encode('utf-8'))


def _safe_open(path, mode='rb'):
    path = Path(os.path.abspath(os.fspath(path)))
    with _trusted_dir(path.parent) as directory:
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=directory
        )
    return os.fdopen(descriptor, mode)


def _read_file(path):
    """Read one pinned regular-file inode and verify it stayed unchanged."""
    with _safe_open(path) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('audit evidence must be a regular file')
        content = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda details: (
        details.st_dev, details.st_ino, details.st_size,
        details.st_mtime_ns, details.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise RuntimeError('audit evidence changed while being read')
    return content, before


def _same_file_identity(first, second):
    return (
        first.st_dev, first.st_ino, first.st_size,
        first.st_mtime_ns, first.st_ctime_ns,
    ) == (
        second.st_dev, second.st_ino, second.st_size,
        second.st_mtime_ns, second.st_ctime_ns,
    )


def _same_inode_content(first, second):
    """Compare an inode across expected link/rename ctime changes."""
    return (
        first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns,
    ) == (
        second.st_dev, second.st_ino, second.st_size, second.st_mtime_ns,
    )


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _sha256(path):
    return _digest(_read_file(path)[0])


def _safe_stat(path):
    path = Path(os.path.abspath(os.fspath(path)))
    with _trusted_dir(path.parent) as directory:
        details = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
    if stat.S_ISLNK(details.st_mode):
        raise ValueError('symlinked audit file is not allowed')
    return details


def _safe_json(path):
    value = json.loads(_read_file(path)[0].decode('utf-8'))
    if type(value) is not dict:
        raise ValueError('immutable evidence JSON must be an object')
    return value


def _safe_torch_load(path):
    content, _ = _read_file(path)
    return _restricted_torch_load(content)


def _restricted_torch_load(content):
    try:
        return torch.load(
            io.BytesIO(content), map_location='cpu', weights_only=True
        )
    except Exception as error:
        raise ValueError('restricted adaptive checkpoint load failed') from error


def _tensor_sha256(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError('audit tensor evidence must use tensors')
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(repr(tuple(value.shape)).encode('ascii'))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainer_state_sha256(state):
    """Hash nested trainer tensor state without serialization metadata."""
    digest = hashlib.sha256()

    def token(tag, payload=b''):
        digest.update(len(tag).to_bytes(8, 'big'))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)

    def update(value):
        if isinstance(value, torch.Tensor):
            token(b'tensor', _tensor_sha256(value).encode('ascii'))
        elif isinstance(value, Mapping):
            ordered = sorted(
                value, key=lambda item: (type(item).__name__, repr(item))
            )
            token(b'mapping', len(ordered).to_bytes(8, 'big'))
            for key in ordered:
                update(key)
                update(value[key])
        elif type(value) in (list, tuple):
            token(type(value).__name__.encode('ascii'),
                  len(value).to_bytes(8, 'big'))
            for item in value:
                update(item)
        elif value is None or type(value) in (str, bool, int, float):
            token(type(value).__name__.encode('ascii'),
                  repr(value).encode('utf-8'))
        else:
            raise TypeError('trainer state contains unsupported hash evidence')

    update(state)
    return digest.hexdigest()


def _validated_trainer_state(state, protocol):
    if (type(state) is not dict or set(state) != {'bottoms', 'top_model'}
            or type(protocol) is not dict
            or type(protocol.get('num_parties')) is not int
            or protocol['num_parties'] < 0
            or type(state['bottoms']) is not list
            or len(state['bottoms']) != protocol['num_parties']
            or not isinstance(state['top_model'], Mapping)
            or any(not isinstance(bottom, Mapping)
                   for bottom in state['bottoms'])):
        raise ValueError('adaptive trainer state schema/count is invalid')
    return trainer_state_sha256(state)


def _strict_load_trainer_state(trainer, state, protocol):
    expected_hash = _validated_trainer_state(state, protocol)
    bottoms = getattr(trainer, 'bottoms', None)
    top = getattr(trainer, 'top_model', None)
    if bottoms is None or top is None:
        trainer.load_state(state)
    else:
        if len(bottoms) != len(state['bottoms']):
            raise ValueError('fresh trainer party count mismatch')
        for bottom, model_state in zip(bottoms, state['bottoms']):
            bottom.load_state_dict(model_state, strict=True)
        top.load_state_dict(state['top_model'], strict=True)
    if trainer_state_sha256(trainer.get_state()) != expected_hash:
        raise RuntimeError('strict full trainer reload changed state bytes')
    return expected_hash


def _reject_unsafe_tree(run_dir):
    run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    with _trusted_dir(run_dir):
        pass
    for root, directories, files in os.walk(run_dir, followlinks=False):
        for name in directories + files:
            path = Path(root) / name
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode):
                raise ValueError(f'symlinked audit path is not allowed: {path}')
            if name.endswith('.tmp') or '.tmp.' in name:
                raise ValueError(f'temporary audit remnant is not allowed: {path}')
    return run_dir


def recover_atomic_write_temps(run_dir):
    """Reconcile only exact temps left by ``_atomic_write_new_bytes``."""
    run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    with _trusted_dir(run_dir):
        pass
    pattern = re.compile(r'^\.(.+)\.([0-9a-f]{16})\.tmp$')
    for root, directories, files in os.walk(run_dir, followlinks=False):
        if any((Path(root) / name).is_symlink() for name in directories):
            raise ValueError('symlinked temp recovery directory is not allowed')
        for name in files:
            match = pattern.fullmatch(name)
            if match is None:
                continue
            path = Path(root) / name
            content, pinned = _read_file(path)
            target_name = match.group(1)
            with _trusted_dir(root) as directory:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (not stat.S_ISREG(current.st_mode)
                        or not _same_file_identity(pinned, current)):
                    raise RuntimeError('atomic evidence temp changed during recovery')
                if current.st_mode & 0o222:
                    os.unlink(name, dir_fd=directory)
                    os.fsync(directory)
                    continue
                try:
                    target = os.stat(
                        target_name, dir_fd=directory, follow_symlinks=False
                    )
                except FileNotFoundError:
                    os.link(
                        name, target_name,
                        src_dir_fd=directory, dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                    installed = os.stat(
                        target_name, dir_fd=directory, follow_symlinks=False
                    )
                    if not _same_inode_content(pinned, installed):
                        raise RuntimeError(
                            'atomic evidence recovery installed wrong inode'
                        )
                else:
                    target_path = Path(root) / target_name
                    target_content, target_pinned = _read_file(target_path)
                    target_now = os.stat(
                        target_name, dir_fd=directory, follow_symlinks=False
                    )
                    if (not stat.S_ISREG(target.st_mode)
                            or not _same_file_identity(target, target_pinned)
                            or not _same_file_identity(target_pinned, target_now)
                            or target_content != content):
                        raise ValueError('atomic evidence temp target mismatch')
                current = os.stat(
                    name, dir_fd=directory, follow_symlinks=False
                )
                if not _same_inode_content(pinned, current):
                    raise RuntimeError(
                        'atomic evidence temp changed during recovery cleanup'
                    )
                os.unlink(name, dir_fd=directory)
                os.fsync(directory)


def _replay_manifest(replay_raw, replay_embeddings):
    if (type(replay_raw) is not dict or not replay_raw
            or type(replay_embeddings) is not dict
            or set(replay_raw) != set(replay_embeddings)):
        raise ValueError('adaptive replay raw/embedding evidence is missing')
    ordered_ids, by_class, hashed = [], {}, []
    for class_id in sorted(replay_raw):
        if type(class_id) is not int:
            raise ValueError('adaptive replay class identity is invalid')
        raw_rows = replay_raw[class_id]
        embedded_rows = replay_embeddings[class_id]
        if (not isinstance(raw_rows, torch.Tensor) or raw_rows.ndim < 2
                or not raw_rows.numel()
                or not isinstance(embedded_rows, torch.Tensor)
                or embedded_rows.ndim != 2
                or embedded_rows.size(0) != raw_rows.size(0)):
            raise ValueError('adaptive replay raw/embedding rows are invalid')
        identities = []
        for raw_row, embedded_row in zip(raw_rows, embedded_rows):
            identity = _tensor_sha256(raw_row)
            identities.append(identity)
            ordered_ids.append(identity)
            hashed.append([identity, _tensor_sha256(embedded_row)])
        by_class[str(class_id)] = identities
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError('adaptive replay contains duplicate sample identities')
    return {
        'ordered_sample_ids': ordered_ids,
        'by_class': by_class,
        'count': len(ordered_ids),
        'sha256': hashlib.sha256(json.dumps(
            hashed, separators=(',', ':')
        ).encode('utf-8')).hexdigest(),
    }


def _canonical_validation_evidence(manifest, labels, ordered_classes):
    """Filter a validation manifest and return its class-major row order."""
    classes = [int(class_id) for class_id in ordered_classes]
    identity_fields = set(manifest) & {'ordered_indices', 'ordered_sample_ids'} \
        if type(manifest) is dict else set()
    required = {'dataset', 'seed', 'per_class', 'by_class', 'sha256'}
    if (len(identity_fields) != 1
            or set(manifest) != required | identity_fields
            or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1 or not classes):
        raise TypeError('adaptive validation evidence must be strict JSON schema')
    identity_field = next(iter(identity_fields))
    by_class = manifest.get('by_class')
    if type(by_class) is not dict:
        raise ValueError('adaptive validation class identities are invalid')
    filtered_by_class, row_order = {}, []
    for class_id in classes:
        identities = by_class.get(str(class_id))
        positions = torch.nonzero(
            labels.detach().cpu() == class_id, as_tuple=False
        ).flatten()
        if (type(identities) is not list or not identities
                or positions.numel() != len(identities)):
            raise ValueError('adaptive validation class count is inconsistent')
        filtered_by_class[str(class_id)] = copy.deepcopy(identities)
        row_order.extend(int(position) for position in positions.tolist())
    ordered = [
        identity for class_id in classes
        for identity in filtered_by_class[str(class_id)]
    ]
    filtered = {
        key: copy.deepcopy(manifest[key])
        for key in ('dataset', 'seed', 'per_class')
    }
    filtered['by_class'] = filtered_by_class
    filtered[identity_field] = ordered
    filtered['sha256'] = hashlib.sha256(json.dumps(
        ordered, separators=(',', ':')
    ).encode('utf-8')).hexdigest()
    return filtered, torch.tensor(row_order, dtype=torch.long)


def _top(metadata):
    if type(metadata) is not dict or set(metadata) != {
            'input_dim', 'num_classes', 'cosine'}:
        raise ValueError('adaptive checkpoint top schema is incomplete')
    if (type(metadata['input_dim']) is not int
            or type(metadata['num_classes']) is not int
            or type(metadata['cosine']) is not bool):
        raise ValueError('adaptive checkpoint top schema is invalid')
    return TopModel(
        metadata['input_dim'], metadata['num_classes'],
        cosine=metadata['cosine'],
    ).eval()


def _validated_snapshot_top(metadata, trainer_state):
    if (type(trainer_state) is not dict
            or not isinstance(trainer_state.get('top_model'), Mapping)):
        raise ValueError('snapshot top model state is invalid')
    top_state = trainer_state['top_model']
    weight = top_state.get('classifier.weight')
    if weight is None:
        if metadata is not None:
            raise ValueError('snapshot top model metadata has no classifier')
        return None
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or not weight.numel()):
        raise ValueError('snapshot top model classifier state is invalid')
    if (type(metadata) is not dict
            or set(metadata) != {'input_dim', 'num_classes', 'cosine'}
            or type(metadata.get('input_dim')) is not int
            or type(metadata.get('num_classes')) is not int
            or type(metadata.get('cosine')) is not bool
            or metadata['input_dim'] != weight.shape[1]
            or metadata['num_classes'] != weight.shape[0]
            or metadata['cosine'] != ('scale' in top_state)):
        raise ValueError('snapshot top model metadata/state mismatch')
    return metadata


def _probe_hash(value):
    return _tensor_sha256(value.to(torch.float64))


def _source_commit(root):
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
    ).strip()


def _formal_reload_sha256(payload):
    return trainer_state_sha256({
        key: payload[key]
        for key in (
            'trainer_state', 'cl_state', 'tracker_state', 'rng_state',
            'protocol', 'top_model',
        )
    })


def _formal_snapshot_method_state(protocol, state):
    """Return the minimal method state needed by a formal stage evaluator."""
    if protocol.get('cl_method') == 'er_ace':
        from cl_methods.er_ace import ERAccCL
        from runner import _decode_checkpoint_value
        method = ERAccCL(None, SimpleNamespace(**protocol))
        method.load_state(_decode_checkpoint_value(state))
        if not method.buffer.size():
            raise ValueError('formal ER-ACE requires nonempty replay state')
    if protocol.get('cl_method') != 'fedprotip_vfl':
        return state
    from cl_methods.fedprotip_vfl import FORMAL_EVALUATION_STATE_KEYS
    from runner import _decode_checkpoint_value
    decoded = _decode_checkpoint_value(state)
    if (type(decoded) is not dict
            or not FORMAL_EVALUATION_STATE_KEYS.issubset(decoded)):
        raise ValueError('formal FedProTIP evaluation state is incomplete')
    return {
        key: copy.deepcopy(decoded[key])
        for key in sorted(FORMAL_EVALUATION_STATE_KEYS)
    }


def _formal_snapshot_record(path, run_dir):
    content, _ = _read_file(path)
    payload = _restricted_torch_load(content)
    match = re.fullmatch(r'event_(\d+)_CIL\.pt', Path(path).name)
    required = {
        'schema_version', 'kind', 'protocol_kind', 'event_idx', 'task_id',
        'introduced_classes', 'seen_task_classes', 'trainer_state',
        'cl_state', 'tracker_state', 'rng_state', 'protocol', 'top_model',
        'source_identity', 'source_provenance', 'strict_reload_sha256',
    }
    source = payload.get('source_identity') if type(payload) is dict else None
    if (type(payload) is not dict or set(payload) != required
            or payload.get('schema_version') != 1
            or payload.get('kind') != 'formal_deferred_cil_snapshot'
            or payload.get('protocol_kind') != 'formal'
            or match is None
            or type(payload.get('event_idx')) is not int
            or payload['event_idx'] != int(match.group(1))
            or type(payload.get('task_id')) is not int
            or type(payload.get('introduced_classes')) is not list
            or any(type(value) is not int
                   for value in payload['introduced_classes'])
            or type(payload.get('seen_task_classes')) is not dict
            or payload['task_id'] not in payload['seen_task_classes']
            or payload['seen_task_classes'][payload['task_id']]
            != payload['introduced_classes']
            or type(payload.get('protocol')) is not dict
            or payload['protocol'].get('formal_deferred_evaluation') is not True
            or payload.get('source_provenance')
                != payload['protocol'].get('source_provenance')
            or type(source) is not dict
            or set(source) != {'path', 'sha256', 'schema_version', 'step'}
            or source.get('path')
            != f"checkpoints/event_{payload.get('event_idx')}_CIL.pt"
            or source.get('schema_version') != 4
            or source.get('step') != f"event_{payload.get('event_idx')}_CIL"
            or type(source.get('sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', source['sha256']) is None
            or type(payload.get('strict_reload_sha256')) is not str
            or re.fullmatch(
                r'[0-9a-f]{64}', payload['strict_reload_sha256']
            ) is None):
        raise ValueError('formal stage snapshot identity is invalid')
    trainer_hash = _validated_trainer_state(
        payload['trainer_state'], payload['protocol']
    )
    _validated_snapshot_top(payload['top_model'], payload['trainer_state'])
    from runner import (
        _decode_checkpoint_value,
        _valid_method_checkpoint_state,
        _valid_tracker_checkpoint_state,
        _validate_rng_checkpoint_state,
    )
    adaptive = bool(
        payload['protocol'].get('head_consolidation_enabled')
        and payload['protocol'].get('head_consolidation_mode')
        == 'adaptive_dual_branch'
    )
    try:
        method_state = _decode_checkpoint_value(payload['cl_state'])
    except Exception as error:
        raise ValueError('formal snapshot method state is invalid') from error
    fedprotip = payload['protocol'].get('cl_method') == 'fedprotip_vfl'
    if fedprotip:
        from cl_methods.fedprotip_vfl import FORMAL_EVALUATION_STATE_KEYS
    if ((fedprotip and set(method_state) != FORMAL_EVALUATION_STATE_KEYS)
            or not _valid_method_checkpoint_state(
                payload['protocol'].get('cl_method'), method_state, adaptive,
                payload['protocol'].get('num_parties'))):
        raise ValueError('formal snapshot method state is invalid')
    if not _valid_tracker_checkpoint_state(
            payload['tracker_state'], payload['protocol']):
        raise ValueError('formal snapshot tracker state is invalid')
    _validate_rng_checkpoint_state(payload['rng_state'])
    try:
        reload_hash = _formal_reload_sha256(payload)
    except (KeyError, TypeError) as error:
        raise ValueError('formal strict reload state is invalid') from error
    if reload_hash != payload['strict_reload_sha256']:
        raise ValueError('formal strict reload hash mismatch')
    source_path = Path(run_dir) / source['path']
    source_content, _ = _read_file(source_path)
    if _digest(source_content) != source['sha256']:
        raise ValueError('formal source checkpoint hash mismatch')
    source_payload = _restricted_torch_load(source_content)
    if (type(source_payload) is not dict
            or source_payload.get('schema_version') != 4
            or source_payload.get('step') != source['step']
            or source_payload.get('event_idx') != payload['event_idx']
            or source_payload.get('task_id') != payload['task_id']
            or source_payload.get('new_classes') != payload['introduced_classes']
            or source_payload.get('seen_task_classes')
            != payload['seen_task_classes']
            or source_payload.get('protocol') != payload['protocol']):
        raise ValueError('formal source checkpoint identity mismatch')
    try:
        source_reload_hash = _formal_reload_sha256({
            **source_payload,
            'cl_state': _formal_snapshot_method_state(
                payload['protocol'], source_payload['cl_state']
            ),
            'top_model': payload['top_model'],
        })
    except (KeyError, TypeError) as error:
        raise ValueError('formal source checkpoint state is invalid') from error
    if source_reload_hash != reload_hash:
        raise ValueError('formal source checkpoint state mismatch')
    return payload, {
        'path': Path(path).relative_to(run_dir).as_posix(),
        'sha256': _digest(content),
        'protocol_kind': 'formal',
        'event_idx': payload['event_idx'],
        'task_id': payload['task_id'],
        'introduced_classes': payload['introduced_classes'],
        'seen_task_classes': {
            str(task_id): classes
            for task_id, classes in sorted(payload['seen_task_classes'].items())
        },
        'protocol_sha256': _digest(
            _strict_json(payload['protocol']).encode('utf-8')
        ),
        'trainer_state_sha256': trainer_hash,
        'strict_reload_sha256': reload_hash,
        'source_identity': copy.deepcopy(source),
        'source_provenance': copy.deepcopy(payload['source_provenance']),
    }


def _snapshot_record(path, run_dir, protocol_kind='adaptive'):
    if protocol_kind == 'formal':
        return _formal_snapshot_record(path, run_dir)
    if protocol_kind != 'adaptive':
        raise ValueError('unsupported deferred snapshot protocol kind')
    content, _ = _read_file(path)
    payload = _restricted_torch_load(content)
    match = re.fullmatch(r'event_(\d+)_CIL\.pt', Path(path).name)
    required = {
        'schema_version', 'kind', 'event_idx', 'task_id',
        'introduced_classes', 'seen_task_classes', 'trainer_state',
        'cl_state', 'protocol', 'top_model',
    }
    if (type(payload) is not dict or set(payload) != required
            or payload.get('schema_version') != 1
            or payload.get('kind') != 'adaptive_deferred_cil_snapshot'
            or match is None
            or type(payload.get('event_idx')) is not int
            or payload['event_idx'] != int(match.group(1))
            or type(payload.get('task_id')) is not int
            or type(payload.get('introduced_classes')) is not list
            or any(type(value) is not int
                   for value in payload['introduced_classes'])
            or type(payload.get('seen_task_classes')) is not dict
            or payload['task_id'] not in payload['seen_task_classes']
            or payload['seen_task_classes'][payload['task_id']]
            != payload['introduced_classes']
            or type(payload.get('protocol')) is not dict):
        raise ValueError('adaptive stage snapshot identity is invalid')
    trainer_hash = _validated_trainer_state(
        payload['trainer_state'], payload['protocol']
    )
    _validated_snapshot_top(payload['top_model'], payload['trainer_state'])
    return payload, {
        'path': Path(path).relative_to(run_dir).as_posix(),
        'sha256': _digest(content),
        'event_idx': payload['event_idx'],
        'task_id': payload['task_id'],
        'introduced_classes': payload['introduced_classes'],
        'seen_task_classes': {
            str(task_id): classes
            for task_id, classes in sorted(payload['seen_task_classes'].items())
        },
        'protocol_sha256': _digest(
            _strict_json(payload['protocol']).encode('utf-8')
        ),
        'trainer_state_sha256': trainer_hash,
    }


def _snapshot_manifest(run_dir, protocol_kind='adaptive'):
    if protocol_kind == 'formal':
        root = run_dir / 'formal_snapshots'
        if not root.exists():
            return []
        all_files = [path for path in root.iterdir() if path.is_file()]
        matches = [
            (path, re.fullmatch(r'event_(\d+)_CIL\.pt', path.name))
            for path in all_files
        ]
        if any(match is None for _, match in matches):
            raise ValueError('unexpected formal stage snapshot artifact')
        all_files = [
            path for path, match in sorted(
                matches, key=lambda item: int(item[1].group(1))
            )
        ]
        records = [
            _snapshot_record(path, run_dir, protocol_kind='formal')[1]
            for path in all_files
        ]
        events = [record['event_idx'] for record in records]
        tasks = [record['task_id'] for record in records]
        if (events != sorted(set(events)) or tasks != sorted(set(tasks))
                or any(records[index]['seen_task_classes'] != {
                    str(record['task_id']): record['introduced_classes']
                    for record in records[:index + 1]
                } for index in range(len(records)))
                or len({record['protocol_sha256'] for record in records}) > 1):
            raise ValueError('formal stage snapshot sequence is inconsistent')
        return records
    if protocol_kind != 'adaptive':
        raise ValueError('unsupported deferred snapshot protocol kind')
    root = run_dir / 'adaptive_snapshots'
    if not root.exists():
        return []
    all_files = [path for path in root.iterdir() if path.is_file()]
    matches = [
        (path, re.fullmatch(r'event_(\d+)_CIL\.pt', path.name))
        for path in all_files
    ]
    if any(match is None for _, match in matches):
        raise ValueError('unexpected adaptive stage snapshot artifact')
    all_files = [
        path for path, match in sorted(
            matches, key=lambda item: int(item[1].group(1))
        )
    ]
    records = [
        _snapshot_record(path, run_dir)[1]
        for path in all_files
    ]
    events = [record['event_idx'] for record in records]
    tasks = [record['task_id'] for record in records]
    if (events != sorted(set(events)) or tasks != sorted(set(tasks))
            or any(records[index]['seen_task_classes'] != {
                str(record['task_id']): record['introduced_classes']
                for record in records[:index + 1]
            } for index in range(len(records)))
            or len({record['protocol_sha256'] for record in records}) > 1):
        raise ValueError('adaptive stage snapshot sequence is inconsistent')
    return records


def prepare_adaptive_run_provenance(run_dir):
    """Create or verify the immutable pre-checkpoint source records."""
    recover_atomic_write_temps(run_dir)
    run_dir = _reject_unsafe_tree(run_dir)
    record_paths = [
        run_dir / 'ADAPTIVE_RUN_PLANNED.json',
        run_dir / 'ADAPTIVE_RUN_LAUNCHED.json',
    ]
    missing = []
    for path in record_paths:
        try:
            _safe_stat(path)
        except FileNotFoundError:
            missing.append(path)
    if missing:
        forbidden = []
        for root, _, files in os.walk(run_dir, followlinks=False):
            for name in files:
                path = Path(root) / name
                relative = path.relative_to(run_dir)
                if (name in {'adaptive_final.pt', 'ADAPTIVE_STATE_FROZEN.json'}
                        or (path.suffix == '.pt'
                            and relative.parts[0] in {
                                'adaptive_snapshots', 'checkpoints'})):
                    forbidden.append(relative)
        if forbidden:
            raise ValueError(
                'adaptive provenance cannot be backfilled after checkpoints'
            )
    source_root = Path(__file__).resolve().parent
    core = {
        'source_version': ADAPTIVE_METHOD_VERSION,
        'source_commit': _source_commit(source_root),
        'source_sha256': {
            name: _sha256(source_root / name) for name in SOURCE_FILES
        },
    }
    for name, record in (
            ('ADAPTIVE_RUN_PLANNED.json', 'planned'),
            ('ADAPTIVE_RUN_LAUNCHED.json', 'launched')):
        path = run_dir / name
        expected = {'record': record, **core}
        try:
            _safe_stat(path)
            exists = True
        except FileNotFoundError:
            exists = False
        if exists:
            if _safe_json(path) != expected:
                raise ValueError('adaptive source provenance mismatch on resume')
        else:
            atomic_write_new_json(path, expected)
    return {
        **core,
        'planned_sha256': _sha256(run_dir / 'ADAPTIVE_RUN_PLANNED.json'),
        'launched_sha256': _sha256(run_dir / 'ADAPTIVE_RUN_LAUNCHED.json'),
        'data_flow': copy.deepcopy(_DATA_FLOW),
    }


def audit_adaptive_checkpoint(run_dir, expected_spec):
    """Recompute immutable adaptive evidence without changing ``run_dir``."""
    run_dir = _reject_unsafe_tree(run_dir)
    if type(expected_spec) is not dict:
        raise TypeError('adaptive audit spec must be a dictionary')
    required = {
        'source_version', 'source_commit', 'source_sha256', 'checkpoint',
        'planned_sha256', 'launched_sha256', 'data_flow',
    }
    if set(expected_spec) != required:
        raise ValueError('adaptive audit spec is incomplete')
    if (type(expected_spec['source_version']) is not int
            or expected_spec['source_version'] != ADAPTIVE_METHOD_VERSION
            or type(expected_spec['source_commit']) is not str
            or type(expected_spec['checkpoint']) is not str
            or Path(expected_spec['checkpoint']).name != expected_spec['checkpoint']
            or expected_spec['data_flow'] != _DATA_FLOW):
        raise ValueError('adaptive audit source/data-flow identity is invalid')

    source_root = Path(__file__).resolve().parent
    expected_sources = expected_spec['source_sha256']
    if type(expected_sources) is not dict or set(expected_sources) != set(SOURCE_FILES):
        raise ValueError('adaptive source identity is incomplete')
    actual_sources = {name: _sha256(source_root / name) for name in SOURCE_FILES}
    actual_commit = _source_commit(source_root)
    if (actual_sources != expected_sources
            or actual_commit != expected_spec['source_commit']):
        raise ValueError('adaptive source hash/commit mismatch')

    planned = run_dir / 'ADAPTIVE_RUN_PLANNED.json'
    launched = run_dir / 'ADAPTIVE_RUN_LAUNCHED.json'
    checkpoint = run_dir / expected_spec['checkpoint']
    planned_content, planned_details = _read_file(planned)
    launched_content, launched_details = _read_file(launched)
    checkpoint_content, checkpoint_details = _read_file(checkpoint)
    if (_digest(planned_content) != expected_spec['planned_sha256']
            or _digest(launched_content) != expected_spec['launched_sha256']):
        raise ValueError('adaptive planned/launch provenance mismatch')
    core = {
        'source_version': expected_spec['source_version'],
        'source_commit': expected_spec['source_commit'],
        'source_sha256': expected_sources,
    }
    try:
        planned_record = json.loads(planned_content.decode('utf-8'))
        launched_record = json.loads(launched_content.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('adaptive provenance JSON is invalid') from error
    if (planned_record != {'record': 'planned', **core}
            or launched_record != {'record': 'launched', **core}):
        raise ValueError('adaptive planned/launch provenance is invalid')
    if (planned_details.st_mtime_ns > checkpoint_details.st_mtime_ns
            or launched_details.st_mtime_ns > checkpoint_details.st_mtime_ns
            or planned_details.st_ctime_ns > checkpoint_details.st_ctime_ns
            or launched_details.st_ctime_ns > checkpoint_details.st_ctime_ns):
        raise ValueError('adaptive provenance was written after the checkpoint')

    payload = _restricted_torch_load(checkpoint_content)
    required_payload = {
        'schema_version', 'kind', 'provenance', 'protocol', 'top_model',
        'trainer_state', 'cl_state',
    }
    if (type(payload) is not dict or set(payload) != required_payload
            or payload.get('schema_version') != 1
            or payload.get('kind') != 'adaptive_final_checkpoint'
            or type(payload.get('protocol')) is not dict):
        raise ValueError('adaptive checkpoint schema mismatch')
    if payload.get('provenance') != {
            'source_version': expected_spec['source_version'],
            'source_commit': expected_spec['source_commit'],
            'planned_sha256': expected_spec['planned_sha256'],
            'launched_sha256': expected_spec['launched_sha256'],
            }:
        raise ValueError('adaptive checkpoint provenance is missing or stale')
    cl_state = payload.get('cl_state')
    bundle = cl_state.get('adaptive_audit_bundle') if type(cl_state) is dict else None
    if type(bundle) is not dict:
        raise ValueError('adaptive checkpoint audit bundle is missing')
    full_trainer_hash = _validated_trainer_state(
        payload['trainer_state'], payload['protocol']
    )
    if payload['protocol']['num_parties']:
        isolated_args = SimpleNamespace(**copy.deepcopy(payload['protocol']))
        isolated_args.device = 'cpu'
        try:
            audit_trainer = _fresh_trainer(payload, isolated_args)
            _strict_load_trainer_state(
                audit_trainer, payload['trainer_state'], payload['protocol']
            )
        except Exception as error:
            raise ValueError('adaptive full trainer strict reload failed') from error
    expected_order = [
        'candidates_frozen', 'validation_iterated', 'gate_solved',
        'state_installed', 'diagnostics_computed',
    ]
    if bundle.get('event_order') != expected_order:
        raise ValueError('adaptive checkpoint data-flow order mismatch')
    base_data_flow = {
        'candidates_frozen_before_validation': (
            bundle['event_order'].index('candidates_frozen')
            < bundle['event_order'].index('validation_iterated')
        ),
        'validation_before_freeze': False,
        'test_before_install': 'test_iterated' in bundle['event_order'],
        'test_used_for_diagnostics': 'test_diagnostics' in bundle['event_order'],
        'solver_input': 'class_balanced_validation_nll',
    }
    data_flow_path = run_dir / 'data_flow_audit.jsonl'
    try:
        data_flow_content, _ = _read_file(data_flow_path)
    except FileNotFoundError:
        data_flow_content = b''
    lines = data_flow_content.splitlines(keepends=True)
    try:
        records = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('adaptive data-flow audit log is invalid') from error
    freeze_path = run_dir / 'ADAPTIVE_STATE_FROZEN.json'
    try:
        frozen_record = _safe_json(freeze_path)
    except FileNotFoundError:
        frozen_record = None
    if frozen_record is None:
        if any(record.get('split') == 'test' for record in records):
            base_data_flow['test_before_install'] = True
        data_flow = {
            **base_data_flow,
            'audit_prefix_record_count': len(records),
            'audit_prefix_sha256': _digest(data_flow_content),
        }
    else:
        data_flow = frozen_record.get('data_flow')
        if type(data_flow) is not dict:
            raise ValueError('frozen data-flow prefix evidence is missing')
        prefix_count = data_flow.get('audit_prefix_record_count')
        if (type(prefix_count) is not int or prefix_count < 0
                or prefix_count > len(records)):
            raise ValueError('frozen data-flow prefix count is invalid')
        prefix = b''.join(lines[:prefix_count])
        if (data_flow.get('audit_prefix_sha256') != _digest(prefix)
                or any(record.get('split') == 'test'
                       for record in records[:prefix_count])):
            raise ValueError('frozen data-flow prefix no longer matches')
        if {key: data_flow.get(key) for key in _DATA_FLOW} != _DATA_FLOW:
            raise ValueError('frozen data-flow flags mismatch')
    if base_data_flow != _DATA_FLOW:
        raise ValueError('adaptive checkpoint data-flow flags mismatch')
    result = AdaptiveConsolidationResult.from_dict(bundle.get('result')).to_dict()
    if bundle.get('method_version') != ADAPTIVE_METHOD_VERSION:
        raise ValueError('adaptive checkpoint method version mismatch')

    metadata = payload.get('top_model')
    pre = _top(metadata)
    installed = _top(metadata)
    branch_states = {
        'full': bundle.get('full_state'), 'bias': bundle.get('bias_state'),
    }
    inactive = {
        branch
        for branch, config in result['candidate_configs'].items()
        if config == {'mode': 'inactive', 'parameters': 0}
    }
    branches = {}
    try:
        pre.load_state_dict(bundle['pre_state'], strict=True)
        for branch, state in branch_states.items():
            if branch in inactive:
                if type(state) is not dict or state:
                    raise ValueError('inactive adaptive branch state is not empty')
            else:
                branches[branch] = _top(metadata)
                branches[branch].load_state_dict(state, strict=True)
        installed.load_state_dict(bundle['installed_state'], strict=True)
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError('adaptive checkpoint branch reload failed') from error
    actual_hashes = {
        'pre': hash_top_state(pre),
        'full': hash_top_state(branch_states['full']),
        'bias': hash_top_state(branch_states['bias']),
    }
    if (actual_hashes != result['candidate_hashes']
            or result['candidate_configs'] != {
                'full': FULL_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG,
            }):
        raise ValueError('adaptive candidate hash/config mismatch')

    replay = bundle.get('replay_embeddings')
    replay_raw = bundle.get('replay_raw')
    replay_manifest = _replay_manifest(replay_raw, replay)
    claimed_replay = bundle.get('replay_manifest')
    if (type(claimed_replay) is not dict
            or len(set(claimed_replay.get('ordered_sample_ids', ())))
            != len(claimed_replay.get('ordered_sample_ids', ()))
            or claimed_replay != replay_manifest):
        raise ValueError('adaptive replay identity/hash/count mismatch')
    persistent_raw = cl_state.get('head_raw_replay')
    if (type(persistent_raw) is not dict
            or set(persistent_raw) != set(replay_raw)
            or any(_tensor_sha256(persistent_raw[class_id])
                   != _tensor_sha256(replay_raw[class_id])
                   for class_id in replay_raw)):
        raise ValueError('adaptive replay is not the persistent training replay')
    if cl_state.get('head_task_classes') != bundle.get('task_classes'):
        raise ValueError('adaptive task classes were not sanitized persistently')
    validation_x = bundle.get('validation_embeddings')
    validation_y = bundle.get('validation_labels')
    validation_manifest = result['validation_manifest']
    expected_validation_labels = [
        int(class_id)
        for class_id in sorted(validation_manifest['by_class'], key=int)
        for _ in validation_manifest['by_class'][class_id]
    ]
    if (not isinstance(validation_x, torch.Tensor)
            or not isinstance(validation_y, torch.Tensor)
            or bundle.get('validation_embeddings_sha256') != _tensor_sha256(validation_x)
            or bundle.get('validation_labels_sha256') != _tensor_sha256(validation_y)
            or validation_x.size(0) != validation_y.numel()
            or validation_y.detach().cpu().tolist()
            != expected_validation_labels
            or validation_y.numel() != len(
                validation_manifest.get('ordered_indices',
                validation_manifest.get('ordered_sample_ids', ()))
            )):
        raise ValueError('adaptive validation identity/hash/count mismatch')

    candidates = FrozenAdaptiveCandidates(
        pre_head_sha256=actual_hashes['pre'],
        full_state=freeze_state(branch_states['full']),
        bias_state=freeze_state(branch_states['bias']),
        full_head_sha256=actual_hashes['full'],
        bias_head_sha256=actual_hashes['bias'],
        full_audit={}, bias_audit={},
        ordered_classes=tuple(result['ordered_classes']),
    )
    full_p, bias_p = adaptive_candidate_log_probabilities(
        pre, candidates, validation_x
    )
    solver = solve_global_mixture_weight(
        full_p, bias_p, validation_y, result['ordered_classes']
    )
    if solver != result['gate']:
        raise ValueError('adaptive solver evidence mismatch')
    expected_installed = install_and_reload_verify(pre, candidates, solver)
    expected_installed_hash = hash_top_state(expected_installed)
    if hash_top_state(installed) != expected_installed_hash:
        raise ValueError('adaptive installed state is not derived from candidates/g')
    diagnostics = build_adaptive_diagnostics(
        pre, expected_installed, candidates, replay, validation_x, validation_y,
        bundle.get('task_classes'), solver,
    )
    if diagnostics != bundle.get('diagnostics'):
        raise ValueError('adaptive diagnostics mismatch')

    live_state = payload.get('trainer_state', {}).get('top_model')
    fresh = _top(metadata)
    try:
        fresh.load_state_dict(live_state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError('adaptive final checkpoint strict reload failed') from error
    installed_hash = expected_installed_hash
    if hash_top_state(fresh) != installed_hash:
        raise RuntimeError('adaptive final checkpoint does not match installed state')
    seed = int(actual_hashes['pre'][:16], 16)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    probe = torch.randn(7, metadata['input_dim'], generator=generator)
    probe_full, probe_bias = fresh.branch_log_probabilities(probe)
    mixed = fresh(probe)
    fresh_again = _top(metadata)
    fresh_again.load_state_dict(live_state, strict=True)
    if (hash_top_state(fresh_again) != installed_hash
            or not torch.equal(fresh_again(probe), mixed)):
        raise RuntimeError('adaptive deterministic fresh reload probe mismatch')
    snapshots = _snapshot_manifest(run_dir)
    expected_tasks = payload['protocol'].get('num_tasks')
    if (type(expected_tasks) is not int or expected_tasks <= 0
            or len(snapshots) != expected_tasks
            or [snapshot['task_id'] for snapshot in snapshots]
            != list(range(expected_tasks))):
        raise ValueError('adaptive stage snapshots are incomplete')
    final_protocol_sha256 = _digest(
        _strict_json(payload['protocol']).encode('utf-8')
    )
    if any(snapshot['protocol_sha256'] != final_protocol_sha256
           for snapshot in snapshots):
        raise ValueError('adaptive stage/final protocol mismatch')
    boundary_match = re.fullmatch(
        r'event_(\d+)_(CIL|UL)', result['task_boundary']
    )
    if boundary_match is None:
        raise ValueError('adaptive result boundary is invalid')
    boundary_index = int(boundary_match.group(1))
    boundary_type = boundary_match.group(2)
    boundary_evidence = None
    if snapshots and boundary_type == 'CIL':
        matches = [
            snapshot for snapshot in snapshots
            if snapshot['event_idx'] == boundary_index
            and snapshot['task_id'] == result['task_id']
        ]
        if (len(matches) != 1
                or matches[0]['trainer_state_sha256'] != full_trainer_hash):
            raise ValueError('adaptive CIL result boundary is not installed state')
        boundary_evidence = {
            'type': 'CIL', 'event_idx': boundary_index,
            'sha256': matches[0]['sha256'],
        }
    elif snapshots and boundary_type == 'UL':
        boundary_path = (
            run_dir / 'checkpoints' / f'event_{boundary_index}_UL.pt'
        )
        boundary_content, _ = _read_file(boundary_path)
        boundary_payload = _restricted_torch_load(boundary_content)
        required_boundary = {
            'schema_version', 'step', 'event_idx', 'task_id', 'new_classes',
            'seen_task_classes', 'forgotten_classes', 'trainer_state',
            'cl_state', 'bic_state', 'bic_history', 'tracker_state',
            'rng_state', 'protocol',
        }
        boundary_state = boundary_payload.get('trainer_state') \
            if type(boundary_payload) is dict else None
        boundary_cl_state = boundary_payload.get('cl_state') \
            if type(boundary_payload) is dict else None
        boundary_bundle = boundary_cl_state.get('adaptive_audit_bundle') \
            if type(boundary_cl_state) is dict else None
        if (type(boundary_payload) is not dict
                or set(boundary_payload) != required_boundary
                or boundary_payload.get('schema_version') != 4
                or boundary_payload.get('protocol') != payload['protocol']):
            raise ValueError('adaptive UL boundary schema mismatch')
        boundary_hash = _validated_trainer_state(
            boundary_state, boundary_payload['protocol']
        )
        if payload['protocol']['num_parties']:
            isolated_args = SimpleNamespace(**copy.deepcopy(payload['protocol']))
            isolated_args.device = 'cpu'
            try:
                boundary_trainer = _fresh_trainer(payload, isolated_args)
                _strict_load_trainer_state(
                    boundary_trainer, boundary_state,
                    boundary_payload['protocol'],
                )
            except Exception as error:
                raise ValueError(
                    'adaptive UL boundary strict reload failed'
                ) from error
        if (boundary_payload.get('event_idx') != boundary_index
                or boundary_payload.get('step')
                != f'event_{boundary_index}_UL'
                or boundary_hash != full_trainer_hash
                or type(boundary_bundle) is not dict
                or boundary_bundle.get('result') != result):
            raise ValueError('adaptive UL result boundary checkpoint mismatch')
        boundary_evidence = {
            'type': 'UL', 'event_idx': boundary_index,
            'sha256': _digest(boundary_content),
        }
    if boundary_evidence is None:
        raise ValueError('adaptive result boundary evidence is missing')

    return {
        'status': 'ADAPTIVE_STATE_FROZEN',
        'audit_version': AUDIT_VERSION,
        'source': core,
        'result': result,
        'candidate_hashes': actual_hashes,
        'candidate_configs': result['candidate_configs'],
        'replay': replay_manifest,
        'validation': {
            'manifest': result['validation_manifest'],
            'count': int(validation_y.numel()),
            'embeddings_sha256': _tensor_sha256(validation_x),
            'labels_sha256': _tensor_sha256(validation_y),
        },
        'ordered_classes': result['ordered_classes'],
        'solver': solver,
        'diagnostics': diagnostics,
        'checkpoint': {
            'path': checkpoint.name,
            'sha256': _digest(checkpoint_content),
            'installed_top_sha256': installed_hash,
            'trainer_state_sha256': full_trainer_hash,
            'protocol_sha256': final_protocol_sha256,
        },
        'data_flow': data_flow,
        'probes': {
            'seed': seed,
            'full': _probe_hash(probe_full),
            'bias': _probe_hash(probe_bias),
            'mixed': _probe_hash(mixed),
        },
        'strict_fresh_reload': True,
        'snapshots': snapshots,
        'result_boundary': boundary_evidence,
        'planned_sha256': expected_spec['planned_sha256'],
        'launched_sha256': expected_spec['launched_sha256'],
        'audit_spec': copy.deepcopy(expected_spec),
    }


def _snapshot_args(args):
    return {
        key: copy.deepcopy(value)
        for key, value in vars(args).items()
        if key not in {'output_dir', 'resume_run_dir'}
        and isinstance(value, (type(None), str, bool, int, float, list, tuple, dict))
    }


def _torch_bytes(payload):
    handle = io.BytesIO()
    torch.save(payload, handle)
    return handle.getvalue()


def _save_formal_cil_snapshot(trainer, args, event_idx, task_id,
                              seen_task_classes):
    task_id = int(task_id)
    event_idx = int(event_idx)
    task_classes = {
        int(key): [int(class_id) for class_id in value]
        for key, value in seen_task_classes.items()
    }
    if task_id not in task_classes:
        raise ValueError('deferred snapshot is missing its introduced task')
    checkpoint_path = (
        Path(args.output_dir) / 'checkpoints' / f'event_{event_idx}_CIL.pt'
    )
    checkpoint_content, _ = _read_file(checkpoint_path)
    checkpoint = _restricted_torch_load(checkpoint_content)
    if (type(checkpoint) is not dict
            or checkpoint.get('schema_version') != 4
            or checkpoint.get('step') != f'event_{event_idx}_CIL'
            or checkpoint.get('event_idx') != event_idx
            or checkpoint.get('task_id') != task_id
            or checkpoint.get('new_classes') != task_classes[task_id]
            or checkpoint.get('seen_task_classes') != task_classes
            or type(checkpoint.get('protocol')) is not dict
            or checkpoint['protocol'].get('formal_deferred_evaluation') is not True
            or any(key not in checkpoint for key in (
                'trainer_state', 'cl_state', 'tracker_state', 'rng_state'
            ))):
        raise ValueError('formal source checkpoint identity is invalid')
    directory = _ensure_child_dir(args.output_dir, 'formal_snapshots')
    top = getattr(trainer, 'top_model', None)
    metadata = None
    if (top is not None and hasattr(top, 'classifier')
            and hasattr(top.classifier, 'in_features')
            and hasattr(top.classifier, 'out_features')):
        metadata = {
            'input_dim': int(top.classifier.in_features),
            'num_classes': int(top.classifier.out_features),
            'cosine': bool(getattr(top, 'cosine', False)),
        }
    payload = {
        'schema_version': 1,
        'kind': 'formal_deferred_cil_snapshot',
        'protocol_kind': 'formal',
        'event_idx': event_idx,
        'task_id': task_id,
        'introduced_classes': task_classes[task_id],
        'seen_task_classes': task_classes,
        'trainer_state': checkpoint['trainer_state'],
        'cl_state': _formal_snapshot_method_state(
            checkpoint['protocol'], checkpoint['cl_state']
        ),
        'tracker_state': checkpoint['tracker_state'],
        'rng_state': checkpoint['rng_state'],
        'protocol': checkpoint['protocol'],
        'source_provenance': copy.deepcopy(
            checkpoint['protocol']['source_provenance']
        ),
        'top_model': metadata,
        'source_identity': {
            'path': f'checkpoints/event_{event_idx}_CIL.pt',
            'sha256': _digest(checkpoint_content),
            'schema_version': 4,
            'step': f'event_{event_idx}_CIL',
        },
    }
    payload['strict_reload_sha256'] = _formal_reload_sha256(payload)
    target = directory / f'event_{event_idx}_CIL.pt'
    content = _torch_bytes(payload)
    try:
        existing, _ = _read_file(target)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != content:
            raise FileExistsError(target)
        return target
    return _atomic_write_new_bytes(target, content)


def save_deferred_cil_snapshot(trainer, cl_method, args, event_idx, task_id,
                               seen_task_classes, protocol_kind='adaptive'):
    """Save one deterministic training-only stage snapshot exactly once."""
    if protocol_kind == 'formal':
        return _save_formal_cil_snapshot(
            trainer, args, event_idx, task_id, seen_task_classes
        )
    if protocol_kind != 'adaptive':
        raise ValueError('unsupported deferred snapshot protocol kind')
    task_id = int(task_id)
    event_idx = int(event_idx)
    task_classes = {
        int(key): [int(class_id) for class_id in value]
        for key, value in seen_task_classes.items()
    }
    if task_id not in task_classes:
        raise ValueError('deferred snapshot is missing its introduced task')
    directory = _ensure_child_dir(args.output_dir, 'adaptive_snapshots')
    top = getattr(trainer, 'top_model', None)
    metadata = None
    if (top is not None and hasattr(top, 'classifier')
            and hasattr(top.classifier, 'in_features')
            and hasattr(top.classifier, 'out_features')):
        metadata = {
            'input_dim': int(top.classifier.in_features),
            'num_classes': int(top.classifier.out_features),
            'cosine': bool(getattr(top, 'cosine', False)),
        }
    payload = {
        'schema_version': 1,
        'kind': 'adaptive_deferred_cil_snapshot',
        'event_idx': event_idx,
        'task_id': task_id,
        'introduced_classes': task_classes[task_id],
        'seen_task_classes': task_classes,
        'trainer_state': trainer.get_state(),
        # Evaluation needs model state only. Keeping method replay/validation out
        # of stage artifacts makes the pre-freeze data boundary explicit.
        'cl_state': {},
        'protocol': _snapshot_args(args),
        'top_model': metadata,
    }
    target = directory / f'event_{event_idx}_CIL.pt'
    content = _torch_bytes(payload)
    try:
        existing, _ = _read_file(target)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != content:
            raise FileExistsError(target)
        return target
    return _atomic_write_new_bytes(target, content)


def save_final_adaptive_checkpoint(trainer, cl_method, args, provenance):
    """Persist the installed adaptive state and its audit bundle exactly once."""
    top = trainer.top_model
    payload = {
        'schema_version': 1,
        'kind': 'adaptive_final_checkpoint',
        'provenance': {
            'source_version': provenance['source_version'],
            'source_commit': provenance['source_commit'],
            'planned_sha256': provenance['planned_sha256'],
            'launched_sha256': provenance['launched_sha256'],
        },
        'trainer_state': trainer.get_state(),
        'cl_state': cl_method.get_state(),
        'protocol': _snapshot_args(args),
        'top_model': {
            'input_dim': int(top.classifier.in_features),
            'num_classes': int(top.classifier.out_features),
            'cosine': bool(getattr(top, 'cosine', False)),
        },
    }
    target = Path(args.output_dir) / 'adaptive_final.pt'
    content = _torch_bytes(payload)
    try:
        existing, _ = _read_file(target)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != content:
            raise FileExistsError(target)
        return target
    return _atomic_write_new_bytes(target, content)


def load_frozen_adaptive_evidence(run_dir):
    return _safe_json(Path(run_dir) / 'ADAPTIVE_STATE_FROZEN.json')


def _fresh_trainer(payload, args):
    metadata = payload.get('top_model')
    isolated_args = copy.deepcopy(args)
    bottoms, top = build_models(isolated_args)
    if metadata is not None:
        top = _top(metadata).to(isolated_args.device)
    return VFLTrainer(bottoms, top, isolated_args)


def _task_readouts(trainer, loader, classes):
    accuracy, probabilities, labels = trainer.evaluate(loader)
    classes = [int(class_id) for class_id in classes]
    if probabilities.ndim != 2 or labels.ndim != 1 or labels.numel() == 0:
        raise ValueError('deferred test evaluation returned invalid probabilities')
    top = getattr(trainer, 'top_model', None)
    if (top is not None and bool(getattr(top, '_adaptive_enabled', False))):
        output_classes = [
            int(class_id) for class_id in top._adaptive_class_order.tolist()
        ]
    else:
        output_classes = list(range(probabilities.size(1)))
    if len(output_classes) != probabilities.size(1):
        raise ValueError('deferred checkpoint output identity is inconsistent')
    class_to_column = {
        class_id: column for column, class_id in enumerate(output_classes)
    }
    if any(class_id not in class_to_column for class_id in classes):
        raise ValueError('deferred task classes are absent from checkpoint output')
    global_prediction = torch.tensor(output_classes, dtype=torch.long).index_select(
        0, probabilities.argmax(dim=1)
    )
    accuracy = float((global_prediction == labels).to(torch.float64).mean())
    columns = torch.tensor(
        [class_to_column[class_id] for class_id in classes], dtype=torch.long
    )
    prediction = torch.tensor(classes, dtype=torch.long).index_select(
        0, probabilities.index_select(1, columns).argmax(dim=1)
    )
    task_il = float((prediction == labels).to(torch.float64).mean())
    return round(float(accuracy), 4), round(task_il, 4)


def evaluate_deferred_cil_trajectory(snapshot_paths, final_checkpoint, dataset,
                                     task_classes, args):
    """Evaluate diagonal/final CIL rows only after a fresh immutable audit."""
    final_checkpoint = Path(final_checkpoint)
    run_dir = final_checkpoint.parent
    freeze_path = run_dir / 'ADAPTIVE_STATE_FROZEN.json'
    freeze = _safe_json(freeze_path)
    if freeze.get('status') != 'ADAPTIVE_STATE_FROZEN':
        raise ValueError('adaptive state freeze evidence is invalid')
    audited = audit_adaptive_checkpoint(run_dir, freeze.get('audit_spec'))
    if audited != freeze:
        raise RuntimeError('adaptive state freeze evidence no longer passes audit')
    final_content, _ = _read_file(final_checkpoint)
    checkpoint_evidence = freeze.get('checkpoint')
    if (type(checkpoint_evidence) is not dict
            or checkpoint_evidence.get('sha256') != _digest(final_content)):
        raise ValueError('deferred final checkpoint no longer matches freeze')
    final_payload = _restricted_torch_load(final_content)
    frozen_snapshots = freeze.get('snapshots')
    final_protocol_sha256 = _digest(
        _strict_json(final_payload.get('protocol')).encode('utf-8')
    )
    if (type(frozen_snapshots) is not list
            or any(snapshot.get('protocol_sha256') != final_protocol_sha256
                   for snapshot in frozen_snapshots)):
        raise ValueError('deferred stage/final protocol mismatch')

    original_task_classes = {
        int(task_id): [int(class_id) for class_id in classes]
        for task_id, classes in task_classes.items()
    }
    ordered_tasks = sorted(original_task_classes)
    paths = [Path(path) for path in snapshot_paths]
    if (type(frozen_snapshots) is not list
            or len(paths) != len(ordered_tasks)
            or len(frozen_snapshots) != len(paths)):
        raise ValueError('deferred trajectory requires one snapshot per task')
    diagonals = {}
    stage_events = {}
    for expected_task, path, frozen_snapshot in zip(
            ordered_tasks, paths, frozen_snapshots):
        snapshot, actual_record = _snapshot_record(path, run_dir)
        if (actual_record != frozen_snapshot
                or actual_record['path']
                != Path(path).relative_to(run_dir).as_posix()
                or snapshot.get('task_id') != expected_task
                or snapshot.get('introduced_classes')
                != original_task_classes[expected_task]):
            raise ValueError('deferred stage snapshot identity mismatch')
        stage_events[expected_task] = snapshot['event_idx']
        trainer = _fresh_trainer(snapshot, args)
        _strict_load_trainer_state(
            trainer, snapshot['trainer_state'], snapshot['protocol']
        )
        loader = dataset.get_test_loader(original_task_classes[expected_task])
        diagonals[expected_task], _ = _task_readouts(
            trainer, loader, original_task_classes[expected_task]
        )

    final_trainer = _fresh_trainer(final_payload, args)
    _strict_load_trainer_state(
        final_trainer, final_payload['trainer_state'], final_payload['protocol']
    )
    top = getattr(final_trainer, 'top_model', None)
    if top is not None and bool(getattr(top, '_adaptive_enabled', False)):
        active_classes = {
            int(class_id) for class_id in top._adaptive_class_order.tolist()
        }
    else:
        active_classes = {
            class_id for classes in original_task_classes.values()
            for class_id in classes
        }
    final_task_classes = {
        task_id: [class_id for class_id in original_task_classes[task_id]
                  if class_id in active_classes]
        for task_id in ordered_tasks
    }
    final_task_classes = {
        task_id: classes for task_id, classes in final_task_classes.items()
        if classes
    }
    if not final_task_classes:
        raise ValueError('adaptive final checkpoint retained no task classes')
    final_row, task_il = {}, {}
    for task_id, classes in final_task_classes.items():
        loader = dataset.get_test_loader(classes)
        final_row[task_id], task_il[task_id] = _task_readouts(
            final_trainer, loader, classes
        )

    tracker = MetricsTracker()
    for task_id in ordered_tasks[:-1]:
        tracker.record_task_accuracies(
            f'event_{stage_events[task_id]}_CIL',
            {f'task_{task_id}': diagonals[task_id]},
            diagonals[task_id],
        )
        tracker.task_acc_matrix[-1]['deferred_diagonal'] = {
            f'task_{task_id}': diagonals[task_id]
        }
    final_task = ordered_tasks[-1]
    tracker.record_task_accuracies(
        f'event_{stage_events[final_task]}_CIL',
        {f'task_{task_id}': accuracy
         for task_id, accuracy in final_row.items()},
        round(sum(final_row.values()) / len(final_row), 4),
        per_task_taskil={
            f'task_{task_id}': accuracy
            for task_id, accuracy in task_il.items()
        },
    )
    tracker.task_acc_matrix[-1]['deferred_diagonal'] = {
        f'task_{task_id}': diagonals[task_id]
        for task_id in final_task_classes
    }
    tracker.task_acc_matrix[-1]['deferred_final'] = True
    tracker.task_acc_matrix[-1]['deferred_final_task'] = f'task_{final_task}'
    return tracker.to_dict()


_FORMAL_FREEZE = 'FORMAL_STATE_FROZEN.json'
_FORMAL_PENDING = 'FORMAL_EVALUATION_PENDING.json'
_FORMAL_CONSUMING = 'FORMAL_EVALUATION_CONSUMING.json'
_FORMAL_COMPLETE = 'FORMAL_EVALUATION_COMPLETE.json'
_FORMAL_SEALED = 'FORMAL_EVALUATION_SEALED.json'
_FORMAL_PUBLISHING = 'FORMAL_EVALUATION_PUBLISHING.json'
_FORMAL_PUBLISHED = 'FORMAL_EVALUATION_PUBLISHED.json'


def _formal_file_record_from_read(path, output_dir, content, details):
    path = Path(os.path.abspath(os.fspath(path)))
    root = Path(os.path.abspath(os.fspath(output_dir)))
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError('formal evaluation artifact escaped output_dir') from error
    return {
        'path': relative,
        'sha256': _digest(content),
        'device': int(details.st_dev),
        'inode': int(details.st_ino),
        'size': int(details.st_size),
        'mtime_ns': int(details.st_mtime_ns),
        'mode': stat.S_IMODE(details.st_mode),
    }


def _formal_file_record(path, output_dir):
    content, details = _read_file(path)
    return _formal_file_record_from_read(
        path, output_dir, content, details
    )


def _formal_pinned_torch_load(path, expected, output_dir, kind):
    content, details = _read_file(path)
    actual = _formal_file_record_from_read(
        path, output_dir, content, details
    )
    if actual != expected:
        raise ValueError(f'formal frozen {kind} inode/hash mismatch')
    return _restricted_torch_load(content)


def _formal_state_hash(value):
    return trainer_state_sha256(value)


def _formal_task_classes(task_classes, protocol):
    if type(task_classes) is not dict:
        raise ValueError('formal task classes must be a mapping')
    normalized = {
        int(task_id): [int(class_id) for class_id in classes]
        for task_id, classes in task_classes.items()
    }
    ordered = sorted(normalized)
    flat = [class_id for task_id in ordered for class_id in normalized[task_id]]
    if (ordered != list(range(protocol.get('num_tasks', -1)))
            or any(not normalized[task_id] for task_id in ordered)
            or len(flat) != len(set(flat))):
        raise ValueError('formal task class identity is invalid')
    return normalized


def _formal_checkpoint_identity(final_checkpoint, args, task_classes,
                                output_dir):
    content, details = _read_file(final_checkpoint)
    payload = _restricted_torch_load(content)
    from runner import (
        _checkpoint_values_equal,
        _decode_checkpoint_value,
        _validate_resume_checkpoint_payload,
    )
    _validate_resume_checkpoint_payload(payload, args)
    decoded = _decode_checkpoint_value(payload)
    protocol = decoded['protocol']
    normalized = _formal_task_classes(task_classes, protocol)
    final_task = max(normalized)
    if (protocol.get('formal_deferred_evaluation') is not True
            or decoded['task_id'] != final_task
            or decoded['new_classes'] != normalized[final_task]
            or decoded['seen_task_classes'] != normalized
            or decoded['forgotten_classes']
            or decoded['bic_state'] is not None
            or decoded['bic_history']):
        raise ValueError('unsupported or incomplete formal final checkpoint')
    tracker = MetricsTracker()
    tracker.load_dict(decoded['tracker_state'])
    if not _checkpoint_values_equal(tracker.to_dict(), decoded['tracker_state']):
        raise ValueError('formal final tracker strict reload mismatch')
    return payload, decoded, {
        'file': _formal_file_record_from_read(
            final_checkpoint, output_dir, content, details
        ),
        'trainer_state_sha256': _formal_state_hash(payload['trainer_state']),
        'method_state_sha256': _formal_state_hash(payload['cl_state']),
        'tracker_state_sha256': _formal_state_hash(payload['tracker_state']),
        'rng_state_sha256': _formal_state_hash(payload['rng_state']),
    }


def _formal_identity(args, snapshot_paths, final_checkpoint, task_classes,
                     output_dir):
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    if Path(os.path.abspath(os.fspath(args.output_dir))) != output_dir:
        raise ValueError('formal output_dir identity mismatch')
    final_payload, decoded, final = _formal_checkpoint_identity(
        final_checkpoint, args, task_classes, output_dir
    )
    normalized = _formal_task_classes(task_classes, decoded['protocol'])
    paths = [Path(path) for path in snapshot_paths]
    manifest = _snapshot_manifest(output_dir, protocol_kind='formal')
    if (len(paths) != len(normalized) or len(manifest) != len(paths)
            or [record['task_id'] for record in manifest] != sorted(normalized)):
        raise ValueError('formal stage snapshot sequence is incomplete')
    snapshots = []
    last_payload = None
    for task_id, path, expected in zip(sorted(normalized), paths, manifest):
        _validated, record = _snapshot_record(
            path, output_dir, protocol_kind='formal'
        )
        if (record != expected or record['task_id'] != task_id
                or record['introduced_classes'] != normalized[task_id]
                or record['path']
                != Path(path).relative_to(output_dir).as_posix()):
            raise ValueError('formal stage snapshot identity mismatch')
        content, details = _read_file(path)
        if _digest(content) != record['sha256']:
            raise RuntimeError('formal snapshot changed after validation')
        pinned_payload = _restricted_torch_load(content)
        source_path = output_dir / record['source_identity']['path']
        source_content, source_details = _read_file(source_path)
        if _digest(source_content) != record['source_identity']['sha256']:
            raise RuntimeError('formal source checkpoint changed after validation')
        snapshots.append({
            'record': record,
            'file': _formal_file_record_from_read(
                path, output_dir, content, details
            ),
            'source_file': _formal_file_record_from_read(
                source_path, output_dir, source_content, source_details
            ),
        })
        last_payload = pinned_payload
    protocol_sha256 = _digest(
        _strict_json(decoded['protocol']).encode('utf-8')
    )
    source_provenance = _validated_formal_source_provenance(
        decoded['protocol'].get('source_provenance')
    )
    exact_final = last_payload is not None and all(_formal_state_equal(
        final_payload[key], last_payload[key]
    ) for key in ('trainer_state', 'tracker_state', 'rng_state')) and \
        _formal_state_equal(
            _formal_snapshot_method_state(
                decoded['protocol'], final_payload['cl_state']
            ),
            last_payload['cl_state'],
        )
    if (any(item['record']['protocol_sha256'] != protocol_sha256
            for item in snapshots)
            or last_payload is None
            or any(item['record'].get('source_provenance') != source_provenance
                   for item in snapshots)
            or not (exact_final or _formal_internal_install_matches(
                final_payload, last_payload
            ))):
        raise ValueError('formal stage/final state or protocol mismatch')
    return {
        'source_commit': _source_commit(Path(__file__).resolve().parent),
        'source_provenance': copy.deepcopy(source_provenance),
        'protocol': copy.deepcopy(decoded['protocol']),
        'protocol_sha256': protocol_sha256,
        'task_classes': {
            str(task_id): classes for task_id, classes in sorted(normalized.items())
        },
        'snapshots': snapshots,
        'final_checkpoint': final,
    }


def _formal_state_equal(left, right):
    from runner import _checkpoint_values_equal
    return _checkpoint_values_equal(left, right)


def _formal_internal_install_matches(final_payload, stage_payload):
    protocol = final_payload['protocol']
    if (protocol.get('cl_method') != 'proto_evolve'
            or not protocol.get('head_consolidation_enabled')
            or protocol.get('head_consolidation_mode')
            != 'adaptive_dual_branch'):
        return False
    from runner import _decode_checkpoint_value
    final_method = _decode_checkpoint_value(final_payload['cl_state'])
    stage_method = _decode_checkpoint_value(stage_payload['cl_state'])
    if type(final_method) is not dict or type(stage_method) is not dict:
        return False
    changing = {
        'head_consolidation_history', 'head_validation_sha256',
        'adaptive_audit_bundle', 'adaptive_top_version',
        'adaptive_class_order', 'adaptive_gate',
    }
    bundle = final_method.get('adaptive_audit_bundle')
    history = final_method.get('head_consolidation_history')
    result = bundle.get('result') if type(bundle) is dict else None
    installed = bundle.get('installed_state') if type(bundle) is dict else None
    if (type(history) is not list or len(history) != 1
            or history[0] != result
            or type(result) is not dict
            or result.get('task_id') != protocol['num_tasks'] - 1
            or result.get('task_boundary')
            != f"event_{protocol['num_tasks'] - 1}_CIL"
            or final_method.get('head_validation_sha256')
            != result.get('validation_manifest', {}).get('sha256')
            or not _formal_state_equal(
                installed, final_payload['trainer_state']['top_model']
            )
            or not _formal_state_equal(
                final_payload['trainer_state']['bottoms'],
                stage_payload['trainer_state']['bottoms'],
            )
            or not _formal_state_equal(
                final_payload['tracker_state'], stage_payload['tracker_state']
            )
            or not _formal_state_equal(
                {key: value for key, value in final_method.items()
                 if key not in changing},
                {key: value for key, value in stage_method.items()
                 if key not in changing},
            )):
        return False
    return True


def _revalidate_formal_frozen_inputs(
        identity, args, snapshot_paths, task_classes, output_dir):
    from runner import _checkpoint_protocol
    current_source = _validated_formal_source_provenance(
        identity.get('source_provenance')
    )
    if (identity.get('source_commit') != current_source['source_commit']
            or identity.get('source_commit')
            != _source_commit(Path(__file__).resolve().parent)
            or identity.get('protocol') != _checkpoint_protocol(args)):
        raise ValueError('formal frozen source/protocol identity mismatch')
    normalized = _formal_task_classes(task_classes, identity['protocol'])
    if identity.get('task_classes') != {
            str(task_id): classes for task_id, classes in sorted(normalized.items())
    }:
        raise ValueError('formal frozen task class identity mismatch')
    paths = [Path(path) for path in snapshot_paths]
    frozen = identity.get('snapshots')
    manifest = _snapshot_manifest(Path(output_dir), protocol_kind='formal')
    if (type(frozen) is not list or len(paths) != len(frozen)
            or len(manifest) != len(frozen)):
        raise ValueError('formal frozen snapshot sequence is incomplete')
    for path, expected, actual in zip(paths, frozen, manifest):
        _payload, record = _snapshot_record(
            path, Path(output_dir), protocol_kind='formal'
        )
        if (record != actual or record != expected.get('record')
                or _formal_file_record(path, output_dir) != expected.get('file')):
            raise ValueError('formal frozen snapshot inode/hash mismatch')
        _formal_pinned_torch_load(
            path, expected['file'], output_dir, 'snapshot'
        )
        source_path = Path(output_dir) / record['source_identity']['path']
        _formal_pinned_torch_load(
            source_path, expected['source_file'], output_dir,
            'source checkpoint'
        )


def _formal_marker(path, status, keys):
    try:
        value = _safe_json(path)
    except OSError as error:
        raise ValueError(f'unsafe formal {status} marker') from error
    if (set(value) != set(keys) or value.get('schema_version') != 1
            or value.get('status') != status):
        raise ValueError(f'formal {status} marker schema is invalid')
    return value


def _formal_transaction_sha256(identity, freeze):
    return _digest(_strict_json({
        'identity': identity, 'freeze': freeze,
    }).encode('utf-8'))


def _validate_formal_transaction(marker, identity, freeze, status):
    expected = _formal_transaction_sha256(identity, freeze)
    if (marker['identity'] != identity or marker['freeze'] != freeze
            or marker['transaction_sha256'] != expected):
        raise ValueError(f'formal {status} transaction identity mismatch')
    return expected


def _formal_bic_enabled(identity):
    return identity.get('protocol', {}).get('bic_enabled') is True


def _formal_complete_keys(identity):
    keys = {
        'schema_version', 'status', 'transaction_sha256', 'identity',
        'freeze', 'cache_identity', 'evaluation', 'evaluation_sha256',
    }
    if _formal_bic_enabled(identity):
        keys.update({'calibration_cache_identity', 'bic', 'bic_sha256'})
    return keys


def _validate_formal_bic_bundle(value, identity, cache_identity=None):
    protocol = identity['protocol']
    if (protocol.get('data') != 'cifar100'
            or protocol.get('num_tasks') != 10
            or protocol.get('bic_fit_mode') != 'joint_each_stage'
            or type(value) is not dict
            or set(value) != {
                'state', 'history', 'calibration_audit',
                'validation_manifest', 'selection_audit', 'fit_corpus',
            }):
        raise ValueError('formal BiC reconstruction schema is invalid')
    _validate_formal_bic_fit_corpus(
        value['fit_corpus'], identity, cache_identity
    )
    from three_dataset_formal_audit import (
        _EvidenceError, _validate_formal_bic_producer_bundle,
    )
    try:
        _validate_formal_bic_producer_bundle(
            history=value['history'], state=value['state'],
            calibration_audit=value['calibration_audit'],
            task_classes={
                int(task_id): list(classes)
                for task_id, classes in identity['task_classes'].items()
            },
            options={
                key: protocol[key] for key in (
                    'cl_method', 'head_consolidation_enabled',
                    'head_consolidation_mode',
                    'bic_fit_mode', 'bic_lr', 'bic_steps', 'bic_per_class',
                    'lambda_validation_enabled',
                    'lambda_validation_per_class',
                    'lambda_validation_split_seed',
                )
            },
            validation_manifest=value['validation_manifest'],
            selection_audit=value['selection_audit'],
            fit_corpus=value['fit_corpus'],
        )
    except (_EvidenceError, KeyError, TypeError, ValueError) as error:
        raise ValueError('formal BiC producer evidence is invalid') from error
    return value


def _formal_result(complete):
    if not _formal_bic_enabled(complete['identity']):
        return complete['evaluation']
    bic = complete['bic']
    return {
        'tracker_state': complete['evaluation'],
        'bic_history': copy.deepcopy(bic['history']),
        'bic_state': copy.deepcopy(bic['state']),
        'calibration_audit': copy.deepcopy(bic['calibration_audit']),
        'bic_fit_corpus': copy.deepcopy(bic['fit_corpus']),
    }


def _load_formal_complete(output_dir, identity, freeze, pending, consuming):
    complete = _formal_marker(
        Path(output_dir) / _FORMAL_COMPLETE, 'complete',
        _formal_complete_keys(identity),
    )
    transaction = _validate_formal_transaction(
        pending, identity, freeze, 'pending'
    )
    _validate_formal_transaction(consuming, identity, freeze, 'consuming')
    evaluation_sha256 = _digest(
        _strict_json(complete['evaluation']).encode('utf-8')
    )
    if (complete['transaction_sha256'] != transaction
            or complete['identity'] != identity
            or complete['freeze'] != freeze
            or complete['evaluation_sha256'] != evaluation_sha256
            or type(complete['cache_identity']) is not dict
            or complete['cache_identity'] != consuming['cache_identity']):
        raise ValueError('formal complete artifact identity mismatch')
    if _formal_bic_enabled(identity):
        bic_sha256 = _digest(
            _strict_json(complete['bic']).encode('utf-8')
        )
        if (complete['bic_sha256'] != bic_sha256
                or type(complete['calibration_cache_identity']) is not dict
                or complete['cache_identity'].get('calibration')
                != complete['calibration_cache_identity']):
            raise ValueError('formal complete BiC identity mismatch')
        _validate_formal_bic_bundle(
            complete['bic'], identity,
            complete['calibration_cache_identity']['cache'],
        )
    tracker = MetricsTracker()
    tracker.load_dict(complete['evaluation'])
    if (tracker.to_dict() != complete['evaluation']):
        raise ValueError('formal complete tracker strict reload mismatch')
    return complete


def _formal_seal_payload(
        output_dir, identity, freeze, pending, consuming, complete):
    output_dir = Path(output_dir)
    return {
        'schema_version': 1,
        'status': 'sealed',
        'transaction_sha256': complete['transaction_sha256'],
        'identity_sha256': _digest(
            _strict_json(identity).encode('utf-8')
        ),
        'protocol_sha256': identity['protocol_sha256'],
        'source_provenance': copy.deepcopy(identity['source_provenance']),
        'freeze': freeze,
        'cache_identity': copy.deepcopy(consuming['cache_identity']),
        'pending': _formal_file_record(
            output_dir / _FORMAL_PENDING, output_dir
        ),
        'consuming': _formal_file_record(
            output_dir / _FORMAL_CONSUMING, output_dir
        ),
        'complete': _formal_file_record(
            output_dir / _FORMAL_COMPLETE, output_dir
        ),
    }


def _load_formal_seal(
        output_dir, identity, freeze, pending, consuming, complete):
    output_dir = Path(output_dir)
    seal = _formal_marker(output_dir / _FORMAL_SEALED, 'sealed', {
        'schema_version', 'status', 'transaction_sha256',
        'identity_sha256', 'protocol_sha256', 'freeze', 'cache_identity',
        'source_provenance', 'pending', 'consuming', 'complete',
    })
    expected = _formal_seal_payload(
        output_dir, identity, freeze, pending, consuming, complete
    )
    if seal != expected:
        raise ValueError('formal complete seal identity mismatch')
    return seal


def _validated_complete_artifact(path):
    candidate = _safe_json(path)
    identity = candidate.get('identity') if type(candidate) is dict else {}
    complete = _formal_marker(
        path, 'complete', _formal_complete_keys(identity)
    )
    if (type(complete['transaction_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', complete['transaction_sha256']) is None
            or type(complete['cache_identity']) is not dict
            or complete['evaluation_sha256'] != _digest(
                _strict_json(complete['evaluation']).encode('utf-8')
            )):
        raise ValueError('formal complete artifact hash is invalid')
    if _formal_bic_enabled(identity):
        if complete['bic_sha256'] != _digest(
                _strict_json(complete['bic']).encode('utf-8')):
            raise ValueError('formal complete BiC hash is invalid')
        _validate_formal_bic_bundle(
            complete['bic'], identity,
            complete['calibration_cache_identity']['cache'],
        )
    return complete


def _load_sealed_complete_artifact(output_dir):
    output_dir = Path(output_dir)
    candidate = _validated_complete_artifact(
        output_dir / _FORMAL_COMPLETE
    )
    identity = candidate['identity']
    freeze = candidate['freeze']
    pending = _formal_marker(output_dir / _FORMAL_PENDING, 'pending', {
        'schema_version', 'status', 'transaction_sha256', 'identity', 'freeze',
    })
    consuming = _formal_marker(output_dir / _FORMAL_CONSUMING, 'consuming', {
        'schema_version', 'status', 'transaction_sha256', 'identity', 'freeze',
        'cache_identity',
    })
    complete = _load_formal_complete(
        output_dir, identity, freeze, pending, consuming
    )
    seal = _load_formal_seal(
        output_dir, identity, freeze, pending, consuming, complete
    )
    return complete, seal


def _formal_existing(path):
    return os.path.lexists(os.fspath(path))


def _validate_formal_published(output_dir, identity, complete, seal):
    output_dir = Path(output_dir)
    seal_record = _formal_file_record(
        output_dir / _FORMAL_SEALED, output_dir
    )
    publishing_path = output_dir / _FORMAL_PUBLISHING
    published_path = output_dir / _FORMAL_PUBLISHED
    if _formal_existing(publishing_path) and not _formal_existing(published_path):
        _formal_marker(publishing_path, 'publishing', {
            'schema_version', 'status', 'transaction_sha256',
            'evaluation_sha256', 'results_sha256', 'seal',
            'source_provenance',
        })
        raise RuntimeError('formal publication is incomplete; failing closed')
    if not _formal_existing(published_path):
        return
    publishing = _formal_marker(publishing_path, 'publishing', {
        'schema_version', 'status', 'transaction_sha256',
        'evaluation_sha256', 'results_sha256', 'seal',
        'source_provenance',
    })
    published = _formal_marker(published_path, 'published', {
        'schema_version', 'status', 'transaction_sha256',
        'evaluation_sha256', 'results_sha256', 'seal',
        'source_provenance', 'checkpoint', 'results',
    })
    if (publishing['transaction_sha256'] != complete['transaction_sha256']
            or seal['transaction_sha256'] != complete['transaction_sha256']
            or publishing['evaluation_sha256']
            != complete['evaluation_sha256']
            or published['transaction_sha256']
            != complete['transaction_sha256']
            or published['evaluation_sha256']
            != complete['evaluation_sha256']
            or publishing['seal'] != seal_record
            or published['seal'] != seal_record
            or published['results_sha256'] != publishing['results_sha256']
            or publishing['source_provenance']
                != identity['source_provenance']
            or published['source_provenance']
                != identity['source_provenance']):
        raise ValueError('formal publication transaction identity mismatch')
    checkpoint_path = output_dir / published['checkpoint']['path']
    results_path = output_dir / published['results']['path']
    if (published['checkpoint']['path']
            != identity['final_checkpoint']['file']['path']
            or _formal_file_record(checkpoint_path, output_dir)
            != published['checkpoint']
            or _formal_file_record(results_path, output_dir)
            != published['results']):
        raise ValueError('formal published artifact inode/hash mismatch')
    payload = _safe_torch_load(checkpoint_path)
    results = _safe_json(results_path)
    if identity['protocol'].get('cl_method') == 'er_ace':
        _formal_snapshot_method_state(identity['protocol'], payload.get('cl_state'))
    bic = complete.get('bic')
    if (payload.get('protocol') != identity['protocol']
            or _formal_state_hash(payload.get('trainer_state'))
            != identity['final_checkpoint']['trainer_state_sha256']
            or _formal_state_hash(payload.get('cl_state'))
            != identity['final_checkpoint']['method_state_sha256']
            or _formal_state_hash(payload.get('rng_state'))
            != identity['final_checkpoint']['rng_state_sha256']
            or _formal_state_hash(payload.get('tracker_state'))
            != _formal_state_hash(complete['evaluation'])
            or (_formal_bic_enabled(identity) and (
                payload.get('bic_state') != bic['state']
                or payload.get('bic_history') != bic['history']
                or results.get('bic_history') != bic['history']
                or results.get('bic_final') != bic['history'][-1]
                or results.get('calibration_audit')
                != bic['calibration_audit']
                or (bic['selection_audit'] is not None
                    and results.get('selection_audit')
                    != bic['selection_audit'])))
            or _digest(_strict_json(results).encode('utf-8'))
            != publishing['results_sha256']
            or results.get('source_provenance')
                != identity['source_provenance']
            or (_formal_bic_enabled(identity)
                and results.get('bic_fit_corpus') != bic['fit_corpus'])):
        raise ValueError('formal published checkpoint state mismatch')


def prepare_formal_deferred_evaluation(
        *, args, snapshot_paths, final_checkpoint, task_classes, output_dir):
    """Freeze identities and durably reserve the sole formal-test transaction."""
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    freeze_path = output_dir / _FORMAL_FREEZE
    complete_path = output_dir / _FORMAL_COMPLETE
    seal_path = output_dir / _FORMAL_SEALED
    publishing_path = output_dir / _FORMAL_PUBLISHING
    published_path = output_dir / _FORMAL_PUBLISHED
    if (_formal_existing(seal_path)
            and not _formal_existing(complete_path)):
        try:
            _formal_marker(seal_path, 'sealed', {
                'schema_version', 'status', 'transaction_sha256',
                'identity_sha256', 'protocol_sha256', 'freeze',
                'cache_identity', 'source_provenance',
                'pending', 'consuming', 'complete',
            })
        except (ValueError, OSError) as error:
            raise ValueError(
                'orphan formal evaluation seal conflicts with preparation'
            ) from error
        raise ValueError(
            'orphan formal evaluation seal conflicts with preparation'
        )
    if _formal_existing(publishing_path) and not _formal_existing(published_path):
        _formal_marker(publishing_path, 'publishing', {
            'schema_version', 'status', 'transaction_sha256',
            'evaluation_sha256', 'results_sha256', 'seal',
            'source_provenance',
        })
        raise RuntimeError('formal publication is incomplete; failing closed')
    if _formal_existing(published_path):
        freeze = _formal_marker(freeze_path, 'frozen', {
            'schema_version', 'status', 'identity',
        })
        identity = freeze['identity']
        expected_final = identity['final_checkpoint']['file']['path']
        if (Path(os.path.abspath(os.fspath(final_checkpoint))).relative_to(
                output_dir).as_posix() != expected_final):
            raise ValueError('formal published checkpoint path mismatch')
        _revalidate_formal_frozen_inputs(
            identity, args, snapshot_paths, task_classes, output_dir
        )
    else:
        identity = _formal_identity(
            args, snapshot_paths, final_checkpoint, task_classes, output_dir
        )
        if _formal_existing(freeze_path):
            freeze = _formal_marker(freeze_path, 'frozen', {
                'schema_version', 'status', 'identity',
            })
            if freeze['identity'] != identity:
                raise ValueError('formal freeze identity mismatch')
        else:
            freeze = {
                'schema_version': 1, 'status': 'frozen', 'identity': identity,
            }
            atomic_write_new_json(freeze_path, freeze)
            freeze = _formal_marker(freeze_path, 'frozen', {
                'schema_version', 'status', 'identity',
            })
    freeze_record = _formal_file_record(freeze_path, output_dir)
    pending_path = output_dir / _FORMAL_PENDING
    consuming_path = output_dir / _FORMAL_CONSUMING
    if _formal_existing(complete_path):
        pending = _formal_marker(pending_path, 'pending', {
            'schema_version', 'status', 'transaction_sha256', 'identity', 'freeze',
        })
        consuming = _formal_marker(consuming_path, 'consuming', {
            'schema_version', 'status', 'transaction_sha256', 'identity', 'freeze',
            'cache_identity',
        })
        complete = _load_formal_complete(
            output_dir, identity, freeze_record, pending, consuming
        )
        seal = _load_formal_seal(
            output_dir, identity, freeze_record, pending, consuming, complete
        )
        _validate_formal_published(output_dir, identity, complete, seal)
        return {
            'status': ('published' if _formal_existing(published_path)
                       else 'complete'),
            'result': _formal_result(complete),
        }
    for path, status, keys in (
            (consuming_path, 'consuming', {
                'schema_version', 'status', 'transaction_sha256', 'identity',
                'freeze', 'cache_identity',
            }),
            (pending_path, 'pending', {
                'schema_version', 'status', 'transaction_sha256', 'identity',
                'freeze',
            })):
        if _formal_existing(path):
            marker = _formal_marker(path, status, keys)
            _validate_formal_transaction(
                marker, identity, freeze_record, status
            )
            raise RuntimeError(
                f'formal evaluation {status} without complete artifact; '
                'failing closed'
            )
    transaction = _formal_transaction_sha256(identity, freeze_record)
    atomic_write_new_json(pending_path, {
        'schema_version': 1, 'status': 'pending',
        'transaction_sha256': transaction,
        'identity': identity, 'freeze': freeze_record,
    })
    return {'status': 'pending', 'transaction_sha256': transaction}


def _formal_cache_identity(cached_test_batches):
    if type(cached_test_batches) is not tuple or not cached_test_batches:
        raise ValueError('formal cached batches must be a non-empty tuple')
    batches = []
    sample_count = 0
    for batch in cached_test_batches:
        if (type(batch) is not tuple or len(batch) != 2
                or not all(isinstance(value, torch.Tensor) for value in batch)):
            raise ValueError('formal cache batch schema is invalid')
        batch_x, batch_y = batch
        if (batch_x.device.type != 'cpu' or batch_y.device.type != 'cpu'
                or batch_x.requires_grad or batch_y.requires_grad
                or batch_x.grad_fn is not None or batch_y.grad_fn is not None
                or batch_y.ndim != 1 or batch_x.size(0) != batch_y.size(0)):
            raise ValueError('formal cache must contain detached CPU tensors')
        sample_count += int(batch_y.numel())
        batches.append({
            'input_sha256': _tensor_sha256(batch_x),
            'label_sha256': _tensor_sha256(batch_y),
            'input_shape': list(batch_x.shape),
            'label_shape': list(batch_y.shape),
            'input_dtype': str(batch_x.dtype),
            'label_dtype': str(batch_y.dtype),
        })
    return {
        'batch_count': len(batches), 'sample_count': sample_count,
        'batches': batches,
    }


def _formal_bic_fit_corpus(
        identity, cached_batches, calibration_audit,
        validation_manifest, selection_audit):
    protocol = identity.get('protocol', {})
    internal = (
        protocol.get('cl_method') == 'proto_evolve'
        and bool(protocol.get('head_consolidation_enabled'))
        and protocol.get('head_consolidation_mode') != 'full_classifier'
    )
    split = 'validation' if internal else 'calibration'
    phase = ('final_validation_pre_install' if internal else
             'final_bic_calibration_post_freeze')
    selection_key = f'{split}_manifest_sha256'
    source = validation_manifest if internal else calibration_audit
    manifest_sha256 = source.get('sha256' if internal else 'manifest_sha256') \
        if type(source) is dict else None
    if (type(selection_audit) is not dict
            or selection_audit.get(selection_key) != manifest_sha256):
        raise ValueError('formal BiC fit corpus selection manifest mismatch')
    corpus = {
        'schema_version': 1,
        'split': split,
        'phase': phase,
        'cache_identity': _formal_cache_identity(cached_batches),
        'manifest_sha256': manifest_sha256,
        'selection_manifest_key': selection_key,
    }
    return _validate_formal_bic_fit_corpus(corpus, identity)


def _validate_formal_bic_fit_corpus(value, identity, cache_identity=None):
    protocol = identity.get('protocol', {})
    internal = (
        protocol.get('cl_method') == 'proto_evolve'
        and bool(protocol.get('head_consolidation_enabled'))
        and protocol.get('head_consolidation_mode') != 'full_classifier'
    )
    expected = (
        ('validation', 'final_validation_pre_install',
         'validation_manifest_sha256')
        if internal else
        ('calibration', 'final_bic_calibration_post_freeze',
         'calibration_manifest_sha256')
    )
    if (type(value) is not dict or set(value) != {
            'schema_version', 'split', 'phase', 'cache_identity',
            'manifest_sha256', 'selection_manifest_key'}
            or value.get('schema_version') != 1
            or (value.get('split'), value.get('phase'),
                value.get('selection_manifest_key')) != expected
            or type(value.get('manifest_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', value['manifest_sha256']) is None
            or (cache_identity is not None
                and value.get('cache_identity') != cache_identity)):
        raise ValueError('formal BiC fit corpus identity is invalid')
    _formal_cache_identity_from_record(value.get('cache_identity'))
    return value


def _formal_cache_identity_from_record(value):
    if (type(value) is not dict
            or set(value) != {'batch_count', 'sample_count', 'batches'}
            or type(value.get('batch_count')) is not int
            or type(value.get('sample_count')) is not int
            or type(value.get('batches')) is not list
            or value['batch_count'] != len(value['batches'])
            or value['batch_count'] <= 0 or value['sample_count'] <= 0):
        raise ValueError('formal cache identity record is invalid')
    return value


def _fresh_formal_state(payload, args):
    isolated_args = copy.deepcopy(args)
    isolated_args.device = 'cpu'
    trainer = _fresh_trainer(payload, isolated_args)
    _strict_load_trainer_state(
        trainer, payload['trainer_state'], payload['protocol']
    )
    from cl_methods import get_cl_method
    from runner import _checkpoint_values_equal, _decode_checkpoint_value
    if payload['protocol'].get('cl_method') == 'er_ace':
        _formal_snapshot_method_state(payload['protocol'], payload['cl_state'])
    method_state = _decode_checkpoint_value(payload['cl_state'])
    method = get_cl_method(payload['protocol']['cl_method'], trainer, isolated_args)
    formal_method_state = False
    if payload['protocol']['cl_method'] == 'fedprotip_vfl':
        from cl_methods.fedprotip_vfl import FORMAL_EVALUATION_STATE_KEYS
        formal_method_state = set(method_state) == FORMAL_EVALUATION_STATE_KEYS
    if formal_method_state:
        method.load_formal_evaluation_state(copy.deepcopy(method_state))
    elif hasattr(method, 'load_state'):
        method.load_state(copy.deepcopy(method_state))
    reloaded = (
        method.get_formal_evaluation_state()
        if formal_method_state
        else method.get_state()
    ) if hasattr(method, 'get_state') else method_state
    if not _checkpoint_values_equal(reloaded, method_state):
        raise RuntimeError('formal method strict reload changed state')
    tracker = MetricsTracker()
    tracker.load_dict(copy.deepcopy(payload['tracker_state']))
    if not _checkpoint_values_equal(tracker.to_dict(), payload['tracker_state']):
        raise RuntimeError('formal tracker strict reload changed state')
    from runner import _validate_rng_checkpoint_state
    _validate_rng_checkpoint_state(payload['rng_state'])
    return trainer, method, tracker


def evaluate_formal_deferred_trajectory(
        *, args, snapshot_paths, final_checkpoint, task_classes,
        cached_test_batches, output_dir, cached_calibration_batches=None,
        calibration_audit=None, validation_manifest=None,
        selection_audit=None, recompute_payloads=None):
    """Evaluate every formal stage from one immutable all-class CPU cache."""
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    recomputing = recompute_payloads is not None
    if recomputing:
        if (type(recompute_payloads) is not dict
                or set(recompute_payloads) != {'snapshots', 'final'}
                or type(recompute_payloads['snapshots']) is not list
                or len(recompute_payloads['snapshots']) != len(task_classes)
                or type(recompute_payloads['final']) is not dict):
            raise ValueError('formal recomputation payload schema is invalid')
        final_probe = recompute_payloads['final']
        identity = {
            'protocol': copy.deepcopy(final_probe['protocol']),
            'task_classes': {
                str(task_id): list(classes)
                for task_id, classes in sorted(task_classes.items())
            },
        }
        consuming_path = None
    else:
        identity = _formal_identity(
            args, snapshot_paths, final_checkpoint, task_classes, output_dir
        )
        freeze = _formal_marker(output_dir / _FORMAL_FREEZE, 'frozen', {
            'schema_version', 'status', 'identity',
        })
        if freeze['identity'] != identity:
            raise ValueError('formal freeze identity mismatch before evaluation')
        freeze_record = _formal_file_record(
            output_dir / _FORMAL_FREEZE, output_dir
        )
        pending = _formal_marker(output_dir / _FORMAL_PENDING, 'pending', {
            'schema_version', 'status', 'transaction_sha256', 'identity',
            'freeze',
        })
        transaction = _validate_formal_transaction(
            pending, identity, freeze_record, 'pending'
        )
        consuming_path = output_dir / _FORMAL_CONSUMING
        if _formal_existing(consuming_path) or _formal_existing(
                output_dir / _FORMAL_COMPLETE):
            raise RuntimeError(
                'formal evaluation transaction was already consumed'
            )
    bic_enabled = _formal_bic_enabled(identity)
    test_cache_identity = _formal_cache_identity(cached_test_batches)
    calibration_cache_identity = None
    fit_corpus = None
    if bic_enabled:
        if cached_calibration_batches is None:
            raise ValueError('formal BiC calibration cache is missing')
        internal = (
            identity['protocol'].get('cl_method') == 'proto_evolve'
            and bool(identity['protocol'].get('head_consolidation_enabled'))
            and identity['protocol'].get('head_consolidation_mode')
            != 'full_classifier'
        )
        calibration_cache_identity = {
            'split': 'validation' if internal else 'calibration',
            'phase': ('final_validation_pre_install' if internal else
                      'final_bic_calibration_post_freeze'),
            'cache': _formal_cache_identity(cached_calibration_batches),
        }
        if (type(calibration_audit) is not dict
                or calibration_audit.get('passed') is not True
                or calibration_audit.get('test_used_for_fit') is not False):
            raise ValueError('formal BiC calibration audit is invalid')
        fit_corpus = _formal_bic_fit_corpus(
            identity, cached_calibration_batches, calibration_audit,
            validation_manifest, selection_audit,
        )
    cache_identity = (
        {'test': test_cache_identity, 'calibration': calibration_cache_identity}
        if bic_enabled else test_cache_identity
    )
    if not recomputing:
        atomic_write_new_json(consuming_path, {
            'schema_version': 1, 'status': 'consuming',
            'transaction_sha256': transaction, 'identity': identity,
            'freeze': freeze_record, 'cache_identity': cache_identity,
        })
    normalized = {
        int(task_id): list(classes) for task_id, classes in task_classes.items()
    }
    final_task_id = max(normalized)
    diagonals = {}
    events = {}
    bic_history = []
    bic_calibrator = None
    from runner import _capture_rng_state, _checkpoint_values_equal, _restore_rng_state
    from runner import (
        _evaluate_cil_readouts_cached,
        _fit_and_evaluate_final_bic_cached,
    )
    if bic_enabled:
        from bic_calibration import TaskAffineCalibrator
        bic_calibrator = TaskAffineCalibrator()
    live_rng = _capture_rng_state()
    try:
        for position, (task_id, path) in enumerate(zip(
                sorted(normalized), snapshot_paths)):
            if recomputing:
                payload = recompute_payloads['snapshots'][position]
                if type(payload) is bytes:
                    payload = _restricted_torch_load(payload)
                if type(payload) is not dict:
                    raise ValueError(
                        'formal recomputation snapshot payload is invalid'
                    )
            else:
                frozen = identity['snapshots'][position]
                _validated, record = _snapshot_record(
                    path, output_dir, protocol_kind='formal'
                )
                if record != frozen['record']:
                    raise ValueError('formal frozen snapshot record mismatch')
                payload = _formal_pinned_torch_load(
                    path, frozen['file'], output_dir, 'snapshot'
                )
                _formal_pinned_torch_load(
                    output_dir / record['source_identity']['path'],
                    frozen['source_file'], output_dir, 'source checkpoint'
                )
            trainer, method, _tracker = _fresh_formal_state(payload, args)
            stage_classes = {
                key: normalized[key] for key in sorted(normalized)
                if key <= task_id
            }
            reuse_union_logits = (
                bic_enabled and task_id != final_task_id
                and not hasattr(method, 'evaluate_class_il_readouts_cached')
            )
            readouts = _evaluate_cil_readouts_cached(
                method, trainer, cached_test_batches, stage_classes, 'cpu',
                **({'collect_union_logits': True} if reuse_union_logits else {}),
            )
            diagonals[task_id] = readouts['primary'][f'task_{task_id}']
            events[task_id] = payload['event_idx']
            if bic_enabled and task_id != final_task_id:
                bic_history.append(_fit_and_evaluate_final_bic_cached(
                    trainer, cached_calibration_batches, cached_test_batches,
                    bic_calibrator, args, f"event_{payload['event_idx']}_CIL",
                    stage_classes, calibration_audit, fit_corpus,
                    **({'precomputed_test': readouts.pop('formal_union_logits')}
                       if reuse_union_logits else {}),
                ))
            del trainer, method, _tracker, payload, readouts
        final_payload = (
            recompute_payloads['final'] if recomputing else
            _formal_pinned_torch_load(
                final_checkpoint, identity['final_checkpoint']['file'],
                output_dir, 'final checkpoint'
            )
        )
        final_trainer, final_method, tracker = _fresh_formal_state(
            final_payload, args
        )
        reuse_union_logits = (
            bic_enabled and not hasattr(
                final_method, 'evaluate_class_il_readouts_cached'
            )
        )
        final_readouts = _evaluate_cil_readouts_cached(
            final_method, final_trainer, cached_test_batches, normalized, 'cpu',
            **({'collect_union_logits': True} if reuse_union_logits else {}),
        )
        if bic_enabled:
            bic_history.append(_fit_and_evaluate_final_bic_cached(
                final_trainer, cached_calibration_batches,
                cached_test_batches, bic_calibrator, args,
                f"event_{final_payload['event_idx']}_CIL",
                normalized, calibration_audit, fit_corpus,
                **({'precomputed_test': final_readouts.pop(
                    'formal_union_logits')}
                   if reuse_union_logits else {}),
            ))
    finally:
        _restore_rng_state(live_rng)
    if not _checkpoint_values_equal(_capture_rng_state(), live_rng):
        raise RuntimeError('formal evaluation changed live RNG state')
    if _formal_cache_identity(cached_test_batches) != test_cache_identity:
        raise RuntimeError('formal evaluation mutated the cached test batches')
    if (bic_enabled and _formal_cache_identity(cached_calibration_batches)
            != calibration_cache_identity['cache']):
        raise RuntimeError('formal evaluation mutated the calibration cache')
    tracker.task_acc_matrix = []
    ordered = sorted(normalized)
    for task_id in ordered[:-1]:
        key = f'task_{task_id}'
        tracker.record_task_accuracies(
            f'event_{events[task_id]}_CIL', {key: diagonals[task_id]},
            diagonals[task_id],
        )
        tracker.task_acc_matrix[-1]['deferred_diagonal'] = {
            key: diagonals[task_id]
        }
    final_task = ordered[-1]
    tracker.record_task_accuracies(
        f'event_{events[final_task]}_CIL', final_readouts['primary'],
        final_readouts['overall'],
        per_task_taskil=final_readouts['task_il'],
        companion_readouts=final_readouts['companions'],
    )
    tracker.task_acc_matrix[-1].update({
        'deferred_diagonal': {
            f'task_{task_id}': diagonals[task_id] for task_id in ordered
        },
        'deferred_final': True,
        'deferred_final_task': f'task_{final_task}',
    })
    evaluation = tracker.to_dict()
    from runner import _valid_tracker_checkpoint_state
    if not _valid_tracker_checkpoint_state(evaluation, identity['protocol']):
        raise RuntimeError('formal evaluated tracker schema is invalid')
    evaluation_sha256 = _digest(
        _strict_json(evaluation).encode('utf-8')
    )
    complete_payload = None if recomputing else {
        'schema_version': 1, 'status': 'complete',
        'transaction_sha256': transaction, 'identity': identity,
        'freeze': freeze_record, 'cache_identity': cache_identity,
        'evaluation': evaluation, 'evaluation_sha256': evaluation_sha256,
    }
    if bic_enabled:
        bic = {
            'state': bic_calibrator.state_dict(),
            'history': bic_history,
            'calibration_audit': copy.deepcopy(calibration_audit),
            'validation_manifest': copy.deepcopy(validation_manifest),
            'selection_audit': copy.deepcopy(selection_audit),
            'fit_corpus': copy.deepcopy(fit_corpus),
        }
        _validate_formal_bic_bundle(
            bic, identity, calibration_cache_identity['cache']
        )
    if recomputing:
        if not bic_enabled:
            return evaluation
        return {
            'tracker_state': evaluation,
            'bic_history': copy.deepcopy(bic['history']),
            'bic_state': copy.deepcopy(bic['state']),
            'calibration_audit': copy.deepcopy(bic['calibration_audit']),
            'bic_fit_corpus': copy.deepcopy(bic['fit_corpus']),
        }
    if bic_enabled:
        complete_payload.update({
            'calibration_cache_identity': calibration_cache_identity,
            'bic': bic,
            'bic_sha256': _digest(_strict_json(bic).encode('utf-8')),
        })
    atomic_write_new_json(output_dir / _FORMAL_COMPLETE, complete_payload)
    consuming = _formal_marker(consuming_path, 'consuming', {
        'schema_version', 'status', 'transaction_sha256', 'identity',
        'freeze', 'cache_identity',
    })
    complete = _load_formal_complete(
        output_dir, identity, freeze_record, pending, consuming
    )
    seal_payload = _formal_seal_payload(
        output_dir, identity, freeze_record, pending, consuming, complete
    )
    atomic_write_new_json(output_dir / _FORMAL_SEALED, seal_payload)
    _load_formal_seal(
        output_dir, identity, freeze_record, pending, consuming, complete
    )
    return _formal_result(complete)


def _validate_formal_publication_checkpoint(
        args, final_checkpoint, output_dir, complete):
    identity = complete['identity']
    expected_path = identity['final_checkpoint']['file']['path']
    content, details = _read_file(final_checkpoint)
    checkpoint_record = _formal_file_record_from_read(
        final_checkpoint, output_dir, content, details
    )
    if checkpoint_record['path'] != expected_path:
        raise ValueError('formal published checkpoint path mismatch')
    payload = _restricted_torch_load(content)
    from runner import (
        _capture_rng_state,
        _checkpoint_values_equal,
        _decode_checkpoint_value,
        _restore_rng_state,
        _validate_resume_checkpoint_payload,
    )
    _validate_resume_checkpoint_payload(payload, args)
    decoded = _decode_checkpoint_value(payload)
    task_classes = {
        int(task_id): list(classes)
        for task_id, classes in identity['task_classes'].items()
    }
    final_task = max(task_classes)
    frozen = identity['final_checkpoint']
    bic = complete.get('bic')
    if (decoded['protocol'] != identity['protocol']
            or decoded['task_id'] != final_task
            or decoded['event_idx'] != final_task
            or decoded['step'] != f'event_{final_task}_CIL'
            or decoded['new_classes'] != task_classes[final_task]
            or decoded['seen_task_classes'] != task_classes
            or decoded['forgotten_classes']
            or (_formal_bic_enabled(identity) and (
                decoded['bic_state'] != bic['state']
                or decoded['bic_history'] != bic['history']))
            or (not _formal_bic_enabled(identity) and (
                decoded['bic_state'] is not None or decoded['bic_history']))
            or _formal_state_hash(payload['trainer_state'])
            != frozen['trainer_state_sha256']
            or _formal_state_hash(payload['cl_state'])
            != frozen['method_state_sha256']
            or _formal_state_hash(payload['rng_state'])
            != frozen['rng_state_sha256']
            or _formal_state_hash(payload['tracker_state'])
            != _formal_state_hash(complete['evaluation'])):
        raise ValueError('formal published checkpoint frozen identity mismatch')
    live_rng = _capture_rng_state()
    try:
        _trainer, _method, tracker = _fresh_formal_state(payload, args)
    finally:
        _restore_rng_state(live_rng)
    if (not _checkpoint_values_equal(_capture_rng_state(), live_rng)
            or not _checkpoint_values_equal(
                tracker.to_dict(), complete['evaluation'])):
        raise RuntimeError('formal published checkpoint strict reload mismatch')
    return payload, checkpoint_record


def _revalidate_formal_publication_inputs(
        args, output_dir, complete, seal, final_checkpoint,
        checkpoint_record, results_path, results_record):
    output_dir = Path(output_dir)
    current_complete, current_seal = _load_sealed_complete_artifact(
        output_dir
    )
    if current_complete != complete or current_seal != seal:
        raise ValueError('formal sealed transaction changed before publication')
    freeze_path = output_dir / _FORMAL_FREEZE
    freeze_content, freeze_details = _read_file(freeze_path)
    freeze_record = _formal_file_record_from_read(
        freeze_path, output_dir, freeze_content, freeze_details
    )
    try:
        freeze = json.loads(freeze_content.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('formal frozen identity is invalid') from error
    if (freeze_record != complete['freeze']
            or type(freeze) is not dict
            or set(freeze) != {'schema_version', 'status', 'identity'}
            or freeze.get('schema_version') != 1
            or freeze.get('status') != 'frozen'
            or freeze.get('identity') != complete['identity']):
        raise ValueError('formal frozen inode/identity changed before publication')
    identity = complete['identity']
    snapshot_paths = [
        output_dir / item['file']['path'] for item in identity['snapshots']
    ]
    task_classes = {
        int(task_id): list(classes)
        for task_id, classes in identity['task_classes'].items()
    }
    _revalidate_formal_frozen_inputs(
        identity, args, snapshot_paths, task_classes, output_dir
    )
    _payload, current_checkpoint = _validate_formal_publication_checkpoint(
        args, final_checkpoint, output_dir, complete
    )
    if (current_checkpoint != checkpoint_record
            or _formal_file_record(results_path, output_dir) != results_record):
        raise RuntimeError(
            'formal publication output changed during final revalidation'
        )


def begin_formal_deferred_publication(
        *, output_dir, tracker_state, final_result):
    output_dir = Path(output_dir)
    complete, _seal = _load_sealed_complete_artifact(output_dir)
    bic = complete.get('bic')
    if (complete.get('schema_version') != 1
            or complete.get('status') != 'complete'
            or complete.get('evaluation') != tracker_state
            or any(final_result.get(key) != value
                   for key, value in tracker_state.items())
            or final_result.get('source_provenance')
                != complete['identity']['source_provenance']
            or (_formal_bic_enabled(complete['identity']) and (
                final_result.get('bic_history') != bic['history']
                or final_result.get('bic_final') != bic['history'][-1]
                or final_result.get('calibration_audit')
                != bic['calibration_audit']
                or final_result.get('bic_fit_corpus') != bic['fit_corpus']
                or (bic['selection_audit'] is not None
                    and final_result.get('selection_audit')
                    != bic['selection_audit'])))):
        raise ValueError('formal publication payload mismatches complete artifact')
    publishing_path = output_dir / _FORMAL_PUBLISHING
    published_path = output_dir / _FORMAL_PUBLISHED
    if _formal_existing(published_path):
        raise RuntimeError('formal result is already published')
    if _formal_existing(publishing_path):
        _formal_marker(publishing_path, 'publishing', {
            'schema_version', 'status', 'transaction_sha256',
            'evaluation_sha256', 'results_sha256', 'seal',
            'source_provenance',
        })
        raise RuntimeError('formal publication is incomplete; failing closed')
    results_sha256 = _digest(_strict_json(final_result).encode('utf-8'))
    atomic_write_new_json(publishing_path, {
        'schema_version': 1, 'status': 'publishing',
        'transaction_sha256': complete['transaction_sha256'],
        'evaluation_sha256': complete['evaluation_sha256'],
        'results_sha256': results_sha256,
        'seal': _formal_file_record(
            output_dir / _FORMAL_SEALED, output_dir
        ),
        'source_provenance': copy.deepcopy(
            complete['identity']['source_provenance']
        ),
    })
    return complete


def complete_formal_deferred_publication(
        *, args, output_dir, final_checkpoint, results_path):
    output_dir = Path(output_dir)
    complete, seal = _load_sealed_complete_artifact(output_dir)
    publishing = _formal_marker(
        output_dir / _FORMAL_PUBLISHING, 'publishing', {
            'schema_version', 'status', 'transaction_sha256',
            'evaluation_sha256', 'results_sha256', 'seal',
            'source_provenance',
        }
    )
    _payload, checkpoint_record = _validate_formal_publication_checkpoint(
        args, final_checkpoint, output_dir, complete
    )
    results_record = _formal_file_record(results_path, output_dir)
    results = _safe_json(results_path)
    if (publishing['transaction_sha256'] != complete.get('transaction_sha256')
            or publishing['evaluation_sha256']
            != complete.get('evaluation_sha256')
            or publishing['seal'] != _formal_file_record(
                output_dir / _FORMAL_SEALED, output_dir
            )
            or publishing['results_sha256']
            != _digest(_strict_json(results).encode('utf-8'))
            or publishing['source_provenance']
                != complete['identity']['source_provenance']
            or results.get('source_provenance')
                != complete['identity']['source_provenance']):
        raise ValueError('formal publication output identity mismatch')
    if (_formal_file_record(final_checkpoint, output_dir) != checkpoint_record
            or _formal_file_record(results_path, output_dir) != results_record):
        raise RuntimeError('formal publication output changed before success marker')
    _revalidate_formal_publication_inputs(
        args, output_dir, complete, seal, final_checkpoint,
        checkpoint_record, results_path, results_record
    )
    marker = {
        'schema_version': 1, 'status': 'published',
        'transaction_sha256': complete['transaction_sha256'],
        'evaluation_sha256': complete['evaluation_sha256'],
        'results_sha256': publishing['results_sha256'],
        'seal': publishing['seal'],
        'source_provenance': copy.deepcopy(
            complete['identity']['source_provenance']
        ),
        'checkpoint': checkpoint_record, 'results': results_record,
    }
    atomic_write_new_json(output_dir / _FORMAL_PUBLISHED, marker)
    return marker
