"""Dataset loading, N-party VFL split, task management."""
import json
import os
from pathlib import Path
import stat
import threading
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, Sampler
from torchvision import datasets, transforms
from calibration_split import build_manifest, manifest_indices, write_manifest
from determinism import derive_seed, seed_worker


# Datasets whose samples are FLAT FEATURE VECTORS (tabular / multi-view), split
# by a party->column-range map rather than by an image width axis.
VECTOR_DATASETS = {'mfeat', 'synthvfl', 'tabvfl'}


class DeterministicSampler(Sampler):
    def __init__(self, original_indices, generator, shuffle, record=False):
        self.original_indices = list(original_indices)
        self.generator = generator
        self.shuffle = shuffle
        self.record = record
        self.epoch_orders = []

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.original_indices), generator=self.generator).tolist()
        else:
            order = list(range(len(self.original_indices)))
        if self.record:
            self.epoch_orders.append([self.original_indices[i] for i in order])
        return iter(order)

    def __len__(self):
        return len(self.original_indices)


def make_deterministic_loader(dataset, indices, args, loader_key, shuffle, audit=False):
    """Build a loader whose sampling and worker RNG streams are method-independent."""
    sampler_generator = torch.Generator().manual_seed(
        derive_seed(args.seed, 'data', loader_key, 'sampler')
    )
    worker_generator = torch.Generator().manual_seed(
        derive_seed(args.seed, 'data', loader_key, 'workers')
    )
    sampler = DeterministicSampler(indices, sampler_generator, shuffle, record=audit)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        persistent_workers=False,
    )
    loader.audit_sampler = sampler if audit else None
    loader.audit_key = repr(loader_key)
    return loader


def relative_sample_path(path, root):
    """Return a stable POSIX-style identity for an ImageFolder sample."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


class _FirstAccessLoader:
    def __init__(self, loader, on_first_iteration, one_shot=False):
        self._loader = loader
        self._on_first_iteration = on_first_iteration
        self._iteration_lock = threading.Lock() if one_shot else None
        self._iterated = False

    def __iter__(self):
        if self._iteration_lock is not None:
            with self._iteration_lock:
                if self._iterated:
                    raise RuntimeError(
                        'formal loader may be iterated exactly once'
                    )
                self._iterated = True
        self._on_first_iteration()
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class TaskManager:
    def __init__(self, args):
        self.args = args
        self.task_classes = {}
        if hasattr(args, 'custom_tasks') and args.custom_tasks:
            for t, cls_str in enumerate(args.custom_tasks.split('|')):
                self.task_classes[t] = [int(c) for c in cls_str.split(',')]
            args.num_tasks = len(self.task_classes)
        else:
            for t in range(args.num_tasks):
                s = t * args.classes_per_task
                self.task_classes[t] = list(range(s, s + args.classes_per_task))
        self.seen_classes = []
        self.forgotten_classes = []
        self.ul_event_idx = 0

    def get_timeline(self):
        timeline = []
        idx = 0
        for t in range(self.args.num_tasks):
            timeline.append({'type':'CIL','task_id':t,'new_classes':self.task_classes[t]})
            if t in self.args.unlearn_after_tasks and idx < len(self.args.unlearn_classes):
                timeline.append({'type':'UL','after_task':t,'forget_classes':self.args.unlearn_classes[idx]})
                idx += 1
        self.ul_event_idx = 0
        return timeline

    def advance_task(self, task_id):
        self.seen_classes.extend(self.task_classes[task_id])

    def apply_unlearn(self, classes):
        self.forgotten_classes.extend(classes)

    def get_effective_classes(self):
        return [c for c in self.seen_classes if c not in self.forgotten_classes]

    def get_all_seen_classes(self):
        return list(self.seen_classes)

    def get_forgotten_classes(self):
        return list(self.forgotten_classes)


class TensorViewDataset(Dataset):
    """In-memory tabular/multi-view dataset returning (x, y) with x a 1-D
    feature vector (the full concatenated multi-view row). The VFL split into
    per-party views happens downstream in split_features via the column-range
    map, exactly mirroring how images are split by width — so every CL/UL method
    that calls split_features(batch_x, args) is unchanged.

    Exposes `.targets` (a python list) so the generic get_class_indices /
    get_task_loaders / get_forget_retain_loaders paths work without special-casing.
    """
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)   # (N, D)
        self.y = torch.as_tensor(y, dtype=torch.long)      # (N,)
        self.targets = self.y.tolist()                     # CIFAR-style label list

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class VFLDataset:
    def __init__(self, args):
        self.args = args
        self._formal_pending_access = None
        self._formal_access_records = []
        if args.data in VECTOR_DATASETS:
            self._init_vector(args)
            return
        if args.data == 'tinyimagenet':
            self._init_tinyimagenet(args)
            return
        if args.data == 'cifar100':
            norm = transforms.Normalize([.507,.487,.441],[.267,.256,.276])
        else:
            norm = transforms.Normalize([.4914,.4822,.4465],[.247,.243,.261])
        tr = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),norm])
        te = transforms.Compose([transforms.ToTensor(),norm])
        DS = datasets.CIFAR100 if args.data == 'cifar100' else datasets.CIFAR10
        self.trainset = DS(args.data_path, True, download=True, transform=tr)
        self.testset = DS(args.data_path, False, transform=te)
        self.calibration_indices = set()
        self.calibration_manifest = None
        self.validation_indices = set()
        self.validation_manifest = None
        if getattr(args, 'bic_enabled', 0):
            if args.data != 'cifar100':
                raise ValueError('formal BiC calibration currently requires CIFAR-100')
            self.calibrationset = DS(args.data_path, True, download=False, transform=te)
            self.calibration_manifest = build_manifest(
                self.trainset.targets, args.bic_per_class, args.bic_split_seed
            )
            self.calibration_indices = manifest_indices(self.calibration_manifest)
            write_manifest(
                self.calibration_manifest,
                os.path.join(args.output_dir, 'bic', 'calibration_manifest.json'),
            )
        if getattr(args, 'lambda_validation_enabled', 0):
            if args.data != 'cifar100':
                raise ValueError('lambda validation currently requires CIFAR-100')
            if not getattr(args, 'bic_enabled', 0):
                raise ValueError('lambda validation requires the fixed BiC holdout')
            self.validationset = DS(args.data_path, True, download=False, transform=te)
            self.validation_manifest = build_manifest(
                self.trainset.targets,
                args.lambda_validation_per_class,
                args.lambda_validation_split_seed,
                excluded_indices=self.calibration_indices,
            )
            self.validation_indices = manifest_indices(self.validation_manifest)
            write_manifest(
                self.validation_manifest,
                os.path.join(args.output_dir, 'validation', 'validation_manifest.json'),
            )

    def _init_tinyimagenet(self, args):
        """TinyImageNet-200 (64x64). Expects a class-foldered layout:
            {data_path}/tiny-imagenet-200/train/<wnid>/images/*.JPEG
            {data_path}/tiny-imagenet-200/val/<wnid>/*.JPEG    (val pre-sorted into class folders)
        If val is still in the raw flat layout, run scripts to reorganize it once,
        or point both splits at train/ subfolders. ImageFolder assigns class ids by
        sorted wnid, so train and val share the same label mapping.
        """
        import os
        root = os.path.join(args.data_path, 'tiny-imagenet-200')
        norm = transforms.Normalize([.480,.448,.398],[.277,.269,.282])
        tr = transforms.Compose([transforms.RandomCrop(64,8),transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),norm])
        te = transforms.Compose([transforms.ToTensor(),norm])
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')
        self.trainset = datasets.ImageFolder(train_dir, transform=tr)
        self.testset = datasets.ImageFolder(val_dir, transform=te)
        # ImageFolder stores labels in .targets (a list) — matches the CIFAR API used below.
        if not hasattr(self.trainset, 'targets'):
            self.trainset.targets = [s[1] for s in self.trainset.samples]
        if not hasattr(self.testset, 'targets'):
            self.testset.targets = [s[1] for s in self.testset.samples]
        expected_classes = int(getattr(args, 'num_classes', 200))
        if (len(self.trainset.classes) != expected_classes
                or self.trainset.classes != self.testset.classes):
            raise ValueError(
                f'TinyImageNet requires {expected_classes} matching train/val classes'
            )
        self.calibration_indices = set()
        self.calibration_manifest = None
        self.validation_indices = set()
        self.validation_manifest = None
        if getattr(args, 'lambda_validation_enabled', 0):
            self.validationset = datasets.ImageFolder(train_dir, transform=te)
            if self.validationset.classes != self.trainset.classes:
                raise ValueError('TinyImageNet validation class mapping mismatch')
            train_sample_ids = [
                relative_sample_path(path, train_dir)
                for path, _ in self.trainset.samples
            ]
            validation_sample_ids = [
                relative_sample_path(path, train_dir)
                for path, _ in self.validationset.samples
            ]
            if set(train_sample_ids) != set(validation_sample_ids):
                raise ValueError('TinyImageNet train/validation views do not match')
            self.validation_manifest = build_manifest(
                self.trainset.targets,
                per_class=50,
                seed=args.lambda_validation_split_seed,
                dataset='tinyimagenet-train',
                sample_ids=train_sample_ids,
            )
            selected = set(self.validation_manifest['ordered_sample_ids'])
            self.validation_indices = {
                index for index, sample_id in enumerate(train_sample_ids)
                if sample_id in selected
            }
            self.validation_view_indices = {
                index for index, sample_id in enumerate(validation_sample_ids)
                if sample_id in selected
            }
            write_manifest(
                self.validation_manifest,
                os.path.join(args.output_dir, 'validation', 'validation_manifest.json'),
            )
        if getattr(args, 'bic_enabled', 0):
            self.calibrationset = datasets.ImageFolder(train_dir, transform=te)
            if self.calibrationset.classes != self.trainset.classes:
                raise ValueError('TinyImageNet calibration class mapping mismatch')
            if not hasattr(self.calibrationset, 'targets'):
                self.calibrationset.targets = [s[1] for s in self.calibrationset.samples]
            self.calibration_manifest = build_manifest(
                self.trainset.targets, args.bic_per_class, args.bic_split_seed,
                dataset='tinyimagenet-train',
            )
            self.calibration_indices = manifest_indices(self.calibration_manifest)
            write_manifest(
                self.calibration_manifest,
                os.path.join(args.output_dir, 'bic', 'calibration_manifest.json'),
            )

    def _init_vector(self, args):
        """Multi-view tabular VFL (e.g. mfeat). Loads the prebuilt npz produced
        by prep_mfeat.py: a z-scored concatenated X, labels y, a fixed train/test
        split, and the per-view (= per natural party) column ranges. The view
        ranges are resolved into the actual party->column map on `args` (used by
        split_features and build_models) via set_vector_party_ranges()."""
        # path resolution: explicit --vector_npz override (used by the synthvfl rho
        # sweep to pick one npz per rho) > per-dataset default.
        path = getattr(args, 'vector_npz', None)
        if not path:
            if args.data == 'mfeat':
                path = getattr(args, 'mfeat_path', None) or os.path.join(args.data_path, 'mfeat', 'mfeat_6view.npz')
            else:
                path = os.path.join(args.data_path, args.data, f'{args.data}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{args.data}] npz not found at {path}. For mfeat run prep_mfeat.py; "
                f"for synthvfl run prep_synth.py, then rsync data/{args.data}/ here.")
        npz = np.load(path, allow_pickle=True)
        X, y = npz['X'], npz['y']
        tr_idx, te_idx = npz['train_idx'], npz['test_idx']
        train_X, train_y = X[tr_idx], y[tr_idx]
        view_names = [str(v) for v in npz['view_names']]
        view_ranges = list(zip(npz['range_lo'].tolist(), npz['range_hi'].tolist()))

        self.trainset = TensorViewDataset(train_X, train_y)
        self.testset = TensorViewDataset(X[te_idx], y[te_idx])
        self.calibration_indices = set()
        self.calibration_manifest = None
        self.validation_indices = set()
        self.validation_manifest = None
        if getattr(args, 'bic_enabled', 0):
            raise ValueError('BiC calibration is disabled for vector protocol selection')
        if getattr(args, 'lambda_validation_enabled', 0):
            self.validationset = TensorViewDataset(train_X, train_y)
            self.validation_manifest = build_manifest(
                self.trainset.targets,
                args.lambda_validation_per_class,
                args.lambda_validation_split_seed,
                dataset=f'{os.path.basename(path)}-train',
            )
            self.validation_indices = manifest_indices(self.validation_manifest)
            write_manifest(
                self.validation_manifest,
                os.path.join(args.output_dir, 'validation', 'validation_manifest.json'),
            )
        self.total_features = int(X.shape[1])
        self.view_names = view_names
        self.view_ranges = view_ranges
        set_vector_party_ranges(args, view_names, view_ranges)
        print(f"[{args.data}] loaded {path}: X={X.shape}, {len(view_names)} views -> "
              f"{args.num_parties} parties, ranges={args.party_col_ranges}")

    def get_class_indices(self, dataset, classes):
        return np.where(np.isin(np.array(dataset.targets), classes))[0].tolist()

    def _loader(self, dataset, indices, key, shuffle, audit=False):
        if getattr(self.args, 'deterministic', 0):
            return make_deterministic_loader(
                dataset, indices, self.args, key, shuffle, audit=audit
            )
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
        )
        loader.audit_key = repr(key)
        return loader

    def _formal_expected_classes(self):
        custom = getattr(self.args, 'custom_tasks', '')
        if type(custom) is str and custom:
            return [
                int(class_id)
                for task in custom.split('|')
                for class_id in task.split(',')
            ]
        if type(custom) is list and all(type(task) is list for task in custom):
            return [int(class_id) for task in custom for class_id in task]
        num_tasks = getattr(self.args, 'num_tasks', None)
        classes_per_task = getattr(self.args, 'classes_per_task', None)
        if (type(num_tasks) is not int or num_tasks <= 0
                or type(classes_per_task) is not int or classes_per_task <= 0):
            raise ValueError('formal access requires an exact task class protocol')
        return list(range(num_tasks * classes_per_task))

    def _validate_formal_access_record(self, record):
        phases = {
            'validation': ('final_validation_pre_install', 'lambda_validation'),
            'calibration': (
                'final_bic_calibration_post_freeze', 'bic_calibration'
            ),
            'test': ('final_test_post_install', 'test'),
        }
        final_task = getattr(self.args, 'num_tasks', None)
        if (type(record) is not dict
                or set(record) != {
                    'event', 'loader_key', 'split', 'phase', 'event_idx',
                    'task_id', 'timeline_step', 'classes',
                }
                or record.get('event') != 'first_iteration'
                or type(record.get('split')) is not str
                or record['split'] not in phases
                or type(record.get('phase')) is not str
                or record['phase'] != phases[record['split']][0]
                or type(record.get('event_idx')) is not int
                or type(record.get('task_id')) is not int
                or type(record.get('timeline_step')) is not str
                or type(record.get('classes')) is not list
                or not record['classes']
                or any(type(class_id) is not int
                       for class_id in record['classes'])
                or len(set(record['classes'])) != len(record['classes'])
                or record['classes'] != self._formal_expected_classes()
                or type(final_task) is not int or final_task <= 0
                or record['event_idx'] != final_task - 1
                or record['task_id'] != final_task - 1
                or record['timeline_step']
                != f'event_{final_task - 1}_CIL'
                or type(record.get('loader_key')) is not str
                or record['loader_key'] != repr((
                    phases[record['split']][1], tuple(record['classes'])
                ))):
            raise ValueError('durable formal access schema/classes are invalid')
        return record

    def _formal_access_paths(self, split, create=True):
        directory = Path(self.args.output_dir) / 'formal_access'
        if create:
            from adaptive_consolidation_audit import _ensure_child_dir
            directory = _ensure_child_dir(self.args.output_dir, 'formal_access')
        return (
            directory / 'FORMAL_ACCESS_PENDING.json',
            directory / f'{split}.consumed.json',
        )

    def _read_formal_marker(self, path, status):
        from adaptive_consolidation_audit import _read_file
        if not os.path.lexists(path.parent):
            return None
        try:
            content, _ = _read_file(path)
        except FileNotFoundError:
            return None
        try:
            marker = json.loads(content.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('durable formal access marker is invalid') from error
        if (type(marker) is not dict
                or set(marker) != {'schema_version', 'status', 'access'}
                or marker.get('schema_version') != 1
                or marker.get('status') != status):
            raise ValueError('durable formal access marker schema is invalid')
        self._validate_formal_access_record(marker.get('access'))
        return marker

    def _formal_audit_records(self):
        from adaptive_consolidation_audit import _read_file
        if not os.path.lexists(self.args.output_dir):
            return []
        path = Path(self.args.output_dir) / 'data_flow_audit.jsonl'
        try:
            content, _ = _read_file(path)
        except FileNotFoundError:
            return []
        try:
            records = [
                json.loads(line) for line in content.decode('utf-8').splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('durable formal access audit is invalid') from error
        formal = []
        for record in records:
            if (type(record) is dict and record.get('event') == 'first_iteration'
                    and 'phase' in record):
                self._validate_formal_access_record(record)
                formal.append(record)
        splits = [record['split'] for record in formal]
        if splits not in (
                [], ['validation'], ['calibration'], ['test'],
                ['validation', 'test'], ['calibration', 'test']):
            raise ValueError('durable formal access phase order is invalid')
        return formal

    def _formal_audit_contains(self, expected):
        return expected in self._formal_audit_records()

    def _formal_durable_state(self):
        history = self._formal_audit_records()
        authorization_path, _ = self._formal_access_paths(
            'validation', create=False
        )
        consuming_path = authorization_path.with_name(
            'FORMAL_ACCESS_CONSUMING.json'
        )
        authorization = self._read_formal_marker(
            authorization_path, 'authorized'
        )
        consuming = self._read_formal_marker(
            consuming_path, 'authorized'
        )
        consumed_by_split = {}
        for split in ('validation', 'calibration', 'test'):
            _pending_path, consumed_path = self._formal_access_paths(
                split, create=False
            )
            consumed = self._read_formal_marker(consumed_path, 'consumed')
            if consumed is not None:
                if (consumed['access']['split'] != split
                        or consumed['access'] not in history):
                    raise ValueError('durable formal consumption audit is missing')
                consumed_by_split[split] = consumed
        if consuming is not None:
            record = consuming['access']
            consumed = consumed_by_split.get(record['split'])
            if (record in history and consumed is not None
                    and consumed['access'] == record):
                self._remove_formal_marker(consuming_path)
                self._remove_formal_marker(authorization_path)
                return history, []
            raise RuntimeError('formal access consumption is incomplete')
        if authorization is None:
            return history, []
        record = authorization['access']
        consumed = consumed_by_split.get(record['split'])
        if record in history:
            if consumed is not None and consumed['access'] != record:
                raise ValueError('durable formal consumption identity mismatch')
            self._remove_formal_marker(authorization_path)
            return history, []
        if consumed is not None:
            raise ValueError('durable formal consumption audit is missing')
        _pending_path, consumed_path = self._formal_access_paths(
            record['split'], create=False
        )
        self._validate_formal_phase_transition(record, history)
        return history, [{
            'record': record,
            'classes': list(record['classes']),
            'reserved': False,
            'authorization_path': authorization_path,
            'consumed_path': consumed_path,
        }]

    def _remove_formal_marker(self, path):
        from adaptive_consolidation_audit import (
            _read_file, _same_file_identity, _trusted_dir,
        )
        try:
            _content, pinned = _read_file(path)
        except FileNotFoundError:
            return
        with _trusted_dir(path.parent) as directory:
            current = os.stat(
                path.name, dir_fd=directory, follow_symlinks=False
            )
            if (not stat.S_ISREG(current.st_mode)
                    or not _same_file_identity(pinned, current)):
                raise RuntimeError('formal pending identity changed before clear')
            os.unlink(path.name, dir_fd=directory)
            os.fsync(directory)

    def _claim_formal_pending(self, pending):
        from adaptive_consolidation_audit import (
            _read_file, _same_file_identity, _same_inode_content,
            _trusted_dir,
        )
        source = pending['authorization_path']
        claim = source.with_name('FORMAL_ACCESS_CONSUMING.json')
        try:
            _content, pinned = _read_file(source)
        except FileNotFoundError as error:
            raise RuntimeError('formal access capability was already claimed') \
                from error
        marker = self._read_formal_marker(source, 'authorized')
        if marker is None or marker['access'] != pending['record']:
            raise ValueError('formal pending claim identity mismatch')
        with _trusted_dir(source.parent) as directory:
            try:
                current = os.stat(
                    source.name, dir_fd=directory, follow_symlinks=False
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    'formal access capability was already claimed'
                ) from error
            if (not stat.S_ISREG(current.st_mode)
                    or not _same_file_identity(pinned, current)):
                raise RuntimeError('formal pending identity changed before claim')
            try:
                os.link(
                    source.name, claim.name,
                    src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise RuntimeError(
                    'formal access capability was already claimed'
                ) from None
            installed = os.stat(
                claim.name, dir_fd=directory, follow_symlinks=False
            )
            if (not stat.S_ISREG(installed.st_mode)
                    or not _same_inode_content(pinned, installed)):
                raise RuntimeError('formal consuming claim identity mismatch')
            source_now = os.stat(
                source.name, dir_fd=directory, follow_symlinks=False
            )
            if not _same_inode_content(pinned, source_now):
                raise RuntimeError('formal pending identity changed during claim')
            os.unlink(source.name, dir_fd=directory)
            os.fsync(directory)
        pending['claim_path'] = claim

    def _validate_formal_phase_transition(self, record, history):
        splits = [item['split'] for item in history]
        if record['split'] in splits:
            raise ValueError('duplicate or stale formal access phase')
        validation_required = bool(getattr(
            self.args, 'head_consolidation_enabled', False
        ))
        calibration_required = (
            bool(getattr(self.args, 'bic_enabled', False))
            and not validation_required
        )
        if record['split'] == 'validation':
            if not validation_required:
                raise ValueError('formal validation phase is not authorized')
            if splits:
                raise ValueError('formal validation phase order is invalid')
            return
        if record['split'] == 'calibration':
            if not calibration_required:
                raise ValueError('formal calibration phase is not authorized')
            if splits:
                raise ValueError('formal calibration phase order is invalid')
            return
        expected = (
            ['validation'] if validation_required
            else ['calibration'] if calibration_required
            else []
        )
        if splits != expected:
            raise ValueError(
                'formal test phase requires ordered validation/calibration'
            )

    def _append_formal_access_record(self, record):
        from adaptive_consolidation_audit import _trusted_dir
        self._validate_formal_access_record(record)
        if self._formal_audit_contains(record):
            raise ValueError('duplicate or stale formal access consumption')
        payload = (json.dumps(record, sort_keys=True) + '\n').encode('utf-8')
        with _trusted_dir(self.args.output_dir) as directory:
            descriptor = os.open(
                'data_flow_audit.jsonl',
                os.O_WRONLY | os.O_APPEND | os.O_CREAT
                | getattr(os, 'O_NOFOLLOW', 0),
                0o600,
                dir_fd=directory,
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError('formal access audit target is not regular')
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError('short formal access audit write')
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory)

    def authorize_formal_access(
            self, *, split, phase, event_idx, task_id, timeline_step, classes):
        """Authorize exactly one final validation/calibration/test iteration."""
        if not bool(getattr(self.args, 'formal_deferred_evaluation', False)):
            raise RuntimeError('formal access authorization requires formal mode')
        phases = {
            'validation': 'final_validation_pre_install',
            'calibration': 'final_bic_calibration_post_freeze',
            'test': 'final_test_post_install',
        }
        if type(split) is not str or split not in phases:
            raise ValueError(
                'formal access split must be validation, calibration, or test'
            )
        if type(phase) is not str or phase != phases[split]:
            raise ValueError('formal access phase does not match its split')
        if type(event_idx) is not int:
            raise TypeError('formal access event_idx must be an exact int')
        if type(task_id) is not int:
            raise TypeError('formal access task_id must be an exact int')
        if type(timeline_step) is not str:
            raise TypeError('formal access timeline_step must be an exact str')
        final_task = getattr(self.args, 'num_tasks', None)
        if type(final_task) is not int or final_task <= 0:
            raise ValueError('formal access requires a positive num_tasks')
        final_task -= 1
        if event_idx != final_task or task_id != final_task:
            raise ValueError('early or stale formal access boundary')
        if timeline_step != f'event_{event_idx}_CIL':
            raise ValueError('formal access timeline key is invalid')
        if (type(classes) is not list
                or not classes
                or any(type(class_id) is not int for class_id in classes)
                or len(set(classes)) != len(classes)
                or classes != self._formal_expected_classes()):
            raise ValueError('formal access classes must be the full ordered classes')
        if getattr(self, '_formal_pending_access', None) is not None:
            raise RuntimeError('previous formal authorization is unconsumed')
        prefix = {
            'validation': 'lambda_validation',
            'calibration': 'bic_calibration',
            'test': 'test',
        }[split]
        record = {
            'event': 'first_iteration',
            'loader_key': repr((prefix, tuple(classes))),
            'split': split,
            'phase': phase,
            'event_idx': event_idx,
            'task_id': task_id,
            'timeline_step': timeline_step,
            'classes': list(classes),
        }
        self._validate_formal_access_record(record)
        records = getattr(self, '_formal_access_records', [])
        if record in records:
            raise ValueError('duplicate or stale formal authorization')
        history, pending = self._formal_durable_state()
        if pending:
            raise RuntimeError(
                'a formal authorization is already pending and unconsumed'
            )
        self._validate_formal_phase_transition(record, history)
        authorization_path, consumed_path = self._formal_access_paths(split)
        if self._formal_audit_contains(record):
            raise ValueError('duplicate or stale formal authorization')
        consumed = self._read_formal_marker(consumed_path, 'consumed')
        if consumed is not None:
            raise ValueError('formal access identity was already consumed')
        authorization = self._read_formal_marker(
            authorization_path, 'authorized'
        )
        if authorization is not None:
            if authorization['access'] == record:
                raise RuntimeError('previous formal authorization is unconsumed')
            raise ValueError('stale or conflicting formal authorization')
        from adaptive_consolidation_audit import atomic_write_new_json
        marker = {
            'schema_version': 1,
            'status': 'authorized',
            'access': record,
        }
        try:
            atomic_write_new_json(authorization_path, marker)
        except FileExistsError:
            authorization = self._read_formal_marker(
                authorization_path, 'authorized'
            )
            if authorization is not None and authorization['access'] == record:
                raise RuntimeError(
                    'previous formal authorization is unconsumed'
                ) from None
            raise ValueError('stale or conflicting formal authorization') from None
        self._formal_access_records = records
        self._formal_pending_access = {
            'record': record,
            'classes': list(classes),
            'reserved': False,
            'authorization_path': authorization_path,
            'consumed_path': consumed_path,
        }

    def authorize_formal_calibration_access(
            self, *, phase, event_idx, task_id, timeline_step, classes):
        return self.authorize_formal_access(
            split='calibration', phase=phase,
            event_idx=event_idx, task_id=task_id,
            timeline_step=timeline_step, classes=classes,
        )

    def _reserve_formal_access(self, split, classes):
        if not bool(getattr(self.args, 'formal_deferred_evaluation', False)):
            return None
        pending = getattr(self, '_formal_pending_access', None)
        if pending is None:
            _history, durable_pending = self._formal_durable_state()
            if not durable_pending:
                raise RuntimeError('formal loader access requires authorization')
            pending = durable_pending[0]
            self._formal_pending_access = pending
        if pending['reserved']:
            raise RuntimeError('formal authorization is unconsumed')
        self._validate_formal_access_record(pending['record'])
        _history, durable_pending = self._formal_durable_state()
        if (len(durable_pending) != 1
                or durable_pending[0]['record'] != pending['record']):
            raise ValueError('durable formal authorization identity mismatch')
        if pending['record']['split'] != split:
            raise ValueError('formal access split does not match authorization')
        ordered = list(classes) if type(classes) in (list, tuple) else None
        if (ordered is None
                or any(type(class_id) is not int for class_id in ordered)
                or ordered != pending['classes']):
            raise ValueError('formal access classes/key do not match authorization')
        prefix = {
            'validation': 'lambda_validation',
            'calibration': 'bic_calibration',
            'test': 'test',
        }[split]
        if pending['record']['loader_key'] != repr((prefix, tuple(ordered))):
            raise ValueError('formal access loader key does not match authorization')
        self._claim_formal_pending(pending)
        pending['reserved'] = True
        return pending

    def _reject_unconsumed_formal_access(self):
        if not bool(getattr(self.args, 'formal_deferred_evaluation', False)):
            return
        _history, durable_pending = self._formal_durable_state()
        if (getattr(self, '_formal_pending_access', None) is not None
                or durable_pending):
            raise RuntimeError('formal authorization is unconsumed or pending')

    def _audit_first_access(self, loader, split, formal_access=None):
        if bool(getattr(self.args, 'formal_deferred_evaluation', False)):
            if formal_access is None:
                raise RuntimeError('formal loader access requires authorization')

            def record_formal():
                pending = getattr(self, '_formal_pending_access', None)
                if pending is not formal_access or not pending['reserved']:
                    raise RuntimeError('formal authorization became stale')
                authorization = self._read_formal_marker(
                    pending['claim_path'], 'authorized'
                )
                if (authorization is None
                        or authorization['access'] != pending['record']):
                    raise ValueError(
                        'durable formal authorization identity mismatch'
                    )
                self._append_formal_access_record(pending['record'])
                from adaptive_consolidation_audit import atomic_write_new_json
                atomic_write_new_json(pending['consumed_path'], {
                    'schema_version': 1,
                    'status': 'consumed',
                    'access': pending['record'],
                })
                self._remove_formal_marker(pending['claim_path'])
                records = getattr(self, '_formal_access_records', [])
                records.append(pending['record'])
                self._formal_access_records = records
                self._formal_pending_access = None

            return _FirstAccessLoader(loader, record_formal, one_shot=True)
        adaptive = (
            getattr(self.args, 'head_consolidation_enabled', 0)
            and getattr(self.args, 'head_consolidation_mode', '') == 'adaptive_dual_branch'
        )
        if not getattr(self.args, 'data_flow_audit', 0) and not adaptive:
            return loader

        def record():
            accessed = getattr(self, '_first_accessed_splits', set())
            if split in accessed:
                return
            accessed.add(split)
            self._first_accessed_splits = accessed
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, 'data_flow_audit.jsonl'),
                      'a', encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'event': 'first_iteration',
                    'loader_key': loader.audit_key,
                    'split': split,
                }, sort_keys=True) + '\n')

        return _FirstAccessLoader(loader, record)

    def get_train_loader(self, classes, shuffle=True):
        tr_idx = self.get_class_indices(self.trainset, classes)
        if getattr(self.args, 'bic_enabled', 0):
            tr_idx = [index for index in tr_idx if index not in self.calibration_indices]
        if getattr(self.args, 'lambda_validation_enabled', 0):
            tr_idx = [index for index in tr_idx if index not in self.validation_indices]
        if tr_idx:
            self._reject_unconsumed_formal_access()
        return self._loader(
            self.trainset,
            tr_idx,
            ('train', tuple(int(c) for c in classes)),
            shuffle,
            audit=bool(getattr(self.args, 'data_flow_audit', 0)),
        )

    def get_validation_loader(self, classes):
        formal_access = self._reserve_formal_access('validation', classes)
        if not getattr(self.args, 'lambda_validation_enabled', 0):
            raise RuntimeError('lambda validation is disabled')
        selected = getattr(self, 'validation_view_indices', self.validation_indices)
        indices = [
            index for index in self.get_class_indices(self.validationset, classes)
            if index in selected
        ]
        loader = self._loader(
            self.validationset, indices,
            ('lambda_validation', tuple(int(c) for c in classes)), False,
        )
        return self._audit_first_access(
            loader, 'validation', formal_access=formal_access
        )

    def get_test_loader(self, classes):
        formal_access = self._reserve_formal_access('test', classes)
        indices = self.get_class_indices(self.testset, classes)
        loader = self._loader(
            self.testset, indices, ('test', tuple(int(c) for c in classes)), False,
        )
        return self._audit_first_access(loader, 'test', formal_access=formal_access)

    def get_task_loaders(self, classes, shuffle_train=True):
        evaluation = (
            self.get_validation_loader(classes)
            if getattr(self.args, 'lambda_validation_enabled', 0)
            else self.get_test_loader(classes)
        )
        return self.get_train_loader(classes, shuffle_train), evaluation

    def get_calibration_loader(self, classes):
        if not getattr(self.args, 'bic_enabled', 0):
            raise RuntimeError('BiC calibration is disabled')
        formal_access = self._reserve_formal_access('calibration', classes)
        indices = [
            index for index in self.get_class_indices(self.calibrationset, classes)
            if index in self.calibration_indices
        ]
        if getattr(self.args, 'deterministic', 0):
            loader = make_deterministic_loader(
                self.calibrationset,
                indices,
                self.args,
                ('bic_calibration', tuple(int(c) for c in classes)),
                False,
            )
        else:
            loader = DataLoader(
                Subset(self.calibrationset, indices),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
            )
        return self._audit_first_access(
            loader, 'calibration', formal_access=formal_access
        )

    def calibration_audit(self):
        selected = set(self.calibration_indices)
        validation = set(getattr(self, 'validation_indices', set()))
        training = set(range(len(self.trainset))) - selected - validation
        return {
            'passed': not selected.intersection(training | validation),
            'manifest_sha256': self.calibration_manifest['sha256'],
            'per_class': self.calibration_manifest['per_class'],
            'calibration_count': len(selected),
            'training_count': len(training),
            'overlap_count': len(selected.intersection(training)),
            'test_used_for_fit': False,
        }

    def selection_audit(self):
        calibration = set(self.calibration_indices)
        validation = set(self.validation_indices)
        training = set(range(len(self.trainset))) - calibration - validation
        calibration_manifest = self.calibration_manifest or {}
        validation_manifest = self.validation_manifest or {}
        overlaps = {
            'training_calibration_overlap_count': len(training & calibration),
            'training_validation_overlap_count': len(training & validation),
            'calibration_validation_overlap_count': len(calibration & validation),
        }
        dataset_name = getattr(getattr(self, 'args', None), 'data', 'cifar100')
        return {
            'passed': not any(overlaps.values()) and bool(validation),
            'calibration_manifest_sha256': calibration_manifest.get('sha256'),
            'validation_manifest_sha256': validation_manifest.get('sha256'),
            'calibration_per_class': calibration_manifest.get('per_class', {}),
            'validation_per_class': validation_manifest.get('per_class', {}),
            'training_count': len(training),
            'calibration_count': len(calibration),
            'validation_count': len(validation),
            **overlaps,
            'evaluation_source': (
                'vector-train-validation'
                if dataset_name in VECTOR_DATASETS
                else f'{dataset_name}-train-validation'
            ),
            'test_used_for_selection': False,
        }

    def get_forget_retain_loaders(self, forget_classes, all_seen):
        retain = [c for c in all_seen if c not in forget_classes]
        return {
            'forget_train': DataLoader(Subset(self.trainset, self.get_class_indices(self.trainset, forget_classes)),
                                       batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers),
            'retain_train': DataLoader(Subset(self.trainset, self.get_class_indices(self.trainset, retain)),
                                       batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers),
            'forget_test': DataLoader(Subset(self.testset, self.get_class_indices(self.testset, forget_classes)),
                                      batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers),
            'retain_test': DataLoader(Subset(self.testset, self.get_class_indices(self.testset, retain)),
                                      batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers),
        }


def set_vector_party_ranges(args, view_names, view_ranges):
    """Resolve the per-view column ranges into the per-PARTY column map and store
    it on args as `args.party_col_ranges` (list of (lo, hi) per party).

    - num_parties == #views (default 6 for mfeat): each party = one view.
    - num_parties  < #views: consecutive views are grouped contiguously into
      parties as evenly as possible (views are adjacent column blocks already,
      so a group is still a single contiguous slice -> one Linear input dim).
    - num_parties  > #views is unsupported (a view can't be sub-split sensibly).
    """
    n_views = len(view_ranges)
    P = args.num_parties
    if P == n_views:
        groups = [[k] for k in range(n_views)]
    elif P < n_views:
        # contiguous near-equal grouping of views into P parties
        base, rem = divmod(n_views, P)
        groups, start = [], 0
        for p in range(P):
            cnt = base + (1 if p < rem else 0)
            groups.append(list(range(start, start + cnt)))
            start += cnt
    else:
        raise ValueError(
            f"num_parties={P} > #views={n_views}: cannot split a single view "
            f"into multiple parties for {args.data}. Use num_parties <= {n_views}.")
    ranges = []
    for g in groups:
        lo = view_ranges[g[0]][0]
        hi = view_ranges[g[-1]][1]
        ranges.append((int(lo), int(hi)))
    args.party_col_ranges = ranges
    args.party_view_groups = [[view_names[k] for k in g] for g in groups]
    return ranges


def get_party_widths(args, w=None):
    """Per-party column widths for the vertical feature split.

    - Homogeneous (default): equal widths; the remainder (when num_parties does
      not divide w) is given to the last party so no pixels are dropped.
    - Feature-quantity heterogeneity: if args.party_widths is set (list of ints
      summing to w), parties hold unequal amounts of features.

    Width resolution: explicit `w` (e.g. the real tensor width) > args.img_size > 32.

    Vector/multi-view data (mfeat): widths are the per-party feature counts taken
    from args.party_col_ranges (hi - lo). build_models uses these to size each
    party's MLP input Linear.
    """
    if args.data in VECTOR_DATASETS:
        ranges = getattr(args, 'party_col_ranges', None)
        if not ranges:
            raise RuntimeError("party_col_ranges not set; VFLDataset must be built "
                               "before get_party_widths for vector data.")
        return [hi - lo for (lo, hi) in ranges]
    if w is None:
        w = getattr(args, 'img_size', 32)
    pw = getattr(args, 'party_widths', None)
    if pw:
        widths = list(pw)
    else:
        base = w // args.num_parties
        widths = [base] * args.num_parties
        widths[-1] += w - base * args.num_parties  # remainder -> last party
    return widths


def split_features(batch_x, args):
    """Split a batch into N per-party feature tensors.

    - Image data (4-D NCHW): slice along the width axis (vertical feature split).
    - Vector/multi-view data (2-D N x D, e.g. mfeat): slice the concatenated
      feature vector by the party->column-range map (each party = one view by
      default). Returns a list of (N, d_k) tensors, one per party.
    """
    if args.data in VECTOR_DATASETS:
        ranges = getattr(args, 'party_col_ranges', None)
        if not ranges:
            raise RuntimeError("party_col_ranges not set; build the dataset first.")
        return [batch_x[:, lo:hi] for (lo, hi) in ranges]
    widths = get_party_widths(args, w=batch_x.size(3))
    parts, off = [], 0
    for wd in widths:
        parts.append(batch_x[:, :, :, off:off + wd])
        off += wd
    return parts
