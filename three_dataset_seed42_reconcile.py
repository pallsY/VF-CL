"""Read-only reconciliation of two distinct, immutable seed-42 authorities."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import MappingProxyType
from urllib.parse import quote


PILOT_ROOT = '/home/c3080/YangXiaoXiang/VF-CL/results/three_dataset_seed42_pilot_20260907_065121'
PILOT_COMMIT = 'f79062f20c54bb43849ecfc43682d643be23afe6'
BASE_COMMIT = 'c3ad9214759a5819e682bc50c7db667d41934e1d'
APPROVED_SOURCE_MIGRATIONS = MappingProxyType({
    'adaptive_consolidation_audit.py': (
        MappingProxyType({
            'commit': '16af9718df58460d3552ca4723ef574874799b8c',
            'parent': 'd8a6276dac59f3635eb9872a5f0cad35f1449a34',
            'old_mode': '100644',
            'new_mode': '100644',
            'old_blob': '519533d610ac1a9fe711696f7ba7586df2398b50',
            'new_blob': '97dd58e0c096cd37abcd5e94125a027f8c91557d',
            'old_sha256': 'aad4c89ff1adc25828fedf31cd2ae2b95d38fb967077d19b40975b1531e2cd97',
            'new_sha256': '251d81fc79c5f26ac32b1a7a809d9fa5e4eff4280bd28ed0beeadd838880d7b0',
        }),
        MappingProxyType({
            'commit': '3301995c170648f2a61a8b0625ae6efc1f49ec83',
            'parent': '7cba0dc00edf57acd281ea1559d3ee57a5d744e9',
            'old_mode': '100644',
            'new_mode': '100644',
            'old_blob': '97dd58e0c096cd37abcd5e94125a027f8c91557d',
            'new_blob': '022804121bf8472b1076d1f1898ad4b4aefb0ad4',
            'old_sha256': '251d81fc79c5f26ac32b1a7a809d9fa5e4eff4280bd28ed0beeadd838880d7b0',
            'new_sha256': '712b2293d1dccb64b775e0b3c9d9e5208c8ba369df7f6368dc19d519fadc3bf4',
        }),
        MappingProxyType({
            'commit': '4dcdb667c71eaab458798c0ec662f53201721990',
            'parent': '3301995c170648f2a61a8b0625ae6efc1f49ec83',
            'old_mode': '100644',
            'new_mode': '100644',
            'old_blob': '022804121bf8472b1076d1f1898ad4b4aefb0ad4',
            'new_blob': '5d5a14f0a580628c9b245da29c58dcfb59d7a656',
            'old_sha256': '712b2293d1dccb64b775e0b3c9d9e5208c8ba369df7f6368dc19d519fadc3bf4',
            'new_sha256': '5f1a6ef1e5d2cbf255cde29e07ffcbaebfdaf58538e6daf005bdb2d484708839',
        }),
    ),
    'runner.py': (
        MappingProxyType({
            'commit': '3301995c170648f2a61a8b0625ae6efc1f49ec83',
            'parent': '7cba0dc00edf57acd281ea1559d3ee57a5d744e9',
            'old_mode': '100644',
            'new_mode': '100644',
            'old_blob': '6581ebf08a6ab6d73b12bf8766d1683eedd35104',
            'new_blob': '17805220fe66d5ba9f78d165670ddff937741d8f',
            'old_sha256': '37d0280b0d2b2e2e1d586e24d0572353c481420adee61806ba71571dad02b8e2',
            'new_sha256': '184fc6c28046862b15150d698ccf64e85635c6f87556c904189e37c602353439',
        }),
    ),
})
# These byte hashes pin the complete canonical identity (including token,
# device/inode/creation metadata), registry, plan, and missing-job authority.
PILOT_PIN = {
    'FORMAL_ROOT_IDENTITY.json': '0489e2df04f6b44adbd7eefc0bff603c1de6c765bef3f3459b5b8d4c3292e00b',
    'FORMAL_REGISTRY.json': '37a1e6af435a21d7eb32aba9c6375ef7fa24cf1af05e9ba87f9dd6640dba8542',
    'FORMAL_PLAN.json': '1ecb1c5b0fab243d89a43324424890482e597fba55fc9c210bd2796e2c2f3b59',
    'MISSING_JOBS.json': 'ee774c40a6ea6ee0f00981e420554bf0f16cf0db526ab0fc2025398ef426aa55',
}
PILOT_EVIDENCE_SHA256 = '6953311df895896bb9d08f9f5f75b465e6e650b05ebc037a983f1a5485bdadb6'
DATASETS = ('cifar100', 'isolet', 'upmc_food101')
METHODS = ('finetune', 'lwf', 'er', 'afc', 'adaptive')
PROFILE = 'seed42-adaptive-recovery'
FORMULA = 'final-minus-diagonal-v1'
SHA = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
SCALAR_METRICS = ('aa_final', 'bwt', 'taskil_final')
VECTOR_METRICS = ('aa_trajectory', 'class_final', 'taskil_final_by_task')
RECORD_KEYS = set(('kind spec_key dataset method seed explanation registry_sha256 '
    'metric_formula_version plan_sha256 source_commit protocol_sha256 trajectory_sha256 '
    'admission_record_sha256 artifact_sha256 metrics command_sha256 log_sha256 '
    'claim_sha256 launch_sha256 resource record_sha256').split())
JOB_KEYS = set(('kind spec_key spec registry_sha256 metric_formula_version plan_sha256 '
    'run_dir source_commit source_sha256 command command_sha256 root_identity').split())
OWNER_KEYS = set(('kind job launcher_token worker_role pid pgid phase process_start_time '
                  'source_commit root_identity').split())
RESOURCE_KEYS = set(('hardware_identity instrumentation runtime_seconds peak_gpu_memory_bytes '
    'checkpoint_size_bytes added_parameters communication_bytes replay_type '
    'raw_examples_per_class persistent_embeddings privacy_label').split())
ARTIFACT_PATHS = {'checkpoint': 'checkpoints/formal_final.pt', 'config': 'config.json',
    'results': 'results.json', 'data_flow': 'data_flow_audit.jsonl',
    'validation_manifest': 'validation/validation_manifest.json'}
ALLOWED_SOURCE_CHANGES = {'three_dataset_formal_audit.py', 'three_dataset_formal_driver.py',
    'three_dataset_formal_registry.py', 'three_dataset_seed42_reconcile.py',
    'three_dataset_full_matrix_reconcile.py',
    'run_three_dataset_formal_comparison.sh', 'run_three_dataset_seed42_pilot.sh',
    'run_three_dataset_adaptive_recovery.sh', 'prune_completed_runs.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def digest(value):
    return sha(canonical(value))


def identical(left, right):
    return canonical(left) == canonical(right)


def schema(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), label + ' schema differs')


def hashes(value, label):
    require(type(value) is dict and bool(value), label + ' hashes missing')
    for name, value in value.items():
        require(type(name) is str and name and type(value) is str
                and SHA.fullmatch(value), label + ' hash invalid')


def relative(name):
    require(type(name) is str and name and '\\' not in name and '\x00' not in name,
            'unsafe relative path')
    parts = PurePosixPath(name).parts
    require(not name.startswith('/') and all(p not in ('.', '..') for p in parts)
            and '/'.join(parts) == name, 'unsafe relative path')
    return parts


def signature(details):
    return (details.st_dev, details.st_ino, details.st_mode, details.st_nlink,
            details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def directory(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'root must be absolute')
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class Evidence:
    """Pin input directories and verify every consumed file again before publication."""
    def __init__(self, path):
        self.path = Path(path)
        self.fd = directory(self.path)
        self.details = os.fstat(self.fd)
        self.seen = {}
        self.directories = {}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        os.close(self.fd)

    def parent(self, name):
        parts = relative(name)
        fd = os.dup(self.fd)
        try:
            for index, part in enumerate(parts[:-1]):
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
                self.pin_directory('/'.join(parts[:index + 1]), fd)
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def pin_directory(self, name, fd):
        current = signature(os.fstat(fd))
        require(self.directories.setdefault(name, current) == current, 'input directory changed')

    def read(self, name, immutable=False, hash_only=False):
        parent, leaf = self.parent(name)
        try:
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                    name + ' is not a single-link regular file (symlink/type)')
            require(not immutable or stat.S_IMODE(before.st_mode) == 0o444,
                    name + ' is not immutable mode 0444')
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, 'rb') as stream:
                require(signature(os.fstat(stream.fileno())) == signature(before), 'input replaced')
                if hash_only:
                    hasher = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                    content = hasher.hexdigest()
                else:
                    content = stream.read()
                require(signature(os.fstat(stream.fileno())) == signature(before), 'input mutated')
            require(signature(os.stat(leaf, dir_fd=parent, follow_symlinks=False)) == signature(before),
                    'input name replaced')
            previous = self.seen.setdefault(name, signature(before))
            require(previous == signature(before), 'input changed between reads')
            return content
        finally:
            os.close(parent)

    def json(self, name):
        content = self.read(name, immutable=True)
        value = json.loads(content)
        require(content == canonical(value) + b'\n', name + ' is not canonical JSON')
        require(type(value) is dict, name + ' is not an object')
        return value

    def names(self, name=''):
        fd = directory(self.path / name)
        try:
            self.pin_directory(name, fd)
            return set(os.listdir(fd))
        finally:
            os.close(fd)

    def verify(self):
        fd = directory(self.path)
        try:
            require(signature(os.fstat(fd)) == signature(self.details), 'root changed')
        finally:
            os.close(fd)
        for name, expected in self.directories.items():
            fd = directory(self.path / name)
            try:
                require(signature(os.fstat(fd)) == expected, 'input directory membership changed')
            finally:
                os.close(fd)
        for name, expected in self.seen.items():
            parent, leaf = self.parent(name)
            try:
                require(signature(os.stat(leaf, dir_fd=parent, follow_symlinks=False)) == expected,
                        'consumed input changed: ' + name)
            finally:
                os.close(parent)


def git(worktree, *args):
    fd = directory(worktree)
    os.close(fd)
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(worktree), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout


def git_bytes(worktree, *args):
    fd = directory(worktree)
    os.close(fd)
    return subprocess.run(
        ['git', '--no-optional-locks', '-C', str(worktree), *args],
        check=True, capture_output=True).stdout


def _source_tree_entry(worktree, revision, path):
    entry = git(worktree, 'ls-tree', revision, '--', path).strip().split(None, 3)
    require(len(entry) == 4 and entry[1] == 'blob' and entry[3] == path,
            'approved source migration blob differs')
    return entry[0], entry[2]


def _approved_source_migrations(worktree, head, changed):
    approved = {}
    fields = {'commit', 'parent', 'old_mode', 'new_mode', 'old_blob', 'new_blob',
              'old_sha256', 'new_sha256'}
    for path, configured in APPROVED_SOURCE_MIGRATIONS.items():
        if path not in changed:
            continue
        relative(path)
        require(type(configured) in (tuple, list) and bool(configured),
                'approved source migration chain invalid')
        transitions = []
        for raw in configured:
            transition = dict(raw)
            require(set(transition) == fields, 'approved source migration schema differs')
            require(all(COMMIT.fullmatch(transition[name]) for name in
                        ('commit', 'parent', 'old_blob', 'new_blob')),
                    'approved source migration commit/blob invalid')
            require(all(SHA.fullmatch(transition[name]) for name in
                        ('old_sha256', 'new_sha256')),
                    'approved source migration hash invalid')
            require(all(re.fullmatch(r'100[0-7]{3}', transition[name]) for name in
                        ('old_mode', 'new_mode')),
                    'approved source migration mode invalid')
            transitions.append(transition)
        touches = git(worktree, 'log', '--full-history', '-m', '--format=%H',
                      BASE_COMMIT + '..' + head, '--', path).splitlines()
        require(touches == [item['commit'] for item in reversed(transitions)],
                'approved source migration history differs')
        for index, transition in enumerate(transitions):
            git(worktree, 'merge-base', '--is-ancestor', transition['commit'], head)
            parents = git(worktree, 'rev-list', '--parents', '-n', '1',
                          transition['commit']).split()
            require(parents == [transition['commit'], transition['parent']],
                    'approved source migration parent differs')
            old_mode, old_blob = _source_tree_entry(
                worktree, transition['parent'], path)
            new_mode, new_blob = _source_tree_entry(
                worktree, transition['commit'], path)
            require((old_mode, old_blob) ==
                    (transition['old_mode'], transition['old_blob']) and
                    (new_mode, new_blob) ==
                    (transition['new_mode'], transition['new_blob']),
                    'approved source migration blob/mode differs')
            require(sha(git_bytes(worktree, 'show', transition['parent'] + ':' + path))
                    == transition['old_sha256'],
                    'approved source migration old hash differs')
            require(sha(git_bytes(worktree, 'show', transition['commit'] + ':' + path))
                    == transition['new_sha256'],
                    'approved source migration new hash differs')
            if index:
                previous = transitions[index - 1]
                require((transition['old_mode'], transition['old_blob'],
                         transition['old_sha256']) ==
                        (previous['new_mode'], previous['new_blob'],
                         previous['new_sha256']),
                        'approved source migration chain differs')
        first, final = transitions[0], transitions[-1]
        require(_source_tree_entry(worktree, PILOT_COMMIT, path) ==
                (first['old_mode'], first['old_blob']) and
                sha(git_bytes(worktree, 'show', PILOT_COMMIT + ':' + path)) ==
                first['old_sha256'], 'approved source migration pilot differs')
        require(_source_tree_entry(worktree, head, path) ==
                (final['new_mode'], final['new_blob']) and
                sha(git_bytes(worktree, 'show', head + ':' + path)) ==
                final['new_sha256'], 'approved source migration final differs')
        approved[path] = transitions
    return approved or None


def provenance(pilot_worktree, recovery_worktree, *, operational_paths=()):
    require(git(pilot_worktree, 'rev-parse', '--verify', PILOT_COMMIT + '^{commit}').strip()
            == PILOT_COMMIT, 'pilot source commit differs')
    head = git(recovery_worktree, 'rev-parse', '--verify', 'HEAD^{commit}').strip()
    require(COMMIT.fullmatch(head) is not None, 'recovery source commit invalid')
    for ancestor in (BASE_COMMIT, PILOT_COMMIT):
        git(recovery_worktree, 'merge-base', '--is-ancestor', ancestor, head)
    # Inspect the entire history, not just net diff: a later revert cannot hide a scientific edit.
    changed = sorted({p.strip('\n') for p in git(recovery_worktree, 'log', '--format=',
                     '--name-only', '-z', '-m', BASE_COMMIT + '..' + head, '--').split('\x00')
                     if p.strip('\n')})
    migration = _approved_source_migrations(recovery_worktree, head, changed)
    for path in changed:
        relative(path)
        require((migration is not None and path in migration)
                or path in ALLOWED_SOURCE_CHANGES or path in operational_paths
                or re.fullmatch(r'test_[a-zA-Z0-9_]+\.py', path)
                or (path.startswith('docs/') and path.endswith('.md')), 'Git scope forbids source: ' + path)
    require(not git(recovery_worktree, 'status', '--porcelain', '--untracked-files=all').strip(),
            'recovery worktree is not clean')
    return head, changed, migration


def _source_compatibility(old_jobs, new_jobs, migration):
    old = old_jobs['isolet:adaptive:42'][0]['source_sha256']
    new = new_jobs['isolet:adaptive:42'][0]['source_sha256']
    require(all(job[0]['source_sha256'] == old for job in old_jobs.values()),
            'old scientific source provenance differs')
    require(all(job[0]['source_sha256'] == new for job in new_jobs.values()),
            'new scientific source provenance differs')
    if migration is None:
        require(old == new, 'scientific source provenance differs')
    else:
        require(set(old) == set(new), 'scientific source inventory differs')
        for path, transitions in migration.items():
            require(old.get(path) == transitions[0]['old_sha256'] and
                    new.get(path) == transitions[-1]['new_sha256'],
                    'approved scientific source migration differs')
        require(all(old[name] == new[name] for name in old if name not in migration),
                'scientific source provenance differs outside approved migration')
    return old, new


def verify_pilot_evidence(root):
    keys = [f'{d}:{m}:42' for d in DATASETS for m in METHODS if m != 'adaptive']
    expected = {f'records/{quote(k, safe="")}.json' for k in keys}
    prune_names = {'PRUNE_PLAN.json', 'PRUNED_EVIDENCE.json'}
    expected.update(f'runs/{quote(k, safe="")}/{n}' for k in keys
                    if k != 'cifar100:afc:42' for n in prune_names)
    actual = {'records/' + name for name in root.names('records')}
    for run in root.names('runs'):
        actual.update(f'runs/{run}/{name}' for name in root.names('runs/' + run) & prune_names)
    require(len(expected) == 34 and actual == expected, 'pilot record/pruning evidence membership differs')
    manifest = []
    for name in sorted(expected):
        content = root.read(name, immutable=True)
        manifest.append({'path': name, 'type': 'regular', 'mode': 0o444,
                         'size': len(content), 'sha256': sha(content)})
    require(digest(manifest) == PILOT_EVIDENCE_SHA256, 'pilot record/pruning evidence manifest hash differs')


def authority(root, recovery, commit):
    names = root.names()
    if not recovery:
        require(str(root.path) == PILOT_ROOT, 'pilot root identity path differs')
        for name, expected in PILOT_PIN.items():
            require(sha(root.read(name, immutable=True)) == expected, 'pilot identity pin differs: ' + name)
        verify_pilot_evidence(root)
    binding = {'experiment_profile': PROFILE} if recovery else {}
    keys = ([f'{d}:adaptive:42' for d in DATASETS] if recovery else
            [f'{d}:{m}:42' for m in METHODS for d in DATASETS])
    registry, plan, missing, census, identity = [root.json(n) for n in (
        'FORMAL_REGISTRY.json', 'FORMAL_PLAN.json', 'MISSING_JOBS.json',
        'COMPATIBILITY_CENSUS.json', 'FORMAL_ROOT_IDENTITY.json')]
    common = {'registry_sha256': registry.get('registry_sha256'), 'metric_formula_version': FORMULA}
    hashes({'registry': common['registry_sha256']}, 'registry')
    require(registry == {'kind': 'formal_registry', **binding, **common,
                        'formal_cells': keys, 'explanation_cells': []}, 'registry membership/profile differs')
    require(plan == {'kind': 'formal_plan', **binding, **common, 'formal_cells': keys,
                    'missing_jobs': keys, 'explanation_cells': [], 'census_sha256': digest(census)},
            'plan membership/profile/hash differs')
    require(missing == {'kind': 'formal_missing_jobs', **common, 'missing_jobs': keys,
                       'plan_sha256': digest(plan)}, 'missing jobs manifest differs')
    schema(census, {'kind', 'records', *common, *binding}, 'census')
    require(all(census.get(k) == v for k, v in {**common, **binding,
            'kind': 'formal_compatibility_census'}.items()), 'census identity differs')
    require(type(census['records']) is list and [r.get('spec_key') for r in census['records']
            if type(r) is dict] == keys, 'census membership differs')
    protocols = {}
    for record in census['records']:
        key = record['spec_key']
        d, m, _ = key.split(':')
        raw = {'status': 'RERUN_REQUIRED', 'reason': 'no_declared_candidate',
               'spec': {'dataset': d, 'method': m, 'seed': 42, 'explanation': False},
               'protocol_sha256': record.get('protocol_sha256'), 'source_sha256': '',
               'artifact_sha256': {}, 'metrics': None, 'metric_formula_version': FORMULA,
               'trajectory_sha256': ''}
        require(canonical(record) == canonical({**raw, 'spec_key': key,
                'admission_record_sha256': digest(raw)}), 'census admission contract differs')
        protocols[key] = raw['protocol_sha256']
    hashes(protocols, 'protocol')
    schema(identity, {'kind', 'token', 'root_dev', 'root_inode', 'root_ctime_ns', 'root_size',
           'registry_sha256', 'plan_sha256', 'missing_jobs_sha256', 'source_commit', *binding}, 'identity')
    for field in ('root_dev', 'root_inode', 'root_ctime_ns', 'root_size'):
        require(type(identity[field]) is int and identity[field] >= 0, 'identity integer invalid')
    hashes({'token': identity['token']}, 'identity')
    require(all(identity.get(k) == v for k, v in {**binding, 'kind': 'formal_root_identity',
            'root_dev': root.details.st_dev, 'root_inode': root.details.st_ino,
            'registry_sha256': registry['registry_sha256'], 'plan_sha256': digest(plan),
            'missing_jobs_sha256': digest(missing), 'source_commit': commit}.items()), 'root identity/commit differs')
    identity_bytes = root.read('FORMAL_ROOT_IDENTITY.json', immutable=True)
    owner_identity = {'dev': root.details.st_dev, 'inode': root.details.st_ino,
                      'ctime_ns': root.seen['FORMAL_ROOT_IDENTITY.json'][-1],
                      'size': len(identity_bytes), 'hash': sha(identity_bytes)}
    successes = {n for n in names if n.endswith('_SUCCESS')}
    expected_success = {'RECOVERY_PHASE_SUCCESS', 'RECOVERY_EXECUTION_SUCCESS'} if recovery else set()
    require(successes == expected_success, 'success marker set differs')
    if recovery:
        require(not {'FAILED_JOB', 'FORMAL_STOPPED'} & names, 'recovery failure marker present')
        for name in expected_success:
            marker = root.json(name)
            require(type(marker.get('exit_code')) is int and marker == {
                    'kind': name.lower(), 'role': 'launcher', 'spec_key': '', 'exit_code': 0},
                    'recovery success marker differs')
        for name in ('audit_queue', 'gpu_claims'):
            require(name in names and not root.names(name), 'recovery queue/GPU claims not drained')
            require(stat.S_IMODE(root.directories[name][2]) == 0o700, 'recovery control directory mode differs')
        require(root.names('claims') == {quote(k, safe='') for k in keys}, 'recovery claim membership differs')
        require(stat.S_IMODE(root.directories['claims'][2]) == 0o700, 'recovery claims directory mode differs')
    else:
        failed, stopped = root.json('FAILED_JOB'), root.json('FORMAL_STOPPED')
        require(type(failed.get('exit_code')) is int and failed == {'kind': 'failed_audit', 'role': 'formal-worker-2',
                'spec_key': 'isolet:adaptive:42', 'exit_code': 1}, 'old failed marker differs')
        require(type(stopped.get('exit_code')) is int and stopped == {'kind': 'formal_stopped', 'role': 'formal',
                'spec_key': '', 'exit_code': 1}, 'old stopped marker differs')
    require(root.names('runs') == {quote(k, safe='') for k in keys}, 'run membership differs')
    return {'identity': identity, 'owner_identity': owner_identity, 'plan': plan,
            'protocols': protocols, 'binding': binding, 'keys': keys, 'commit': commit}


def job_spec(root, auth, key):
    run = 'runs/' + quote(key, safe='')
    job = root.json(run + '/FORMAL_JOB_SPEC.json')
    schema(job, JOB_KEYS, 'job')
    d, m, seed = key.split(':')
    expected = {'kind': 'formal_job_spec', 'spec_key': key,
        'spec': {'dataset': d, 'method': m, 'seed': int(seed), 'explanation': False},
        'registry_sha256': auth['plan']['registry_sha256'], 'metric_formula_version': FORMULA,
        'plan_sha256': digest(auth['plan']), 'source_commit': auth['commit'],
        'root_identity': auth['owner_identity'], 'run_dir': str(root.path / run)}
    require(all(canonical(job[k]) == canonical(v) for k, v in expected.items()), 'job authority differs')
    command = job['command']
    require(type(command) is list and len(command) >= 4 and all(type(t) is str and t
            and '\x00' not in t for t in command), 'job command invalid')
    start = 3 if command[1] == '-c' else 2
    options = command[start:]
    require(len(options) % 2 == 0 and all(t.startswith('--') for t in options[::2])
            and len(set(options[::2])) == len(options[::2]), 'command option structure differs')
    options = dict(zip(options[::2], options[1::2]))
    require(options.get('--results_dir') == str(root.path / 'runs')
            and options.get('--exp_name') == quote(key, safe='') and options.get('--seed') == seed,
            'command authority differs')
    require(options.get('--num_tasks', '').isdigit() and int(options['--num_tasks']) > 0,
            'command task contract invalid')
    require(job['command_sha256'] == digest(command), 'command hash differs')
    hashes(job['source_sha256'], 'job source')
    for name in job['source_sha256']:
        relative(name)
    normalized = list(command)
    normalized[normalized.index('--results_dir') + 1] = '<authority-results-root>'
    return job, normalized, int(options['--num_tasks'])


def pruning(root, run, record, artifact_map):
    names = root.names(run)
    present = {'PRUNE_PLAN.json', 'PRUNED_EVIDENCE.json'} & names
    if not present:
        return {}, None
    require(len(present) == 2, 'pruning evidence pair incomplete')
    plan = root.json(run + '/PRUNE_PLAN.json')
    evidence = root.json(run + '/PRUNED_EVIDENCE.json')
    common = {'policy': 'completed-run-intermediate-v1', 'spec_key': record['spec_key'],
              'source_commit': record['source_commit'], 'record_sha256': record['record_sha256']}
    schema(plan, {'kind', 'files', 'bytes', *common}, 'prune plan')
    require(all(plan.get(k) == v for k, v in {**common, 'kind': 'formal_prune_plan'}.items())
            and type(plan['files']) is list and plan['files'], 'pruning plan identity differs')
    old_entry_keys = {'path', 'sha256', 'size'}
    tombstone_entry_keys = {
        'path', 'sha256', 'size', 'device', 'inode', 'mode'}
    entry_schemas = {frozenset(entry) for entry in plan['files']
                     if type(entry) is dict}
    require(len(entry_schemas) == 1 and entry_schemas.pop() in {
            frozenset(old_entry_keys), frozenset(tombstone_entry_keys)},
            'prune entry schema differs')
    tombstones = set(plan['files'][0]) == tombstone_entry_keys
    removed = {}
    for entry in plan['files']:
        schema(entry, tombstone_entry_keys if tombstones else old_entry_keys,
               'prune entry')
        path = entry['path']
        relative(path)
        require(re.fullmatch(r'(checkpoints|formal_snapshots)/event_[0-9]+_CIL\.pt', path)
                or path == 'checkpoints/resume_latest.pt', 'pruning path forbidden')
        require(path not in removed and type(entry['size']) is int and entry['size'] > 0,
                'pruning entry duplicate/size invalid')
        hashes({'sha256': entry['sha256']}, 'pruning entry')
        if tombstones:
            require(all(type(entry[name]) is int and not isinstance(entry[name], bool)
                        for name in ('device', 'inode', 'mode'))
                    and entry['device'] >= 0 and entry['inode'] > 0
                    and 0 <= entry['mode'] <= 0o7777,
                    'pruning tombstone identity invalid')
        if path == 'checkpoints/resume_latest.pt':
            events = [p for p in artifact_map if re.fullmatch(r'checkpoints/event_[0-9]+_CIL\.pt', p)]
            require(events, 'pruning final checkpoint missing')
            expected = artifact_map[max(events, key=lambda p: int(p.split('_')[1]))]
            require(path not in artifact_map or artifact_map[path] == expected,
                    'pruning resume hash differs from the final event checkpoint')
        else:
            require(path in artifact_map, 'pruning artifact is not in the authoritative map')
            expected = artifact_map[path]
        require(entry['sha256'] == expected, 'pruning artifact hash differs')
        parent, leaf = root.parent(run + '/' + path)
        try:
            try:
                os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError('pruned artifact still present')
        finally:
            os.close(parent)
        removed[path] = entry
    require(type(plan['bytes']) is int and plan['bytes'] == sum(e['size'] for e in removed.values()),
            'pruning byte total differs')
    plan_hash = sha(root.read(run + '/PRUNE_PLAN.json', immutable=True))
    if not tombstones:
        require(identical(evidence, {'kind': 'formal_pruned_evidence', **common,
                'plan_sha256': plan_hash, 'freed_bytes': plan['bytes']}),
                'pruning evidence hash/bytes differs')
    else:
        schema(evidence, {'kind', *common, 'plan_sha256', 'quarantine',
                          'files', 'freed_bytes'}, 'pruning tombstone evidence')
        require(type(evidence['quarantine']) is str and re.fullmatch(
                    r'\.prune-quarantine-[0-9a-f]{32}', evidence['quarantine'])
                and type(evidence['files']) is list,
                'pruning quarantine/tombstone schema differs')
        expected_files = []
        for entry in plan['files']:
            group, name = entry['path'].split('/')
            expected_files.append({
                'path': entry['path'], 'tombstone': f'{group}.{name}',
                'inode': entry['inode'], 'original_size': entry['size'],
                'sha256': entry['sha256'], 'final_size': 0,
            })
        require(identical(evidence, {
                    'kind': 'formal_pruned_evidence', **common,
                    'plan_sha256': plan_hash,
                    'quarantine': evidence['quarantine'],
                    'files': expected_files, 'freed_bytes': plan['bytes'],
                }), 'pruning tombstone evidence differs')
        quarantine = run + '/' + evidence['quarantine']
        names = root.names(quarantine)
        require(names == {item['tombstone'] for item in expected_files},
                'pruning quarantine tombstone membership differs')
        directory_mode = root.directories[quarantine][2]
        require(stat.S_ISDIR(directory_mode)
                and stat.S_IMODE(directory_mode) == 0o700,
                'pruning quarantine mode differs')
        by_path = {entry['path']: entry for entry in plan['files']}
        for item in evidence['files']:
            entry = by_path[item['path']]
            tombstone = quarantine + '/' + item['tombstone']
            require(root.read(tombstone) == b'',
                    'pruning tombstone is not empty')
            details = root.seen[tombstone]
            require(details[0] == entry['device']
                    and details[1] == entry['inode']
                    and stat.S_ISREG(details[2])
                    and stat.S_IMODE(details[2]) == entry['mode']
                    and details[3] == 1 and details[4] == 0,
                    'pruning tombstone identity differs')
    return removed, {'plan_path': str(root.path / run / 'PRUNE_PLAN.json'), 'plan_sha256': plan_hash,
                    'evidence_sha256': sha(root.read(run + '/PRUNED_EVIDENCE.json', immutable=True))}


def completed_claim(root, key, job, owner, owner_hash):
    bundle = 'claims/' + quote(key, safe='')
    require(root.names(bundle) == {'owner.json', 'started.json'}, 'completed claim has no exact durable started evidence')
    require(stat.S_IMODE(root.directories[bundle][2]) == 0o700, 'completed claim directory mode differs')
    require(identical(root.json(bundle + '/owner.json'), owner), 'completed claim owner differs from admitted run')
    expected = {'kind': 'formal_job_started', 'job': key, 'owner_sha256': owner_hash,
        'plan_sha256': job['plan_sha256'], 'command_sha256': job['command_sha256'], 'run_dir': job['run_dir'],
        **{k: owner[k] for k in ('source_commit', 'root_identity', 'worker_role', 'phase',
                               'pid', 'pgid', 'process_start_time')}}
    require(identical(root.json(bundle + '/started.json'), expected), 'completed claim started binding differs')
    start = owner['process_start_time']
    require(start.isascii() and start.isdigit() and int(start) > 0, 'completed claim process start time invalid')
    try:
        process = Path(f'/proc/{owner["pid"]}/stat').read_text()
    except FileNotFoundError:
        return
    fields = process.rsplit(')', 1)[-1].split()
    require(len(fields) > 19 and fields[19].isdigit(), 'completed claim process identity unavailable')
    require(fields[19] != start, 'live completed claim process remains')


def completed(root, auth, key, job, tasks):
    run = 'runs/' + quote(key, safe='')
    path = 'records/' + quote(key, safe='') + '.json'
    record = root.json(path)
    schema(record, RECORD_KEYS | set(auth['binding']), 'completed record')
    require(record['record_sha256'] == digest({k: v for k, v in record.items() if k != 'record_sha256'}),
            'completed record hash differs')
    expected = {'kind': 'formal_completed_run', 'spec_key': key, **job['spec'], **auth['binding'],
        'registry_sha256': job['registry_sha256'], 'metric_formula_version': FORMULA,
        'plan_sha256': job['plan_sha256'], 'source_commit': auth['commit'],
        'protocol_sha256': auth['protocols'][key], 'command_sha256': job['command_sha256']}
    require(all(canonical(record[k]) == canonical(v) for k, v in expected.items()), 'completed record authority differs')
    hashes({k: v for k, v in record.items() if k.endswith('_sha256') and k != 'artifact_sha256'}, 'record')
    hashes(record['artifact_sha256'], 'record artifact')
    claim, launch, resource = [root.json(run + '/' + n) for n in
                              ('CLAIM_OWNER.json', 'LAUNCH_STARTED.json', 'RESOURCE_EVIDENCE.json')]
    schema(claim, OWNER_KEYS, 'claim')
    require(claim['kind'] == 'formal_job_claim' and claim['job'] == key
            and claim['source_commit'] == auth['commit'] and identical(claim['root_identity'], auth['owner_identity'])
            and claim['phase'] == 'formal', 'claim identity differs')
    for name in ('pid', 'pgid'):
        require(type(claim[name]) is int and claim[name] > 0, 'claim process type invalid')
    for name in ('launcher_token', 'worker_role', 'process_start_time'):
        require(type(claim[name]) is str and bool(claim[name]), 'claim text invalid')
    claim_hash = sha(root.read(run + '/CLAIM_OWNER.json', immutable=True))
    job_hash = sha(root.read(run + '/FORMAL_JOB_SPEC.json', immutable=True))
    require(identical(launch, {'kind': 'formal_launch_started', 'spec_key': key,
        'plan_sha256': job['plan_sha256'], 'job_spec_sha256': job_hash, 'claim_sha256': claim_hash,
        'command_sha256': job['command_sha256'], **{k: claim[k] for k in
        ('source_commit', 'root_identity', 'worker_role', 'phase', 'pid', 'pgid', 'process_start_time')}}),
        'launch/job/command provenance differs')
    launch_hash = sha(root.read(run + '/LAUNCH_STARTED.json', immutable=True))
    require(record['claim_sha256'] == claim_hash and record['launch_sha256'] == launch_hash,
            'record producer hashes differ')
    if auth['binding']:
        completed_claim(root, key, job, claim, claim_hash)
    require(identical(resource, {'kind': 'formal_resource_evidence', 'spec_key': key,
        'plan_sha256': job['plan_sha256'], 'job_spec_sha256': job_hash, 'claim_sha256': claim_hash,
        'launch_sha256': launch_hash, 'command_sha256': job['command_sha256'],
        'artifact_sha256': resource.get('artifact_sha256'), 'resource': record['resource']}),
        'resource provenance differs')
    schema(record['resource'], RESOURCE_KEYS, 'resource')
    values = record['resource']
    hardware = values['hardware_identity']
    schema(hardware, {'gpu_name', 'gpu_count', 'cuda', 'torch', 'driver'}, 'hardware')
    require(type(hardware['gpu_count']) is int and hardware['gpu_count'] > 0, 'hardware GPU count type invalid')
    for name in ('gpu_name', 'cuda', 'torch', 'driver'):
        require(type(hardware[name]) is str and bool(hardware[name]), 'hardware text invalid')
    for name in RESOURCE_KEYS - {'hardware_identity', 'runtime_seconds', 'instrumentation', 'replay_type', 'privacy_label'}:
        require(type(values[name]) is int and values[name] >= 0, 'resource integer type invalid')
    require(type(values['runtime_seconds']) is float and math.isfinite(values['runtime_seconds'])
            and values['runtime_seconds'] >= 0, 'resource runtime type invalid')
    for name in ('instrumentation', 'replay_type', 'privacy_label'):
        require(type(values[name]) is str and bool(values[name]), 'resource text invalid')
    artifacts = resource['artifact_sha256']
    hashes(artifacts, 'resource artifact')
    removed, prune = pruning(root, run, record, artifacts)
    for name, expected_hash in artifacts.items():
        relative(name)
        if name not in removed:
            require(root.read(run + '/' + name, hash_only=True) == expected_hash, 'artifact hash differs: ' + name)
    require(values['checkpoint_size_bytes'] > 0 and values['checkpoint_size_bytes']
            == root.seen[run + '/checkpoints/formal_final.pt'][4], 'resource checkpoint size differs')
    require(record['log_sha256'] == artifacts.get('job.log'), 'log hash differs')
    require(record['trajectory_sha256'] == record['artifact_sha256'].get('trajectory'), 'trajectory hash differs')
    require({k[7:]: v for k, v in record['artifact_sha256'].items() if k.startswith('source:')}
            == job['source_sha256'], 'record source provenance differs')
    for logical, expected_hash in record['artifact_sha256'].items():
        if logical.startswith(('source:', 'data:')) or logical == 'trajectory':
            continue
        path_in_run = logical[7:] if logical.startswith('formal:') else ARTIFACT_PATHS.get(logical)
        require(path_in_run is not None and artifacts.get(path_in_run) == expected_hash,
                'record artifact contract differs: ' + logical)
    require(set(ARTIFACT_PATHS) <= set(record['artifact_sha256']), 'record required artifacts missing')
    metrics = record['metrics']
    schema(metrics, {*SCALAR_METRICS, *VECTOR_METRICS}, 'metrics')
    values = [metrics[n] for n in SCALAR_METRICS]
    for name in VECTOR_METRICS:
        require(type(metrics[name]) is list and len(metrics[name]) == tasks, 'metric task contract differs')
        values.extend(metrics[name])
    require(all(type(v) is float and math.isfinite(v) for v in values), 'metric type/value invalid')
    return {'spec_key': key, 'dataset': record['dataset'], 'method': record['method'], 'seed': record['seed'],
            'source_commit': auth['commit'], 'source_root': str(root.path), 'decision': 'ADMITTED',
            'record_path': str(root.path / path), 'record_file_sha256': sha(root.read(path, immutable=True)),
            'record_sha256': record['record_sha256'], 'metrics': metrics, 'pruning': prune}


def verify_output(output, parent, fd):
    named_parent = directory(output.parent)
    try:
        require(signature(os.fstat(named_parent))[:2] == signature(os.fstat(parent))[:2],
                'output parent directory replaced')
        current = os.stat(output.name, dir_fd=named_parent, follow_symlinks=False)
        require(stat.S_ISDIR(current.st_mode) and stat.S_IMODE(current.st_mode) == 0o700
                and signature(current)[:2] == signature(os.fstat(fd))[:2], 'output directory replaced')
    finally:
        os.close(named_parent)


def publish(output, report, table, inputs):
    parent = directory(output.parent)
    fd = None
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)
        fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        require(stat.S_IMODE(os.fstat(fd).st_mode) == 0o700 and not os.listdir(fd), 'output mode/contents invalid')
        payloads = {'PILOT_RECONCILIATION.json': canonical(report) + b'\n', 'PILOT_TABLE.csv': table}
        payloads['RECONCILIATION_SUCCESS'] = canonical({'kind': 'pilot_reconciliation_success',
            'row_count': 15, 'artifact_sha256': {n: sha(b) for n, b in payloads.items()}}) + b'\n'
        for name, content in payloads.items():
            verify_output(output, parent, fd)
            temporary = '.' + name + '.tmp'
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(handle, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            require(name not in os.listdir(fd), 'output target already exists')
            if name == 'RECONCILIATION_SUCCESS':
                for root in inputs:
                    root.verify()
            verify_output(output, parent, fd)
            os.rename(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        os.fsync(parent)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _collect_seed42_rows(old, new, pilot_worktree, recovery_worktree, *, recovery_commit=None,
                        operational_paths=()):
    require(str(old.path) == PILOT_ROOT, 'pilot root path differs')
    require((old.details.st_dev, old.details.st_ino) != (new.details.st_dev, new.details.st_ino),
            'source authorities must differ')
    head, changes, migration = provenance(pilot_worktree, recovery_worktree,
                                          operational_paths=operational_paths)
    origin = head if recovery_commit is None else recovery_commit
    require(type(origin) is str and COMMIT.fullmatch(origin), 'recovery origin commit invalid')
    git(recovery_worktree, 'merge-base', '--is-ancestor', origin, head)
    old_auth, new_auth = authority(old, False, PILOT_COMMIT), authority(new, True, origin)
    old_jobs = {key: job_spec(old, old_auth, key) for key in old_auth['keys']}
    new_jobs = {key: job_spec(new, new_auth, key) for key in new_auth['keys']}
    _source_compatibility(old_jobs, new_jobs, migration)
    for key, job in new_jobs.items():
        require(job[1] == old_jobs[key][1] and new_auth['protocols'][key] == old_auth['protocols'][key],
                'adaptive command/scientific contract differs: ' + key)
    rows = []
    for root, auth, jobs in ((old, old_auth, old_jobs), (new, new_auth, new_jobs)):
        admitted = [key for key in jobs if root is new or ':adaptive:' not in key]
        require(root.names('records') == {quote(k, safe='') + '.json' for k in admitted},
                'completed record membership differs (old adaptive prohibited)')
        rows.extend(completed(root, auth, key, jobs[key][0], jobs[key][2]) for key in admitted)
    require(sum(r['pruning'] is not None for r in rows if r['source_root'] == str(old.path)) == 11,
            'old pruning membership must be eleven pruned and one unpruned')
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), METHODS.index(r['method'])))
    require([(r['dataset'], r['method'], r['seed']) for r in rows]
            == [(d, m, 42) for d in DATASETS for m in METHODS], 'fifteen-cell union differs')
    sources = [{'root': str(r.path), 'source_commit': a['commit'], 'authority_identity': a['identity'],
                'identity_file_sha256': a['owner_identity']['hash']}
               for r, a in ((old, old_auth), (new, new_auth))]
    report = {'kind': 'split_authority_seed42_pilot_reconciliation', 'sources': sources,
              'pilot_evidence_manifest_sha256': PILOT_EVIDENCE_SHA256,
              'git_scope': {'base_commit': BASE_COMMIT, 'recovery_head': head, 'changed_paths': changes,
                            'approved_source_migration': migration},
              'metric_formula_version': FORMULA, 'rows': rows,
              'excluded_old_adaptive': list(new_jobs), 'validation': 'ADMITTED'}
    old.verify()
    new.verify()
    require(git(recovery_worktree, 'rev-parse', 'HEAD').strip() == head, 'recovery Git HEAD changed')
    return report


def collect_seed42_rows(pilot_root, recovery_root, pilot_worktree, recovery_worktree) -> dict:
    """Validate both immutable authorities and return the report before publish."""
    require(str(Path(pilot_root)) == PILOT_ROOT, 'pilot root path differs')
    with Evidence(pilot_root) as old, Evidence(recovery_root) as new:
        report = _collect_seed42_rows(old, new, pilot_worktree, recovery_worktree)
        old.verify()
        new.verify()
        return json.loads(canonical(report))


def reconcile(pilot_root, recovery_root, output_root, pilot_worktree, recovery_worktree):
    output = Path(output_root)
    require(output.is_absolute() and output.name not in ('', '.', '..'), 'output must be an absolute fresh path')
    parent = directory(output.parent)
    try:
        require(output.name not in os.listdir(parent), 'output already exists')
    finally:
        os.close(parent)
    require(str(Path(pilot_root)) == PILOT_ROOT, 'pilot root path differs')
    for root in (Path(pilot_root), Path(recovery_root)):
        require(output != root and root not in output.parents, 'output cannot be inside an input root')
    with Evidence(pilot_root) as old, Evidence(recovery_root) as new:
        report = _collect_seed42_rows(old, new, pilot_worktree, recovery_worktree)
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('dataset', 'method', 'seed', *SCALAR_METRICS, 'source_root', 'source_commit', 'record_sha256'))
        for row in report['rows']:
            writer.writerow((row['dataset'], row['method'], row['seed'],
                *(canonical(row['metrics'][n]).decode() for n in SCALAR_METRICS),
                row['source_root'], row['source_commit'], row['record_sha256']))
        publish(output, report, stream.getvalue().encode(), (old, new))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-root', 'recovery-root', 'output-root', 'pilot-worktree', 'recovery-worktree'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        reconcile(args.pilot_root, args.recovery_root, args.output_root,
                  args.pilot_worktree, args.recovery_worktree)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as error:
        print('reconciliation rejected: ' + str(error), file=sys.stderr)
        return 1
    print('RECONCILIATION_SUCCESS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
