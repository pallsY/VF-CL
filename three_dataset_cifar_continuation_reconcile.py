"""Verify completed CIFAR-100 cells before a provenance-preserving continuation."""

from pathlib import Path
from urllib.parse import quote

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
import three_dataset_seed42_reconcile as legacy


_ORIGIN_METHODS = ('finetune', 'lwf', 'ewc', 'er', 'der_pp', 'er_ace')
_SEEDS = (42, 43, 44)
_ORIGIN_ROOT = Path(
    '/home/c3080/YangXiaoXiang/VF-CL/results/'
    'formal-cifar3080-dual-20g-20260928-v8'
)
_ORIGIN_COMMIT = 'a7915143129d986f4561b93b49119fd4cfcd79f6'
_ORIGIN_PLAN_SHA = 'fd6a746fc9b1c5f91bb78bd844563cfe64e80b2b2225b2b721ba6b4d82efd110'
_OLD_WORKTREE = Path(
    '/home/c3080/YangXiaoXiang/VF-CL-worktrees/formal-pipelined-audit-3080'
)
_CIFAR_DATA_ROOT = Path('/home/c3080/YangXiaoXiang/VF-CL/data')
_CIFAR_DATA_LOGICALS = (
    'cifar-100-python/meta', 'cifar-100-python/train',
    'cifar-100-python/test',
)
_CIFAR_SHA256 = {
    'cifar-100-python/meta': 'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9',
    'cifar-100-python/train': '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    'cifar-100-python/test': '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864',
}
_MEMORY_FIX_COMMIT = 'e90f895b4848c1728953802c437d45b2d7005353'
_AUDIT_FIX_COMMIT = 'f9a32fa1614d3e8994cc52d880730e0d475ad142'
_CONTINUATION_PROFILE = 'single-dataset-verified-continuation-v1'
_REVIEWED_CHANGES = frozenset({
    'adaptive_consolidation_audit.py', 'three_dataset_formal_audit.py',
    'three_dataset_cifar_continuation_reconcile.py',
    'three_dataset_formal_registry.py', 'three_dataset_formal_driver.py',
    'three_dataset_cifar_continuation_profile.py',
    'three_dataset_cifar_continuation_report.py',
    'run_three_dataset_formal_comparison.sh', 'prune_completed_runs.py',
    'three_dataset_resource_gate.py', 'vfl_trainer.py',
    'cl_methods/adagauss.py',
})


def origin_keys():
    return tuple(f'cifar100:{method}:{seed}'
                 for method in _ORIGIN_METHODS for seed in _SEEDS)


def require_origin_membership(names):
    expected = {quote(key, safe='') + '.json' for key in origin_keys()}
    legacy.require(type(names) is set and names == expected,
                   'CIFAR origin completed-record membership differs')


def inspect_origin(root_path):
    path = Path(root_path)
    legacy.require(path == _ORIGIN_ROOT, 'CIFAR origin root differs')
    with legacy.Evidence(path) as root:
        identity = root.json('FORMAL_ROOT_IDENTITY.json')
        plan = root.json('FORMAL_PLAN.json')
        census = root.json('COMPATIBILITY_CENSUS.json')
        manifest = root.json('MISSING_JOBS.json')
        binding = {'experiment_profile': 'single-dataset-full-matrix',
                   'formal_dataset': 'cifar100'}
        driver._validate_census(census)
        legacy.require(identity['source_commit'] == _ORIGIN_COMMIT
                       and identity['plan_sha256'] == _ORIGIN_PLAN_SHA
                       and legacy.digest(plan) == _ORIGIN_PLAN_SHA
                       and identity['missing_jobs_sha256'] == legacy.digest(manifest)
                       and identity['registry_sha256'] == registry.registry_sha256()
                       and identity['root_dev'] == root.details.st_dev
                       and identity['root_inode'] == root.details.st_ino,
                       'CIFAR origin authority differs')
        legacy.require(plan['missing_jobs'] == plan['formal_cells']
                       and len(plan['formal_cells']) == 42,
                       'CIFAR origin planned cells differ')
        require_origin_membership(root.names('records'))
        legacy.require(root.json('FAILED_JOB')['spec_key'] == 'cifar100:gpm:42'
                       and root.json('FORMAL_STOPPED')['kind'] == 'formal_stopped',
                       'CIFAR origin terminal evidence differs')
        owner = {
            'dev': root.details.st_dev,
            'inode': root.details.st_ino,
            'ctime_ns': root.seen['FORMAL_ROOT_IDENTITY.json'][-1],
            'size': root.seen['FORMAL_ROOT_IDENTITY.json'][4],
            'hash': legacy.sha(root.read('FORMAL_ROOT_IDENTITY.json', immutable=True)),
        }
        protocols = {row['spec_key']: row['protocol_sha256']
                     for row in census['records']}
        auth = {'binding': binding, 'plan': plan, 'commit': _ORIGIN_COMMIT,
                'owner_identity': owner, 'protocols': protocols}
        rows = []
        for key in origin_keys():
            job, _normalized, tasks = legacy.job_spec(root, auth, key)
            checked = legacy.completed(root, auth, key, job, tasks)
            record_path = 'records/' + quote(key, safe='') + '.json'
            record = root.json(record_path)
            rows.append({
                **{name: checked[name] for name in
                   ('spec_key', 'dataset', 'method', 'seed', 'metrics')},
                'resource': record['resource'],
                'artifact_sha256': record['artifact_sha256'],
                'source_sha256': {name[7:]: digest for name, digest
                                  in record['artifact_sha256'].items()
                                  if name.startswith('source:')},
                'trajectory_sha256': record['trajectory_sha256'],
                'origin_root': str(root.path),
                'origin_record_path': str(root.path / record_path),
                'origin_record_sha256': checked['record_file_sha256'],
                'origin_source_commit': _ORIGIN_COMMIT,
            })
        for row in rows:
            data = {name: row['artifact_sha256']['data:' + name]
                    for name in _CIFAR_DATA_LOGICALS}
            legacy.require(data == _CIFAR_SHA256,
                           'CIFAR origin data differs')
        source_map = rows[0]['source_sha256']
        legacy.require(all(row['source_sha256'] == source_map
                           for row in rows),
                       'CIFAR origin source maps differ')
        legacy.require(legacy.git(_OLD_WORKTREE, 'rev-parse', 'HEAD').strip()
                       == _ORIGIN_COMMIT
                       and not legacy.git(_OLD_WORKTREE,
                                          'status', '--porcelain'),
                       'CIFAR old source worktree differs')
        with legacy.Evidence(_OLD_WORKTREE) as old_source:
            for logical, expected in source_map.items():
                legacy.relative(logical)
                legacy.require(old_source.read(logical, hash_only=True)
                               == expected
                               and legacy.sha(legacy.git_bytes(
                                   _OLD_WORKTREE, 'show',
                                   _ORIGIN_COMMIT + ':' + logical))
                               == expected,
                               'CIFAR origin source blob differs')
            old_source.verify()
        with legacy.Evidence(_CIFAR_DATA_ROOT) as data_root:
            for logical, expected in _CIFAR_SHA256.items():
                legacy.require(data_root.read(logical, hash_only=True)
                               == expected,
                               'CIFAR frozen data differs')
            data_root.verify()
        root.verify()
        return rows, {
            'root': str(root.path), 'source_commit': _ORIGIN_COMMIT,
            'identity_file_sha256': owner['hash'],
            'plan_sha256': _ORIGIN_PLAN_SHA,
            'data_sha256': dict(_CIFAR_SHA256),
            'protocol_sha256': protocols,
        }


def _allowed_change(path):
    return (path in _REVIEWED_CHANGES
            or (path.startswith('docs/') and path.endswith('.md'))
            or (path.startswith('test_') and path.endswith('.py')))


def validate_bundle(bundle):
    legacy.schema(bundle, ('kind', 'experiment_profile', 'dataset',
                           'registry_sha256', 'metric_formula_version',
                           'current_commit', 'sources', 'admitted',
                           'rejected', 'ambiguous', 'changed_paths',
                           'migration_sha256'), 'CIFAR continuation bundle')
    legacy.require(bundle['kind'] == 'cifar_continuation_reuse_v1'
                   and bundle['experiment_profile'] == _CONTINUATION_PROFILE
                   and bundle['dataset'] == 'cifar100'
                   and bundle['registry_sha256'] == registry.registry_sha256()
                   and bundle['metric_formula_version'] == driver.FORMULA_VERSION
                   and bundle['rejected'] == [] and bundle['ambiguous'] == [],
                   'CIFAR continuation bundle authority differs')
    legacy.require(type(bundle['sources']) is list
                   and len(bundle['sources']) == 1,
                   'CIFAR origin source count differs')
    source = bundle['sources'][0]
    legacy.schema(source, ('root', 'source_commit', 'identity_file_sha256',
                           'plan_sha256', 'data_sha256', 'protocol_sha256'),
                  'CIFAR origin source')
    legacy.require(source['root'] == str(_ORIGIN_ROOT)
                   and source['source_commit'] == _ORIGIN_COMMIT
                   and source['plan_sha256'] == _ORIGIN_PLAN_SHA
                   and source['data_sha256'] == _CIFAR_SHA256,
                   'CIFAR origin source differs')
    driver._hash_string(source['identity_file_sha256'],
                        'CIFAR origin identity hash')
    legacy.require(type(bundle['current_commit']) is str
                   and driver._COMMIT.fullmatch(bundle['current_commit'])
                   and type(bundle['migration_sha256']) is str,
                   'CIFAR continuation commit or migration type differs')
    driver._hash_string(bundle['migration_sha256'], 'migration digest')
    current_protocols = {
        driver.spec_key(spec): driver._protocol_sha256(spec)
        for spec in registry.formal_specs()
    }
    legacy.require(len(current_protocols) == 42
                   and source['protocol_sha256'] == current_protocols,
                   'CIFAR continuation protocols differ')
    rows = bundle['admitted']
    legacy.require(type(rows) is list and len(rows) == 18
                   and [row['spec_key'] for row in rows] == list(origin_keys()),
                   'CIFAR reuse membership differs')
    for row in rows:
        legacy.schema(row, ('spec_key', 'dataset', 'method', 'seed',
                            'metrics', 'resource', 'artifact_sha256',
                            'source_sha256', 'trajectory_sha256',
                            'origin_root', 'origin_record_path',
                            'origin_record_sha256', 'origin_source_commit',
                            'compatibility', 'migration_sha256'),
                      'CIFAR reuse row')
        spec = registry.FormalSpec(row['dataset'], row['method'], row['seed'])
        legacy.require(driver.spec_key(spec) == row['spec_key']
                       and row['origin_root'] == str(_ORIGIN_ROOT)
                       and row['origin_source_commit'] == _ORIGIN_COMMIT
                       and row['origin_record_path'] == str(
                           _ORIGIN_ROOT / 'records' /
                           (quote(row['spec_key'], safe='') + '.json'))
                       and row['compatibility'] == 'ADMITTED'
                       and row['migration_sha256'] == bundle['migration_sha256'],
                       'CIFAR reuse row provenance differs')
        driver._hash_string(row['origin_record_sha256'], 'origin record hash')
        driver._hash_string(row['trajectory_sha256'], 'origin trajectory hash')
        legacy.hashes(row['artifact_sha256'], 'origin artifacts')
        legacy.hashes(row['source_sha256'], 'origin source')
        legacy.require({name[7:]: digest for name, digest
                        in row['artifact_sha256'].items()
                        if name.startswith('source:')}
                       == row['source_sha256'],
                       'origin source projection differs')
        driver._validate_metrics(row['metrics'], spec, list)
        driver._validate_record_resource(row['resource'])
    legacy.require(type(bundle['changed_paths']) is list
                   and bundle['changed_paths'] == sorted(set(bundle['changed_paths']))
                   and all(_allowed_change(path)
                           for path in bundle['changed_paths']),
                   'CIFAR source scope differs')
    expected = legacy.digest({
        'origin_commit': _ORIGIN_COMMIT,
        'current_commit': bundle['current_commit'],
        'changed_paths': bundle['changed_paths'],
        'protocol_sha256': source['protocol_sha256'],
        'data_sha256': source['data_sha256'],
    })
    legacy.require(bundle['migration_sha256'] == expected,
                   'CIFAR migration digest differs')


def build_bundle(origin_root, worktree):
    rows, source = inspect_origin(origin_root)
    worktree = Path(worktree)
    head = legacy.git(worktree, 'rev-parse', 'HEAD').strip()
    legacy.git(worktree, 'merge-base', '--is-ancestor', _ORIGIN_COMMIT, head)
    changed = sorted(set(legacy.git(
        worktree, 'diff', '--name-only', _ORIGIN_COMMIT, head
    ).splitlines()))
    legacy.require(all(_allowed_change(path) for path in changed),
                   'CIFAR source changes exceed reviewed scope')
    old_blob = legacy.git(worktree, 'rev-parse',
                          _MEMORY_FIX_COMMIT + ':adaptive_consolidation_audit.py')
    new_blob = legacy.git(worktree, 'rev-parse',
                          head + ':adaptive_consolidation_audit.py')
    legacy.require(old_blob == new_blob,
                   'CIFAR deferred evaluator changed after memory proof')
    old_audit_blob = legacy.git(
        worktree, 'rev-parse',
        _AUDIT_FIX_COMMIT + ':three_dataset_formal_audit.py')
    new_audit_blob = legacy.git(
        worktree, 'rev-parse',
        head + ':three_dataset_formal_audit.py')
    legacy.require(old_audit_blob == new_audit_blob,
                   'CIFAR strict auditor changed after memory proof')
    specs = registry.formal_specs()
    legacy.require(len(specs) == 42, 'CIFAR protocol cardinality differs')
    current_protocols = {driver.spec_key(spec): driver._protocol_sha256(spec)
                         for spec in specs}
    legacy.require(current_protocols == source['protocol_sha256']
                   and source['data_sha256'] == _CIFAR_SHA256,
                   'CIFAR scientific protocol or data differs')
    migration = legacy.digest({
        'origin_commit': _ORIGIN_COMMIT, 'current_commit': head,
        'changed_paths': changed, 'protocol_sha256': current_protocols,
        'data_sha256': _CIFAR_SHA256,
    })
    bundle = {
        'kind': 'cifar_continuation_reuse_v1',
        'experiment_profile': _CONTINUATION_PROFILE,
        'dataset': 'cifar100',
        'registry_sha256': registry.registry_sha256(),
        'metric_formula_version': driver.FORMULA_VERSION,
        'current_commit': head, 'sources': [source],
        'admitted': [{**row, 'compatibility': 'ADMITTED',
                      'migration_sha256': migration} for row in rows],
        'rejected': [], 'ambiguous': [], 'changed_paths': changed,
        'migration_sha256': migration,
    }
    validate_bundle(bundle)
    return bundle


_BUNDLE_NAME = 'CIFAR_CONTINUATION_REUSE.json'
_AUDIT_NAME = 'CIFAR_CONTINUATION_REUSE_AUDIT.json'
_SUCCESS_NAME = 'CIFAR_CONTINUATION_REUSE_SUCCESS'


def _file_bytes(value):
    return legacy.canonical(value) + b'\n'


def _audit_payload(bundle):
    return {
        'kind': 'cifar_continuation_reuse_audit_v1',
        'validation': 'ADMITTED',
        'bundle_sha256': legacy.sha(_file_bytes(bundle)),
        'admitted_count': 18,
        'migration_sha256': bundle['migration_sha256'],
        'rejected': [], 'ambiguous': [],
    }


def _success_payload(bundle, audit):
    return {
        'kind': 'cifar_continuation_reuse_success_v1',
        'row_count': 18,
        'artifact_sha256': {
            _BUNDLE_NAME: legacy.sha(_file_bytes(bundle)),
            _AUDIT_NAME: legacy.sha(_file_bytes(audit)),
        },
    }


def publish_bundle(bundle, output):
    validate_bundle(bundle)
    output = Path(output)
    worktree = Path(__file__).resolve().parent
    legacy.require(output.is_absolute()
                   and output not in (_ORIGIN_ROOT, _CIFAR_DATA_ROOT,
                                      _OLD_WORKTREE, worktree)
                   and all(parent not in output.parents for parent in (
                       _ORIGIN_ROOT, _CIFAR_DATA_ROOT,
                       _OLD_WORKTREE, worktree,
                   )), 'reuse output is inside protected input')
    legacy.require(legacy.git(worktree, 'rev-parse', 'HEAD').strip()
                   == bundle['current_commit'],
                   'reuse publication code commit changed')
    legacy.require(not legacy.git(worktree, 'status', '--porcelain'),
                   'reuse publication worktree is dirty')
    legacy.require(build_bundle(_ORIGIN_ROOT, worktree) == bundle,
                   'origin or migration changed before publication')
    driver._mkdir_exclusive(output.parent, output.name)
    audit = _audit_payload(bundle)
    driver.install_json_exclusive(output / _BUNDLE_NAME, bundle)
    driver.install_json_exclusive(output / _AUDIT_NAME, audit)
    driver.install_json_exclusive(output / _SUCCESS_NAME,
                                  _success_payload(bundle, audit))


def load_bundle(output):
    with legacy.Evidence(Path(output)) as pinned:
        legacy.require(pinned.names() == {
            _BUNDLE_NAME, _AUDIT_NAME, _SUCCESS_NAME,
        }, 'reuse output membership differs')
        bundle = pinned.json(_BUNDLE_NAME)
        audit = pinned.json(_AUDIT_NAME)
        success = pinned.json(_SUCCESS_NAME)
        validate_bundle(bundle)
        legacy.require(legacy.git(Path(__file__).resolve().parent,
                                  'rev-parse', 'HEAD').strip()
                       == bundle['current_commit'],
                       'reuse reader commit differs')
        actual = sorted(set(legacy.git(
            Path(__file__).resolve().parent, 'diff', '--name-only',
            _ORIGIN_COMMIT, bundle['current_commit'],
        ).splitlines()))
        legacy.require(bundle['changed_paths'] == actual
                       and all(_allowed_change(path) for path in actual),
                       'CIFAR reuse Git scope differs from actual history')
        legacy.require(audit == _audit_payload(bundle)
                       and success == _success_payload(bundle, audit),
                       'reuse publication attestation differs')
        pinned.verify()
    with legacy.Evidence(_ORIGIN_ROOT) as origin:
        source = bundle['sources'][0]
        legacy.require(legacy.sha(origin.read(
            'FORMAL_ROOT_IDENTITY.json', immutable=True))
            == source['identity_file_sha256'],
            'reuse origin identity changed')
        for row in bundle['admitted']:
            relative = 'records/' + quote(row['spec_key'], safe='') + '.json'
            legacy.require(legacy.sha(origin.read(relative, immutable=True))
                           == row['origin_record_sha256'],
                           'reuse origin record changed')
        origin.verify()
    return bundle
