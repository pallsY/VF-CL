"""Provenance-preserving CIFAR-100 continuation tables."""

from pathlib import Path
import statistics

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
import three_dataset_seed42_reconcile as legacy
from three_dataset_cifar_continuation_profile import reuse_census
from three_dataset_cifar_continuation_reconcile import origin_keys


def combined_rows(root, plan, census, bundle):
    driver._validate_plan_shape(plan, census)
    if not driver._exact_equal(census, reuse_census(bundle)):
        raise ValueError('continuation census differs from bundle')
    root = Path(root)
    reused = {row['spec_key']: row for row in bundle['admitted']}
    expected_files = {registry.safe_spec_name(spec) + '.json'
                      for spec in registry.formal_specs()
                      if driver.spec_key(spec) in plan['missing_jobs']}
    with legacy.Evidence(root / 'records') as records:
        legacy.require(records.names() == expected_files,
                       'new continuation records differ')
        records.verify()
    rows = []
    for spec in registry.formal_specs():
        key = driver.spec_key(spec)
        if key in reused:
            source = reused[key]
            provenance = {
                'origin': 'reused',
                'origin_root': source['origin_root'],
                'origin_source_commit': source['origin_source_commit'],
                'origin_record_sha256': source['origin_record_sha256'],
            }
        else:
            source = driver._installed_completed_record(root, spec, plan)
            if source is None:
                raise ValueError('new continuation record is missing')
            provenance = {
                'origin': 'new', 'origin_root': str(root),
                'origin_source_commit': source['source_commit'],
                'origin_record_sha256': driver._sha256_bytes(
                    driver._canonical_json(source) + b'\n'),
            }
        rows.append({
            **{name: source[name] for name in
               ('spec_key', 'dataset', 'method', 'seed', 'metrics', 'resource')},
            **provenance, 'audit_status': 'ADMITTED',
        })
    return rows


def render_continuation_tables(rows):
    specs = registry.formal_specs()
    if len(rows) != 42 or [row['spec_key'] for row in rows] != [
            driver.spec_key(spec) for spec in specs]:
        raise ValueError('continuation report membership differs')
    metrics = ('aa_final', 'bwt', 'taskil_final')
    identity = ('spec_key', 'dataset', 'method', 'seed')
    provenance = ('origin', 'origin_root', 'origin_source_commit',
                  'origin_record_sha256', 'audit_status')
    expected_fields = set(identity) | {'metrics', 'resource'} | set(provenance)
    reused_keys = set(origin_keys())
    for row, spec in zip(rows, specs):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError('continuation report row schema differs')
        driver._validate_metrics(row['metrics'], spec, list)
        driver._validate_record_resource(row['resource'])
        if row['origin'] not in ('reused', 'new') or row['audit_status'] != 'ADMITTED':
            raise ValueError('continuation row origin differs')
        if (row['spec_key'] in reused_keys) != (row['origin'] == 'reused'):
            raise ValueError('continuation origin membership differs')
        if not Path(row['origin_root']).is_absolute():
            raise ValueError('continuation origin root is not absolute')
        if (type(row['origin_source_commit']) is not str
                or driver._COMMIT.fullmatch(row['origin_source_commit']) is None):
            raise ValueError('continuation origin commit differs')
        driver._hash_string(row['origin_record_sha256'], 'origin record hash')
    if {row['origin_source_commit'] for row in rows
            if row['origin'] == 'reused'} != {
            'a7915143129d986f4561b93b49119fd4cfcd79f6'}:
        raise ValueError('historical producer commit differs')
    new_commits = {row['origin_source_commit'] for row in rows
                   if row['origin'] == 'new'}
    if len(new_commits) != 1:
        raise ValueError('new producer commits differ')
    per_run = [[row[name] for name in identity]
               + [row['metrics'][name] for name in metrics]
               + [row[name] for name in provenance] for row in rows]
    summary = []
    for method in registry.FULL_MATRIX_METHODS:
        group = [row for row in rows if row['method'] == method]
        if [row['seed'] for row in group] != [42, 43, 44]:
            raise ValueError('continuation seed group differs')
        summary.append([method, 3] + [value for name in metrics for value in (
            statistics.fmean(row['metrics'][name] for row in group),
            statistics.stdev(row['metrics'][name] for row in group),
        )])
    resources = tuple(sorted(driver._RESOURCE_KEYS))
    resource_rows = [[row[name] for name in identity] + [
        driver._canonical_json(row['resource'][name]).decode()
        if type(row['resource'][name]) is dict else row['resource'][name]
        for name in resources] + [row[name] for name in provenance]
        for row in rows]
    tables = {
        'CIFAR_CONTINUATION_PER_RUN.csv': driver._csv_bytes(
            identity + metrics + provenance, per_run),
        'CIFAR_CONTINUATION_TABLE.csv': driver._csv_bytes(
            ('method', 'seed_count') + tuple(name + suffix
                for name in metrics for suffix in ('_mean', '_std')), summary),
        'CIFAR_CONTINUATION_RESOURCE.csv': driver._csv_bytes(
            identity + resources + provenance, resource_rows),
    }
    audit = {
        'kind': 'cifar_continuation_report_audit_v1',
        'experiment_profile': registry.CONTINUATION_PROFILE,
        'registry_sha256': registry.registry_sha256(),
        'metric_formula_version': driver.FORMULA_VERSION,
        'input_rows_sha256': driver._digest(rows),
        'reused_count': 18, 'new_count': 24,
        'origin_commits': {'reused':
                           'a7915143129d986f4561b93b49119fd4cfcd79f6',
                           'new': next(iter(new_commits))},
        'table_sha256': {name: driver._sha256_bytes(content)
                         for name, content in tables.items()},
    }
    tables['CIFAR_CONTINUATION_AUDIT.json'] = (
        driver._canonical_json(audit) + b'\n')
    return tables
