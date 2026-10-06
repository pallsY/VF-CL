"""Full-profile reuse admission and provenance-preserving matrix reports."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from three_dataset_seed42_reconcile import Evidence, require, schema, hashes


def validate_reuse(reuse):
    require(registry.experiment_profile() == registry.FULL_MATRIX_PROFILE,
            'reuse requires full matrix profile')
    schema(reuse, ('kind', 'experiment_profile', 'registry_sha256', 'metric_formula_version',
                   'current_commit', 'sources', 'admitted', 'rejected', 'ambiguous'), 'reuse bundle')
    require(reuse['kind'] == 'full_public_matrix_reuse_v1'
            and reuse['experiment_profile'] == registry.FULL_MATRIX_PROFILE
            and reuse['registry_sha256'] == registry.registry_sha256()
            and reuse['metric_formula_version'] == driver.FORMULA_VERSION
            and reuse['current_commit'] == driver._source_commit(), 'reuse authority differs')
    require(all(type(reuse[k]) is list for k in ('sources', 'admitted', 'rejected', 'ambiguous')),
            'reuse collections invalid')
    sources = set()
    for source in reuse['sources']:
        require(type(source) is dict, 'reuse source invalid')
        driver._path(source['root'], 'reuse origin root')
        require(type(source['source_commit']) is str
                and driver._COMMIT.fullmatch(source['source_commit']), 'reuse origin commit invalid')
        sources.add((source['root'], source['source_commit']))
    specs = {driver.spec_key(s): s for s in registry.formal_specs()}
    seen = set()
    for row in reuse['admitted']:
        schema(row, ('spec_key', 'dataset', 'method', 'seed', 'metrics', 'resource',
                     'artifact_sha256', 'source_sha256', 'trajectory_sha256', 'origin_root',
                     'origin_record_path', 'origin_record_sha256', 'origin_source_commit',
                     'compatibility', 'migration_sha256'), 'reuse row')
        key = row['spec_key']
        require(type(key) is str and key in specs and key not in seen, 'reuse membership invalid')
        seen.add(key)
        spec = specs[key]
        require(row['dataset'] == spec.dataset and row['method'] == spec.method
                and type(row['seed']) is int and row['seed'] == spec.seed
                and row['compatibility'] == 'ADMITTED', 'reuse cell identity differs')
        require((row['origin_root'], row['origin_source_commit']) in sources, 'reuse origin is not audited')
        require(row['origin_record_path'] == str(Path(row['origin_root']) / 'records' /
                (registry.safe_spec_name(spec) + '.json')), 'reuse record path differs')
        for field in ('origin_record_sha256', 'trajectory_sha256', 'migration_sha256'):
            driver._hash_string(row[field], field)
        hashes(row['artifact_sha256'], 'reuse artifacts')
        hashes(row['source_sha256'], 'reuse source')
        projection = {k[7:]: v for k, v in row['artifact_sha256'].items() if k.startswith('source:')}
        require(driver._exact_equal(projection, row['source_sha256']), 'reuse source projection differs')
        driver._validate_metrics(row['metrics'], spec, list)
        driver._validate_record_resource(row['resource'])
    require([r['spec_key'] for r in reuse['admitted']] == [k for k in specs if k in seen],
            'reuse order differs from registry')


def load_reuse_bundle(path):
    """Read Task 2's immutable, canonical bundle and both audit attestations."""
    path = driver._path(path, 'reuse bundle')
    require(path.name == 'FULL_MATRIX_REUSE.json', 'reuse bundle filename differs')
    with Evidence(path.parent) as evidence:
        reuse = evidence.json(path.name)
        audit = evidence.json('FULL_MATRIX_REUSE_AUDIT.json')
        success = evidence.json('REUSE_AUDIT_SUCCESS')
        expected_audit = {'kind': 'full_public_matrix_reuse_audit_v1', 'validation': 'ADMITTED',
                          'bundle_sha256': driver._digest(reuse), 'admitted_count': len(reuse['admitted']),
                          'rejected': reuse['rejected'], 'ambiguous': reuse['ambiguous']}
        expected_success = {'kind': 'full_public_matrix_reuse_audit_success', 'row_count': len(reuse['admitted']),
                            'artifact_sha256': {name: driver._sha256_bytes(evidence.read(name, immutable=True))
                                                for name in (path.name, 'FULL_MATRIX_REUSE_AUDIT.json')}}
        require(driver._exact_equal(audit, expected_audit), 'reuse audit differs')
        require(driver._exact_equal(success, expected_success), 'reuse audit success differs')
        validate_reuse(reuse)
        evidence.verify()
        return reuse


def reuse_census(reuse):
    validate_reuse(reuse)
    census = driver.build_census({})
    admitted = {row['spec_key']: row for row in reuse['admitted']}
    for index, spec in enumerate(registry.formal_specs()):
        key = driver.spec_key(spec)
        if key not in admitted:
            continue
        row = admitted[key]
        # Preserve the legacy admission hash schema; the bundle retains the source map.
        raw = {'status': 'REUSABLE', 'reason': 'admitted', 'spec': asdict(spec),
               'protocol_sha256': driver._protocol_sha256(spec),
               'source_sha256': driver._digest(row['source_sha256']),
               'artifact_sha256': row['artifact_sha256'], 'metrics': row['metrics'],
               'metric_formula_version': driver.FORMULA_VERSION,
               'trajectory_sha256': row['trajectory_sha256']}
        census['records'][index] = {**raw, 'spec_key': key, 'admission_record_sha256': driver._digest(raw)}
    driver._validate_census(census)
    return census


def combined_rows(root: Path, plan: dict, census: dict, reuse: dict) -> list[dict]:
    driver._validate_plan_shape(plan, census)
    require(driver._exact_equal(census, reuse_census(reuse)), 'reuse census projection differs')
    reused = {row['spec_key']: row for row in reuse['admitted']}
    expected = {registry.safe_spec_name(s) + '.json' for s in registry.formal_specs()
                if driver.spec_key(s) in plan['missing_jobs']}
    if (root / 'records').exists() or (root / 'records').is_symlink():
        with Evidence(root / 'records') as evidence:
            require(evidence.names() == expected, 'current record membership differs (missing or overlap)')
    else:
        require(not expected, 'current record set is missing')
    rows = []
    for spec in registry.formal_specs():
        key = driver.spec_key(spec)
        if key in reused:
            source = reused[key]
            origin = {name: source[name] for name in ('origin_root', 'origin_source_commit', 'origin_record_sha256')}
            origin['origin'] = 'reused'
        else:
            source = driver._installed_completed_record(root, spec, plan)
            require(source is not None, 'current completed record is missing')
            origin = {'origin': 'new', 'origin_root': str(root), 'origin_source_commit': source['source_commit'],
                      'origin_record_sha256': driver._sha256_bytes(driver._canonical_json(source) + b'\n')}
        rows.append({**{name: source[name] for name in ('spec_key', 'dataset', 'method', 'seed', 'metrics', 'resource')},
                     **origin, 'audit_status': 'ADMITTED'})
    return rows


def render_full_matrix_tables(rows: list[dict]) -> dict[str, bytes]:
    require(registry.experiment_profile() == registry.FULL_MATRIX_PROFILE, 'full matrix profile required')
    specs = registry.formal_specs()
    require(type(rows) is list and len(rows) == 126, 'full matrix requires 126 rows')
    for row, spec in zip(rows, specs):
        schema(row, ('spec_key', 'dataset', 'method', 'seed', 'metrics', 'resource', 'origin',
                     'origin_root', 'origin_source_commit', 'origin_record_sha256', 'audit_status'), 'report row')
        require(row['spec_key'] == driver.spec_key(spec) and row['dataset'] == spec.dataset
                and row['method'] == spec.method and type(row['seed']) is int and row['seed'] == spec.seed,
                'report membership or registry order differs')
        require(row['origin'] in ('new', 'reused') and row['audit_status'] == 'ADMITTED', 'report origin invalid')
        driver._path(row['origin_root'], 'report origin')
        driver._hash_string(row['origin_record_sha256'], 'report record hash')
        require(type(row['origin_source_commit']) is str and driver._COMMIT.fullmatch(row['origin_source_commit']),
                'report origin commit invalid')
        driver._validate_metrics(row['metrics'], spec, list)
        driver._validate_record_resource(row['resource'])
    metrics = ('aa_final', 'bwt', 'taskil_final')
    provenance = ('origin', 'origin_root', 'origin_source_commit', 'origin_record_sha256', 'audit_status')
    identity = ('spec_key', 'dataset', 'method', 'seed')
    per_run = [[row[k] for k in identity] + [row['metrics'][k] for k in metrics] +
               [row[k] for k in provenance] for row in rows]
    summary = []
    for dataset in registry.DATASETS:
        for method in registry.FULL_MATRIX_METHODS:
            group = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            summary.append([dataset, method, 3] + [value for k in metrics for value in
                           (statistics.fmean(r['metrics'][k] for r in group), statistics.stdev(r['metrics'][k] for r in group))])
    macro = [[method, 3] + [statistics.fmean(r[3 + 2 * i] for r in summary if r[1] == method)
                           for i in range(len(metrics))] for method in registry.FULL_MATRIX_METHODS]
    resources = tuple(sorted(driver._RESOURCE_KEYS))
    resource_rows = [[row[k] for k in identity] +
                     [driver._canonical_json(row['resource'][k]).decode() if type(row['resource'][k]) is dict
                      else row['resource'][k] for k in resources] + [row[k] for k in provenance] for row in rows]
    tables = {
        'FULL_MATRIX_PER_RUN.csv': driver._csv_bytes(identity + metrics + provenance, per_run),
        'FULL_MATRIX_TABLE.csv': driver._csv_bytes(('dataset', 'method', 'seed_count') +
            tuple(k + suffix for k in metrics for suffix in ('_mean', '_std')), summary),
        'FULL_MATRIX_MACRO.csv': driver._csv_bytes(('method', 'dataset_count') + tuple(k + '_mean' for k in metrics), macro),
        'FULL_MATRIX_RESOURCE_PRIVACY.csv': driver._csv_bytes(identity + resources + provenance, resource_rows),
    }
    audit = {'kind': 'full_public_matrix_report_audit_v1', 'experiment_profile': registry.FULL_MATRIX_PROFILE,
             'registry_sha256': registry.registry_sha256(), 'metric_formula_version': driver.FORMULA_VERSION,
             'input_sha256': {'rows': driver._digest(rows)},
             'membership': [{k: row[k] for k in identity + provenance} for row in rows],
             'row_counts': {'per_run': 126, 'dataset_method': 42, 'macro': 14, 'resource_privacy': 126},
             'table_sha256': {name: driver._sha256_bytes(data) for name, data in tables.items()}}
    tables['FULL_MATRIX_AUDIT.json'] = driver._canonical_json(audit) + b'\n'
    return tables
