import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import three_dataset_formal_driver as driver
import three_dataset_formal_registry as registry
from test_three_dataset_formal_driver import (
    _all_completed_records, _completed_record,
)
from three_dataset_cifar_continuation_profile import reuse_census
from three_dataset_cifar_continuation_reconcile import origin_keys
from three_dataset_cifar_continuation_report import (
    combined_rows, render_continuation_tables,
)


class CifarContinuationReportTests(unittest.TestCase):
    @unittest.skipUnless(Path(
        '/home/c3080/YangXiaoXiang/VF-CL/results/'
        'cifar18-origin-proof-20260930-v1/CIFAR_CONTINUATION_REUSE.json'
    ).is_file(), 'real origin proof is required')
    def test_combines_only_exact_twenty_four_new_records(self):
        proof = Path(
            '/home/c3080/YangXiaoXiang/VF-CL/results/'
            'cifar18-origin-proof-20260930-v1/CIFAR_CONTINUATION_REUSE.json'
        )
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE':
                    'single-dataset-verified-continuation-v1',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            bundle = json.loads(proof.read_text(encoding='utf-8'))
            census = reuse_census(bundle)
            plan = driver.build_plan(census)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = root / 'records'
                records.mkdir()
                created = {}
                for spec in registry.formal_specs():
                    key = driver.spec_key(spec)
                    if key not in plan['missing_jobs']:
                        continue
                    (records / (registry.safe_spec_name(spec) + '.json')).write_bytes(b'')
                    created[key] = _completed_record(
                        spec, .5, source_commit='d' * 40,
                        plan_sha256=driver._digest(plan))

                def installed(_root, spec, installed_plan):
                    self.assertEqual(installed_plan, plan)
                    return created[driver.spec_key(spec)]

                with mock.patch.object(driver, '_installed_completed_record',
                                       side_effect=installed):
                    rows = combined_rows(root, plan, census, bundle)
                    self.assertEqual(len(rows), 42)
                    self.assertEqual(sum(row['origin'] == 'reused'
                                         for row in rows), 18)
                    self.assertEqual(rows[18]['spec_key'], 'cifar100:gpm:42')
                    self.assertEqual(rows[18]['origin'], 'new')
                    (records / 'unexpected.json').write_bytes(b'')
                    with self.assertRaises(ValueError):
                        combined_rows(root, plan, census, bundle)

    def test_exact_forty_two_rows_keep_origin_and_statistics(self):
        with mock.patch.dict(os.environ, {
                'VFCL_EXPERIMENT_PROFILE':
                    'single-dataset-verified-continuation-v1',
                'VFCL_FORMAL_DATASET': 'cifar100'}):
            old_keys = set(origin_keys())
            rows = []
            for record in _all_completed_records():
                old = record['spec_key'] in old_keys
                rows.append({
                    **{name: record[name] for name in
                       ('spec_key', 'dataset', 'method', 'seed',
                        'metrics', 'resource')},
                    'origin': 'reused' if old else 'new',
                    'origin_root': '/origin' if old else '/continuation',
                    'origin_source_commit': (
                        'a7915143129d986f4561b93b49119fd4cfcd79f6'
                        if old else 'd' * 40),
                    'origin_record_sha256': driver._digest(record),
                    'audit_status': 'ADMITTED',
                })
            tables = render_continuation_tables(rows)
            self.assertEqual(set(tables), {
                'CIFAR_CONTINUATION_PER_RUN.csv',
                'CIFAR_CONTINUATION_TABLE.csv',
                'CIFAR_CONTINUATION_RESOURCE.csv',
                'CIFAR_CONTINUATION_AUDIT.json',
            })
            per_run = list(csv.DictReader(io.StringIO(
                tables['CIFAR_CONTINUATION_PER_RUN.csv'].decode())))
            self.assertEqual(len(per_run), 42)
            self.assertEqual(sum(row['origin'] == 'reused'
                                 for row in per_run), 18)
            summary = list(csv.DictReader(io.StringIO(
                tables['CIFAR_CONTINUATION_TABLE.csv'].decode())))
            self.assertEqual(len(summary), 14)
            finetune = next(row for row in summary
                            if row['method'] == 'finetune')
            self.assertAlmostEqual(float(finetune['aa_final_mean']), .3)
            self.assertAlmostEqual(float(finetune['aa_final_std']), .1)
            duplicate = copy.deepcopy(rows)
            duplicate[1] = copy.deepcopy(duplicate[0])
            wrong_origin = copy.deepcopy(rows)
            wrong_origin[0]['origin'] = 'new'
            for mutation in (rows[:-1], duplicate, wrong_origin):
                with self.assertRaises(ValueError):
                    render_continuation_tables(mutation)
