import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import frozen_ablation_gpu_admission as admission


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.anchor = None
        patcher = patch.object(admission, 'anchored_source', create=True,
                               side_effect=lambda root: self.anchor)
        self.anchor_mock = patcher.start()
        self.addCleanup(patcher.stop)
        paths = patch.object(admission, 'ALLOWED_ROOTS', (Path(tempfile.gettempdir()),), create=True)
        paths.start()
        self.addCleanup(paths.stop)
        head = patch.object(admission, 'current_commit', return_value='c' * 40, create=True)
        self.head_mock = head.start()
        self.addCleanup(head.stop)

    def test_paths_must_be_in_independent_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(admission, 'ALLOWED_ROOTS', (root / 'allowed',)):
                admission.independent_path(root / 'allowed/new/evidence.json')
                with self.assertRaises(ValueError):
                    admission.independent_path(root / 'other/evidence.json')
                with self.assertRaises(ValueError):
                    admission.write(root / 'other/evidence.json', {})

    def test_all_test_authorizations_and_test_audit_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            access = output / 'run/formal_access'
            access.mkdir(parents=True)
            audit = output / 'run/data_flow_audit.jsonl'
            audit.write_text('{"split":"train"}\n')
            for marker in ('test.authorized.json', 'test.pending.json', 'test.consumed.json',
                           'subdir/test.authorized.json', 'test.authorized.bin'):
                path = access / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                with self.assertRaises(ValueError):
                    admission.check_no_test(output)
                path.unlink()
            audit.write_text('{"split":"train"}\n{"split":"test"}\n')
            with self.assertRaises(ValueError):
                admission.check_no_test(output)
            audit.unlink()
            with self.assertRaises(ValueError):
                admission.check_no_test(output)

    @unittest.skipIf(os.name == 'nt', 'symlink semantics verified on Linux')
    def test_empty_formal_access_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'run').mkdir()
            (root / 'empty').mkdir()
            (root / 'run/data_flow_audit.jsonl').write_text('{"split":"train"}\n')
            (root / 'run/formal_access').symlink_to(root / 'empty', target_is_directory=True)
            with self.assertRaises(ValueError):
                admission.check_no_test(root)

    def test_comparator_binds_orchestration_to_current_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_host(root / 'reference', 'reference')
            self.build_host(root / 'candidate', 'candidate')
            self.head_mock.return_value = 'd' * 40
            with self.assertRaises(ValueError):
                admission.compare(root / 'reference', root / 'candidate', root / 'rejected.json')

    def test_run_rejects_changed_anchor_before_any_gpu_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.anchor_mock.side_effect = ValueError('anchored source changed')
            with patch.object(admission, 'query_gpu') as gpu:
                with self.assertRaisesRegex(ValueError, 'anchored source'):
                    admission.run('base', 'reference', admission.GPUS['reference'][0],
                                  Path('/unused'), Path(temporary) / 'new-attempt')
                gpu.assert_not_called()

    def test_copied_audit_and_formal_access_inventory_cannot_be_omitted_or_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_host(root / 'reference', 'reference')
            self.build_host(root / 'candidate', 'candidate')
            run_dir = root / 'candidate/base/smoke/run'
            audit = run_dir / 'data_flow_audit.jsonl'
            original = audit.read_text()
            for change in ('missing', 'changed', 'inventory'):
                if change == 'missing':
                    audit.unlink()
                elif change == 'changed':
                    audit.write_text('{"split":"train","extra":true}\n')
                else:
                    (run_dir / 'formal_access/train.extra.json').write_text('{}')
                with self.assertRaises(ValueError):
                    admission.compare(root / 'reference', root / 'candidate', root / (change + '.json'))
                audit.write_text(original)

    def build_host(self, root, role):
        source = dict(seed=42, device='cuda:1', num_tasks=10, bic_enabled=1,
                      party_kd_enabled=1, dep_tracking_enabled=1,
                      expected_party_kd_variant='uniform')
        uuid, name = admission.GPUS[role]
        for profile in admission.PROFILES:
            directory = root / profile
            output = directory / 'smoke'
            output.mkdir(parents=True)
            (output / 'run/formal_access').mkdir(parents=True)
            (output / 'run/data_flow_audit.jsonl').write_text('{"split":"train"}\n')
            admission.write(directory / 'SOURCE_CONFIG.json', source)
            if self.anchor is None:
                self.anchor = dict(source_config_sha256=admission.sha(directory / 'SOURCE_CONFIG.json'),
                                   anchor_sha256='a' * 64, final_gate_sha256='b' * 64,
                                   original_tree_sha256='c' * 64)
            config = admission.gpu_config(source, profile, output)
            admission.write(output / 'SMOKE_CONFIG.json', config)
            no_test = dict(schema_version=1, status='no_test_access',
                           smoke_output_dir=str(output), test_accessed=False,
                           stopped_after_two_tasks=True, test_marker_present=False,
                           formal_results_present=False,
                           smoke_config_sha256=admission.sha(output / 'SMOKE_CONFIG.json'))
            admission.write(output / 'NO_TEST_ACCESS.json', no_test)
            orders = [dict(task=t, loaders=[dict(loader_key='train', epoch_orders=[[1, 2, 3]])])
                      for t in (0, 1)]
            fim = [dict(task=t, tensors=[dict(name='party0/weight', shape=[2],
                                            dtype='torch.float32', sha256='f' * 64)])
                   for t in (0, 1)]
            states = [dict(task=t, model_sha256=('b' if t == 1 and profile == 'off' else 'a') * 64,
                           memory_mask_sha256='a' * 64, rng_sha256='a' * 64,
                           fim_masks_sha256='a' * 64,
                           fim_tensor_values_sha256=admission._hash_metadata([fim[t]]),
                           sample_order_sha256=admission.smoke.sample_order_sha256(orders[t]['loaders']))
                      for t in (0, 1)]
            record = dict(dataset='cifar100', mode='base' if profile == 'no_partykd' else profile,
                          code_commit=admission.smoke.BASE_COMMIT if profile in ('base', 'no_partykd')
                          else admission.smoke.PATCH_COMMIT,
                          smoke_output_dir=str(output), orchestration_commit='c' * 40,
                          smoke_script_sha256=admission.sha(admission.smoke.__file__),
                          source_config_sha256=admission.sha(directory / 'SOURCE_CONFIG.json'),
                          smoke_config_sha256=admission.sha(output / 'SMOKE_CONFIG.json'),
                          no_test_access_sha256=admission.sha(output / 'NO_TEST_ACCESS.json'),
                          runtime=dict(host=role, torch='2.5.1+cu124', cuda_runtime='12.4',
                                       python='3.11.15', numpy='x', pillow='x', scikit_learn='x',
                                       torchvision='x'), test_accessed=False,
                          stopped_after_two_tasks=True, train_per_class=64,
                          states=states, sample_orders=orders, fim_tensor_values=fim,
                          losses=[dict(task=t, loss=1.0, accuracy=0.0) for t in (0, 1)],
                          gradient_steps=[dict(task=t, sha256='a' * 64) for t in (0, 1)],
                          fim_scores=[dict(task=t, values_sha256='a' * 64,
                                           threshold_sha256='a' * 64) for t in (0, 1)],
                          applied_freeze=[dict(task=0, frozen_elements=0),
                                          dict(task=1, frozen_elements=0 if profile == 'off' else 1)])
            admission.write(output / 'SMOKE_RESULT.json', record)
            gate = [dict(elapsed_seconds=t, uuid=uuid, name=name, free_mib=8192,
                         total_mib=10000, utilization_percent=99, driver_version='different-allowed')
                    for t in range(121)]
            launch = dict(gate[-1])
            admission.write(directory / 'GPU_RESOURCE_SAMPLES.json',
                            [dict(phase='gate', **row) for row in gate] +
                            [dict(phase='launch_recheck', **launch), dict(phase='training', **launch)])
            admission.write(directory / 'GPU_ADMISSION.json', dict(
                schema_version=1, profile=profile, role=role, hostname=role,
                gpu_uuid=uuid, gpu_name=name, compute_capability=[8, 6], cudnn_version=90100,
                allocator_budget_mib=2048, allocator_peak_allocated_bytes=512,
                allocator_peak_reserved_bytes=1024, harness_sha256=admission.sha(admission.__file__),
                gpu_config_override={'device': 'cuda:0'}, gate=gate, launch_recheck=launch,
                source_config_file_sha256=admission.sha(directory / 'SOURCE_CONFIG.json'),
                source_provenance=self.anchor,
                no_test_inventory=dict(data_flow_audit_sha256=admission.sha(output / 'run/data_flow_audit.jsonl'),
                                       formal_access_files=[]),
                config=config, smoke_result_sha256=admission.sha(output / 'SMOKE_RESULT.json')))

    def test_full_comparator_accepts_exact_evidence_and_preserves_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_host(root / 'reference', 'reference')
            self.build_host(root / 'candidate', 'candidate')
            result = admission.compare(root / 'reference', root / 'candidate', root / 'pass.json')
            self.assertEqual(result['status'], 'admitted')
            artifact = root / 'candidate/no_partykd/smoke/SMOKE_RESULT.json'
            original = artifact.read_text()
            meta_path = root / 'candidate/no_partykd/GPU_ADMISSION.json'
            original_meta = json.loads(meta_path.read_text())
            for field in ('mode', 'code_commit', 'states', 'gradient_steps', 'fim_tensor_values',
                          'sample_orders', 'test_accessed'):
                record = json.loads(original)
                record[field] = True if field == 'test_accessed' else []
                artifact.write_text(json.dumps(record))
                meta = dict(original_meta, smoke_result_sha256=admission.sha(artifact))
                meta_path.write_text(json.dumps(meta))
                report = root / (field + '.json')
                with self.assertRaises(ValueError):
                    admission.compare(root / 'reference', root / 'candidate', report)
                self.assertEqual(json.loads(report.read_text())['status'], 'rejected')
            artifact.write_text(original)
            meta_path.write_text(json.dumps(original_meta))
            config_path = root / 'candidate/no_partykd/smoke/SMOKE_CONFIG.json'
            config = json.loads(config_path.read_text())
            config['bic_enabled'] = 0
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                admission.compare(root / 'reference', root / 'candidate', root / 'bad-config.json')

    def test_identically_rehashed_changed_source_on_both_hosts_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_host(root / 'reference', 'reference')
            self.build_host(root / 'candidate', 'candidate')
            for host in ('reference', 'candidate'):
                for profile in admission.PROFILES:
                    directory = root / host / profile
                    source_path = directory / 'SOURCE_CONFIG.json'
                    source = json.loads(source_path.read_text())
                    source['bic_enabled'] = 0
                    source_path.write_text(json.dumps(source))
                    output = directory / 'smoke'
                    config = admission.gpu_config(source, profile, output)
                    config_path = output / 'SMOKE_CONFIG.json'
                    config_path.write_text(json.dumps(config))
                    proof_path = output / 'NO_TEST_ACCESS.json'
                    proof = json.loads(proof_path.read_text())
                    proof['smoke_config_sha256'] = admission.sha(config_path)
                    proof_path.write_text(json.dumps(proof))
                    record_path = output / 'SMOKE_RESULT.json'
                    record = json.loads(record_path.read_text())
                    record.update(source_config_sha256=admission.sha(source_path),
                                  smoke_config_sha256=admission.sha(config_path),
                                  no_test_access_sha256=admission.sha(proof_path))
                    record_path.write_text(json.dumps(record))
                    meta_path = directory / 'GPU_ADMISSION.json'
                    meta = json.loads(meta_path.read_text())
                    meta['source_config_file_sha256'] = admission.sha(source_path)
                    meta['source_provenance']['source_config_sha256'] = admission.sha(source_path)
                    meta['config'] = config
                    meta['smoke_result_sha256'] = admission.sha(record_path)
                    meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'anchored source'):
                admission.compare(root / 'reference', root / 'candidate', root / 'rejected.json')
            self.assertEqual(self.anchor_mock.call_count, 1)

    def test_profiles_preserve_schedule_and_no_partykd_only_disables_three_fields(self):
        source = dict(seed=42, device='cuda:1', num_tasks=10, bic_enabled=1,
                      party_kd_enabled=1, dep_tracking_enabled=1,
                      expected_party_kd_variant='uniform')
        base = admission.gpu_config(source, 'base', Path('/tmp/run'))
        no_kd = admission.gpu_config(source, 'no_partykd', Path('/tmp/run'))
        self.assertEqual({k for k in base if base[k] != no_kd[k]},
                         {'party_kd_enabled', 'dep_tracking_enabled', 'expected_party_kd_variant'})
        self.assertEqual(base['device'], 'cuda:0')
        self.assertEqual(base['num_tasks'], 10)
        self.assertEqual(base['bic_enabled'], 1)
        with self.assertRaises(ValueError):
            admission.gpu_config(source, 'unknown', Path('/tmp/run'))

    def test_resource_gate_requires_unbroken_120_seconds_and_launch_recheck(self):
        rows = [dict(elapsed_seconds=i, uuid='GPU-ref', free_mib=3584,
                     name=admission.GPUS['reference'][1],
                     total_mib=10000, utilization_percent=30) for i in range(121)]
        admission.validate_gate(rows, 'reference', 'GPU-ref')
        for changed in (rows[:-1], rows[:60] + [dict(rows[60], free_mib=3583)] + rows[61:],
                        rows[:60] + rows[62:]):
            with self.assertRaises(ValueError):
                admission.validate_gate(changed, 'reference', 'GPU-ref')
        with self.assertRaises(ValueError):
            admission.validate_gate(rows, 'candidate', 'GPU-ref')

    def test_cross_host_compares_every_numeric_evidence_field(self):
        fields = admission.NUMERIC_FIELDS
        left = {name: [{'task': 0, 'evidence': 'same'}] for name in fields}
        admission.compare_numeric(left, copy.deepcopy(left))
        for field in fields:
            right = copy.deepcopy(left)
            right[field][0]['evidence'] = 'different'
            with self.assertRaisesRegex(ValueError, field):
                admission.compare_numeric(left, right)
            right = copy.deepcopy(left)
            del right[field]
            with self.assertRaises(ValueError):
                admission.compare_numeric(left, right)


class SourceAnchorTests(unittest.TestCase):
    def test_independent_anchor_final_gate_and_original_tree_are_all_verified(self):
        for changed in ('none', 'anchor', 'gate', 'tree'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'full/cifar100/adaptive/42/run/config.json'
                source.parent.mkdir(parents=True)
                source.write_text('{"seed":42}\n')
                audit = root / 'audit'
                audit.mkdir()
                gate_path = audit / 'gate.json'
                gate_path.write_text(json.dumps({'original_tree_sha256': {
                    'cifar100/adaptive/42': admission.manifest.original_tree_sha256(source.parent.parent)}}))
                anchor_path = audit / 'ADAPTIVE_SAMPLE_ONLY_FINAL_ANCHOR.json'
                anchor_path.write_text(json.dumps(dict(producer_commit=admission.smoke.BASE_COMMIT,
                    basic_passed=9, verified_original_trees=9,
                    final_gate=dict(relative_path='audit/gate.json', sha256=admission.sha(gate_path)))))
                pinned = admission.sha(anchor_path)
                if changed != 'none':
                    target = {'anchor': anchor_path, 'gate': gate_path, 'tree': source}[changed]
                    target.write_text(target.read_text() + ' ')
                with patch.object(admission.manifest, 'SOURCE_ANCHOR_SHA', pinned):
                    if changed == 'none':
                        self.assertEqual(admission.anchored_source(root)['source_config_sha256'],
                                         admission.sha(source))
                    else:
                        with self.assertRaisesRegex(ValueError, 'anchored source'):
                            admission.anchored_source(root)


if __name__ == '__main__':
    unittest.main()
