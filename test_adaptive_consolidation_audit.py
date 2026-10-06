import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

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
from adaptive_consolidation_audit import (
    SOURCE_FILES,
    _canonical_validation_evidence,
    atomic_write_new_json,
    audit_adaptive_checkpoint,
    prepare_adaptive_run_provenance,
    recover_atomic_write_temps,
    _replay_manifest,
)
from head_consolidation import freeze_state, hash_top_state
from models import TopModel
from runner import (
    _load_resume_checkpoint,
    _sanitize_and_finalize_adaptive,
    run_experiment,
)


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(repr(tuple(value.shape)).encode('ascii'))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class AdaptiveAuditTests(unittest.TestCase):
    def test_fixed_endpoints_strict_audit_with_one_persisted_branch(self):
        import adaptive_head_consolidation as head
        import adaptive_tinyimagenet_heldout as heldout

        for variant in ('full', 'bias'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                checkpoint, spec = self._fixture(tmp)
                payload = torch.load(checkpoint, weights_only=False)
                bundle = payload['cl_state']['adaptive_audit_bundle']
                pre = TopModel(3, 4)
                pre.load_state_dict(bundle['pre_state'], strict=True)
                selected = bundle[f'{variant}_state']
                empty = freeze_state({})
                candidates = FrozenAdaptiveCandidates(
                    pre_head_sha256=hash_top_state(pre),
                    full_state=(freeze_state(selected) if variant == 'full' else empty),
                    bias_state=(freeze_state(selected) if variant == 'bias' else empty),
                    full_head_sha256=(hash_top_state(selected) if variant == 'full'
                                      else hash_top_state(empty)),
                    bias_head_sha256=(hash_top_state(selected) if variant == 'bias'
                                      else hash_top_state(empty)),
                    full_audit={}, bias_audit={}, ordered_classes=(0, 1, 2, 3),
                )
                validation_x = bundle['validation_embeddings']
                validation_y = bundle['validation_labels']
                with heldout._fixed_branch_runtime(variant):
                    full_p, bias_p = head.adaptive_candidate_log_probabilities(
                        pre, candidates, validation_x
                    )
                    gate = head.solve_global_mixture_weight(
                        full_p, bias_p, validation_y, candidates.ordered_classes
                    )
                    installed = head.install_and_reload_verify(
                        pre, candidates, gate
                    )
                    result = AdaptiveConsolidationResult(
                        pre_head_sha256=candidates.pre_head_sha256,
                        candidate_hashes={
                            'pre': candidates.pre_head_sha256,
                            'full': candidates.full_head_sha256,
                            'bias': candidates.bias_head_sha256,
                        },
                        candidate_configs={
                            'full': head.FULL_BRANCH_CONFIG,
                            'bias': head.BIAS_BRANCH_CONFIG,
                        },
                        gate=gate,
                        validation_manifest=bundle['result']['validation_manifest'],
                        ordered_classes=candidates.ordered_classes,
                        task_id=1, task_boundary='event_1_CIL',
                    )
                    bundle.update({
                        'result': result.to_dict(),
                        'full_state': dict(candidates.full_state),
                        'bias_state': dict(candidates.bias_state),
                        'installed_state': dict(installed.state_dict()),
                        'diagnostics': build_adaptive_diagnostics(
                            pre, installed, candidates,
                            bundle['replay_embeddings'], validation_x,
                            validation_y, bundle['task_classes'], gate,
                        ),
                    })
                    payload['trainer_state']['top_model'] = installed.state_dict()
                    torch.save(payload, checkpoint)
                    for snapshot in (Path(tmp) / 'adaptive_snapshots').glob('*.pt'):
                        value = torch.load(snapshot, weights_only=False)
                        value['trainer_state']['top_model'] = installed.state_dict()
                        torch.save(value, snapshot)
                    evidence = audit_adaptive_checkpoint(tmp, spec)
                self.assertEqual(evidence['status'], 'ADAPTIVE_STATE_FROZEN')
                self.assertEqual(dict(bundle[
                    'bias_state' if variant == 'full' else 'full_state'
                ]), {})
                self.assertEqual(installed._adaptive_full_weight.numel(), 0)

    def test_adaptive_radapt_is_rejected_before_provenance_or_data_access(self):
        args = SimpleNamespace(
            cl_method='proto_evolve_radapt', ul_method='none', data='toy',
            head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )
        with mock.patch('runner.initialize_experiment_rng') as initialize, \
                mock.patch('runner._prepare_adaptive_provenance') as provenance, \
                mock.patch('runner.VFLDataset') as dataset:
            with self.assertRaisesRegex(ValueError, 'final-test data'):
                run_experiment(args)
        initialize.assert_not_called()
        provenance.assert_not_called()
        dataset.assert_not_called()

    def test_result_boundary_accepts_actual_interleaved_event_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, _ = self._fixture(tmp)
            payload = torch.load(checkpoint, weights_only=False)
            record = payload['cl_state']['adaptive_audit_bundle']['result']
            record['task_boundary'] = 'event_2_CIL'
            parsed = AdaptiveConsolidationResult.from_dict(record)
            self.assertEqual(parsed.task_id, 1)
            self.assertEqual(parsed.task_boundary, 'event_2_CIL')

    def test_source_identity_covers_every_selectable_ul_implementation(self):
        ul_sources = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / 'ul_methods').glob('*.py')
        }
        self.assertTrue(ul_sources)
        self.assertLessEqual(ul_sources, set(SOURCE_FILES))
        self.assertLessEqual({
            'cl_methods/__init__.py',
            'cl_methods/proto_evolve.py',
            'cl_methods/proto_evolve_radapt.py',
        }, set(SOURCE_FILES))

    def test_adaptive_resume_rejects_symlink_before_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / 'checkpoints'
            checkpoints.mkdir()
            target = Path(tmp) / 'outside.pt'
            torch.save({'event_idx': 0}, target)
            os.symlink(target, checkpoints / 'resume_latest.pt')
            args = SimpleNamespace(resume_run_dir=tmp, output_dir=tmp)
            with mock.patch('runner.torch.load') as unrestricted:
                with self.assertRaises(RuntimeError):
                    _load_resume_checkpoint(
                        args, object(), object(), object(), object(), object()
                    )
            unrestricted.assert_not_called()

    def test_adaptive_ul_purges_raw_replay_before_finalization(self):
        class Method:
            head_raw_replay = {
                0: torch.tensor([[0.0]]), 1: torch.tensor([[1.0]])
            }
            head_task_classes = {0: [0], 1: [1]}

            def finalize_adaptive_head_after_sanitize(self):
                self.finalized = (
                    set(self.head_raw_replay), self.head_task_classes.copy()
                )

        method = Method()
        _sanitize_and_finalize_adaptive(method, object(), [0], True)
        self.assertEqual(set(method.head_raw_replay), {1})
        self.assertEqual(method.head_task_classes, {1: [1]})
        self.assertEqual(method.finalized, ({1}, {1: [1]}))

    def test_retained_validation_manifest_binds_canonical_row_order(self):
        identities = [0, 1, 10, 11, 20, 21]
        manifest = {
            'dataset': 'toy', 'seed': 42, 'per_class': 2,
            'by_class': {'0': [0, 1], '1': [10, 11], '2': [20, 21]},
            'ordered_indices': identities,
            'sha256': hashlib.sha256(json.dumps(
                identities, separators=(',', ':')
            ).encode()).hexdigest(),
        }
        filtered, order = _canonical_validation_evidence(
            manifest, torch.tensor([2, 0, 1, 0, 2, 1]), (1, 2)
        )
        self.assertEqual(order.tolist(), [2, 5, 0, 4])
        self.assertEqual(filtered['by_class'], {'1': [10, 11], '2': [20, 21]})
        self.assertEqual(filtered['ordered_indices'], [10, 11, 20, 21])
        self.assertEqual(filtered['sha256'], hashlib.sha256(
            b'[10,11,20,21]'
        ).hexdigest())

    def _fixture(self, run_dir):
        torch.manual_seed(17)
        pre = TopModel(3, 4)
        full = TopModel(3, 4)
        full.load_state_dict(pre.state_dict(), strict=True)
        bias = TopModel(3, 4)
        bias.load_state_dict(pre.state_dict(), strict=True)
        with torch.no_grad():
            full.classifier.weight.add_(0.2)
            bias.classifier.bias.add_(torch.tensor([0.3, -0.1, 0.2, -0.2]))
        candidates = FrozenAdaptiveCandidates(
            pre_head_sha256=hash_top_state(pre),
            full_state=freeze_state(full.state_dict()),
            bias_state=freeze_state(bias.state_dict()),
            full_head_sha256=hash_top_state(full),
            bias_head_sha256=hash_top_state(bias),
            full_audit={}, bias_audit={}, ordered_classes=(0, 1, 2, 3),
        )
        validation_x = torch.tensor([
            [1.0, 0.0, 0.0], [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0], [0.2, 0.0, 0.8],
            [-1.0, 0.0, 0.0], [-0.8, -0.2, 0.0],
        ])
        validation_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        full_p, bias_p = adaptive_candidate_log_probabilities(
            pre, candidates, validation_x
        )
        gate = solve_global_mixture_weight(
            full_p, bias_p, validation_y, candidates.ordered_classes
        )
        installed = install_and_reload_verify(pre, candidates, gate)
        replay = {
            class_id: validation_x[validation_y == class_id][:1].clone()
            for class_id in candidates.ordered_classes
        }
        task_classes = {0: [0, 1], 1: [2, 3]}
        diagnostics = build_adaptive_diagnostics(
            pre, installed, candidates, replay, validation_x, validation_y,
            task_classes, gate,
        )
        identities = list(range(8))
        manifest = {
            'dataset': 'toy', 'seed': 42, 'per_class': 2,
            'by_class': {str(i): identities[2 * i:2 * i + 2] for i in range(4)},
            'ordered_indices': identities,
            'sha256': hashlib.sha256(json.dumps(
                identities, separators=(',', ':')
            ).encode()).hexdigest(),
        }
        result = AdaptiveConsolidationResult(
            pre_head_sha256=hash_top_state(pre),
            candidate_hashes={
                'pre': hash_top_state(pre), 'full': hash_top_state(full),
                'bias': hash_top_state(bias),
            },
            candidate_configs={'full': FULL_BRANCH_CONFIG, 'bias': BIAS_BRANCH_CONFIG},
            gate=gate, validation_manifest=manifest,
            ordered_classes=(0, 1, 2, 3), task_id=1,
            task_boundary='event_1_CIL',
        )
        replay_manifest = _replay_manifest(replay, replay)
        bundle = {
            'method_version': ADAPTIVE_METHOD_VERSION,
            'result': result.to_dict(),
            'pre_state': dict(pre.state_dict()),
            'full_state': dict(full.state_dict()),
            'bias_state': dict(bias.state_dict()),
            'installed_state': dict(installed.state_dict()),
            'replay_embeddings': replay,
            'replay_raw': replay,
            'replay_manifest': replay_manifest,
            'validation_embeddings': validation_x,
            'validation_labels': validation_y,
            'validation_embeddings_sha256': _tensor_sha256(validation_x),
            'validation_labels_sha256': _tensor_sha256(validation_y),
            'task_classes': task_classes,
            'diagnostics': diagnostics,
            'event_order': [
                'candidates_frozen', 'validation_iterated', 'gate_solved',
                'state_installed', 'diagnostics_computed',
            ],
        }
        checkpoint = Path(run_dir) / 'adaptive_final.pt'
        source_files = {
            name: _sha256(SOURCE_ROOT / name)
            for name in SOURCE_FILES
        }
        commit = subprocess.check_output(
            ['git', '-C', str(SOURCE_ROOT), 'rev-parse', 'HEAD'], text=True
        ).strip()
        core = {
            'source_version': ADAPTIVE_METHOD_VERSION,
            'source_commit': commit,
            'source_sha256': source_files,
        }
        planned = Path(run_dir) / 'ADAPTIVE_RUN_PLANNED.json'
        launched = Path(run_dir) / 'ADAPTIVE_RUN_LAUNCHED.json'
        atomic_write_new_json(planned, {'record': 'planned', **core})
        atomic_write_new_json(launched, {'record': 'launched', **core})
        protocol = {'num_parties': 0, 'num_tasks': 2}
        trainer_state = {'bottoms': [], 'top_model': installed.state_dict()}
        torch.save({
            'schema_version': 1,
            'kind': 'adaptive_final_checkpoint',
            'provenance': {
                'source_version': ADAPTIVE_METHOD_VERSION,
                'source_commit': commit,
                'planned_sha256': _sha256(planned),
                'launched_sha256': _sha256(launched),
            },
            'protocol': protocol,
            'top_model': {'input_dim': 3, 'num_classes': 4, 'cosine': False},
            'trainer_state': trainer_state,
            'cl_state': {
                'adaptive_audit_bundle': bundle,
                'head_raw_replay': replay,
                'head_task_classes': task_classes,
            },
        }, checkpoint)
        snapshots = Path(run_dir) / 'adaptive_snapshots'
        snapshots.mkdir()
        for event_idx, task_id in enumerate((0, 1)):
            torch.save({
                'schema_version': 1,
                'kind': 'adaptive_deferred_cil_snapshot',
                'event_idx': event_idx,
                'task_id': task_id,
                'introduced_classes': task_classes[task_id],
                'seen_task_classes': {
                    key: task_classes[key] for key in range(task_id + 1)
                },
                'trainer_state': trainer_state,
                'cl_state': {},
                'protocol': protocol,
                'top_model': {
                    'input_dim': 3, 'num_classes': 4, 'cosine': False,
                },
            }, snapshots / f'event_{event_idx}_CIL.pt')
        spec = {
            **core,
            'checkpoint': checkpoint.name,
            'planned_sha256': _sha256(planned),
            'launched_sha256': _sha256(launched),
            'data_flow': {
                'candidates_frozen_before_validation': True,
                'validation_before_freeze': False,
                'test_before_install': False,
                'test_used_for_diagnostics': False,
                'solver_input': 'class_balanced_validation_nll',
            },
        }
        return checkpoint, spec

    def test_atomic_json_is_exclusive_nofollow_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evidence.json'
            atomic_write_new_json(path, {'b': 2, 'a': 1})
            self.assertEqual(json.loads(path.read_text()), {'a': 1, 'b': 2})
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                atomic_write_new_json(path, {'a': 2})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])
            link = Path(tmp) / 'link.json'
            os.symlink(path, link)
            with self.assertRaises((ValueError, FileExistsError, OSError)):
                atomic_write_new_json(link, {'a': 3})
            nested = Path(tmp) / 'nested'
            nested.mkdir()
            with self.assertRaises(ValueError):
                atomic_write_new_json(nested / '..' / 'escaped.json', {'a': 4})

    def test_atomic_evidence_temp_recovery_promotes_complete_and_discards_torn(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / '.record.json.0123456789abcdef.tmp'
            complete.write_bytes(b'complete')
            complete.chmod(0o444)
            torn = Path(tmp) / '.torn.json.fedcba9876543210.tmp'
            torn.write_bytes(b'torn')
            recover_atomic_write_temps(tmp)
            self.assertEqual((Path(tmp) / 'record.json').read_bytes(), b'complete')
            self.assertFalse(complete.exists())
            self.assertFalse(torn.exists())

    def test_audit_recomputes_complete_checkpoint_without_writing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, spec = self._fixture(tmp)
            before = sorted(
                (p.relative_to(tmp).as_posix(), _sha256(p))
                for p in Path(tmp).rglob('*') if p.is_file()
            )
            evidence = audit_adaptive_checkpoint(tmp, spec)
            after = sorted(
                (p.relative_to(tmp).as_posix(), _sha256(p))
                for p in Path(tmp).rglob('*') if p.is_file()
            )
            self.assertEqual(before, after)
            self.assertEqual(evidence['status'], 'ADAPTIVE_STATE_FROZEN')
            self.assertEqual(evidence['checkpoint']['sha256'], _sha256(checkpoint))
            self.assertEqual(evidence['candidate_hashes'], evidence['result']['candidate_hashes'])
            self.assertEqual(evidence['ordered_classes'], [0, 1, 2, 3])
            self.assertEqual(evidence['solver'], evidence['result']['gate'])
            self.assertEqual(evidence['diagnostics']['g'], evidence['solver']['g'])
            self.assertEqual(set(evidence['probes']), {'seed', 'full', 'bias', 'mixed'})
            self.assertTrue(evidence['strict_fresh_reload'])

    def test_audit_rejects_missing_stage_snapshots_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, spec = self._fixture(tmp)
            for snapshot in (Path(tmp) / 'adaptive_snapshots').iterdir():
                snapshot.unlink()
            (Path(tmp) / 'adaptive_snapshots').rmdir()
            with self.assertRaisesRegex(ValueError, 'stage snapshots'):
                audit_adaptive_checkpoint(tmp, spec)

    def test_audit_rejects_forged_ul_boundary_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, spec = self._fixture(tmp)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            result = payload['cl_state']['adaptive_audit_bundle']['result']
            result['task_boundary'] = 'event_2_UL'
            torch.save(payload, checkpoint)
            boundary_dir = Path(tmp) / 'checkpoints'
            boundary_dir.mkdir()
            torch.save({
                'event_idx': 2,
                'step': 'event_2_UL',
                'trainer_state': payload['trainer_state'],
                'cl_state': {'adaptive_audit_bundle': {'result': result}},
                'forged': True,
            }, boundary_dir / 'event_2_UL.pt')
            with self.assertRaisesRegex(ValueError, 'UL boundary schema'):
                audit_adaptive_checkpoint(tmp, spec)

    def test_audit_rejects_wrong_hash_duplicate_temp_post_provenance_and_reload(self):
        mutations = (
            'wrong_hash', 'duplicate', 'temp', 'post_provenance',
            'forged_backdate', 'data_flow', 'forged_installed',
            'forgotten_replay', 'bottom_count', 'reload',
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                checkpoint, spec = self._fixture(tmp)
                if mutation == 'wrong_hash':
                    spec['source_sha256']['runner.py'] = '0' * 64
                elif mutation == 'duplicate':
                    payload = torch.load(checkpoint, weights_only=False)
                    replay_manifest = payload['cl_state']['adaptive_audit_bundle']['replay_manifest']
                    replay_manifest['ordered_sample_ids'][1] = replay_manifest['ordered_sample_ids'][0]
                    torch.save(payload, checkpoint)
                elif mutation == 'temp':
                    (Path(tmp) / '.unfinished.tmp').write_text('partial')
                elif mutation == 'post_provenance':
                    checkpoint_time = checkpoint.stat().st_mtime_ns
                    planned = Path(tmp) / 'ADAPTIVE_RUN_PLANNED.json'
                    os.utime(planned, ns=(checkpoint_time + 1_000_000, checkpoint_time + 1_000_000))
                elif mutation == 'forged_backdate':
                    time.sleep(0.01)
                    checkpoint_time = checkpoint.stat().st_mtime_ns
                    planned = Path(tmp) / 'ADAPTIVE_RUN_PLANNED.json'
                    content = planned.read_bytes()
                    planned.chmod(0o644)
                    planned.write_bytes(content)
                    planned.chmod(0o444)
                    os.utime(planned, ns=(checkpoint_time - 1, checkpoint_time - 1))
                elif mutation == 'data_flow':
                    payload = torch.load(checkpoint, weights_only=False)
                    payload['cl_state']['adaptive_audit_bundle']['event_order'].insert(
                        3, 'test_iterated'
                    )
                    torch.save(payload, checkpoint)
                elif mutation == 'forged_installed':
                    payload = torch.load(checkpoint, weights_only=False)
                    bundle = payload['cl_state']['adaptive_audit_bundle']
                    for state in (bundle['installed_state'],
                                  payload['trainer_state']['top_model']):
                        state['_adaptive_gate'] = torch.tensor(
                            0.123, dtype=torch.float64
                        )
                    torch.save(payload, checkpoint)
                elif mutation == 'forgotten_replay':
                    payload = torch.load(checkpoint, weights_only=False)
                    payload['cl_state']['head_raw_replay'] = {
                        key: value.clone()
                        for key, value in
                        payload['cl_state']['head_raw_replay'].items()
                    }
                    payload['cl_state']['head_raw_replay'][99] = torch.tensor(
                        [[99.0, 0.0, 0.0]]
                    )
                    payload['cl_state']['head_task_classes'][99] = [99]
                    torch.save(payload, checkpoint)
                elif mutation == 'bottom_count':
                    payload = torch.load(checkpoint, weights_only=False)
                    payload['trainer_state']['bottoms'].append({})
                    torch.save(payload, checkpoint)
                else:
                    payload = torch.load(checkpoint, weights_only=False)
                    payload['trainer_state']['top_model'] = {
                        key: value.clone()
                        for key, value in payload['trainer_state']['top_model'].items()
                    }
                    payload['trainer_state']['top_model']['classifier.bias'][0] += 1
                    torch.save(payload, checkpoint)
                with self.assertRaises((ValueError, RuntimeError)):
                    audit_adaptive_checkpoint(tmp, spec)

    def test_post_freeze_test_records_do_not_invalidate_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, spec = self._fixture(tmp)
            evidence = audit_adaptive_checkpoint(tmp, spec)
            atomic_write_new_json(
                Path(tmp) / 'ADAPTIVE_STATE_FROZEN.json', evidence
            )
            (Path(tmp) / 'data_flow_audit.jsonl').write_text(
                json.dumps({'event': 'first_iteration', 'split': 'test'}) + '\n'
            )
            self.assertEqual(audit_adaptive_checkpoint(tmp, spec), evidence)

    def test_audit_rejects_symlinked_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / 'real'
            real.mkdir()
            _, spec = self._fixture(real)
            link = Path(tmp) / 'link'
            os.symlink(real, link)
            with self.assertRaises((ValueError, OSError)):
                audit_adaptive_checkpoint(link, spec)

    def test_provenance_cannot_be_backfilled_after_any_checkpoint(self):
        for relative in (
                'adaptive_final.pt',
                'ADAPTIVE_STATE_FROZEN.json',
                'adaptive_snapshots/event_0_CIL.pt',
                'checkpoints/event_0_CIL.pt'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b'existing')
                with self.assertRaises(ValueError):
                    prepare_adaptive_run_provenance(tmp)
                self.assertFalse((Path(tmp) / 'ADAPTIVE_RUN_PLANNED.json').exists())
                self.assertFalse((Path(tmp) / 'ADAPTIVE_RUN_LAUNCHED.json').exists())


if __name__ == '__main__':
    unittest.main()
