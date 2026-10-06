"""Regression for canonical ER-ACE transaction identity; no training."""
import copy
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import adaptive_consolidation_audit
import runner
import three_dataset_formal_audit as audit
from three_dataset_formal_registry import protocol_for, spec_for_key


class ErAceProtocolTests(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {
            'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
            'VFCL_FORMAL_DATASET': 'cifar100', 'VFCL_GPU_COUNT': '2'})
        environment.start()
        self.addCleanup(environment.stop)
        self.protocol = protocol_for(spec_for_key('cifar100:er_ace:42'))
        self.source = {'test_provenance': 'canonical-protocol-unit-test'}

    def producer(self, protocol):
        values = dict(protocol['base_options'], seed=protocol['seed'])
        with mock.patch.object(adaptive_consolidation_audit,
                               '_formal_source_provenance', return_value=self.source):
            return runner._checkpoint_protocol(SimpleNamespace(**values))

    def transaction(self):
        tasks = audit._task_classes(self.protocol)
        classes = [c for task in tasks for c in task]
        final = len(tasks) - 1
        contents, records = {}, []
        plan = audit._formal_access_plan(self.protocol)
        for split, prefix, phase in plan:
            access = dict(event='first_iteration', loader_key=repr((prefix, tuple(classes))),
                          split=split, phase=phase, event_idx=final, task_id=final,
                          timeline_step=f'event_{final}_CIL', classes=classes)
            records.append(access)
            contents[f'formal_access/{split}.consumed.json'] = json.dumps(
                dict(schema_version=1, status='consumed', access=access)).encode()
        cache = dict(batch_count=1, sample_count=1, batches=[dict(
            input_sha256='a'*64, label_sha256='b'*64, input_shape=[1, 3, 32, 32],
            label_shape=[1], input_dtype='torch.float32', label_dtype='torch.int64')])
        calibration = dict(split=plan[-2][0], phase=plan[-2][2], cache=cache)
        identity = dict(protocol=self.producer(self.protocol), source_provenance=self.source,
                        task_classes={str(i): list(task) for i, task in enumerate(tasks)})
        shared = dict(schema_version=1, transaction_sha256='c'*64,
                      identity=identity, freeze={'test': 'same-freeze'})
        consuming = dict(shared, status='consuming',
                         cache_identity={'test': cache, 'calibration': calibration})
        markers = {'FORMAL_EVALUATION_PENDING.json': dict(shared, status='pending'),
                   'FORMAL_EVALUATION_CONSUMING.json': consuming,
                   'FORMAL_EVALUATION_COMPLETE.json': dict(consuming, status='complete',
                                                         calibration_cache_identity=calibration)}
        return contents, records, markers

    def validate(self, contents, records, markers):
        merged = dict(contents, **{k: json.dumps(v).encode() for k,v in markers.items()})
        audit._validate_formal_access_artifacts(merged, self.protocol, records, self.source)

    def test_erace_matches_real_producer(self):
        self.assertEqual(self.producer(self.protocol),
                         audit._formal_checkpoint_protocol(self.protocol, self.source))

    def test_valid_erace_transaction_passes(self):
        self.validate(*self.transaction())

    def test_missing_wrong_and_extra_fields_rejected(self):
        for field in ('num_classes', 'er_ace_buffer_size', 'er_ace_batch'):
            for mutation in ('missing', 'wrong'):
                with self.subTest(field=field, mutation=mutation):
                    contents, records, markers = self.transaction()
                    for marker in markers.values():
                        value = marker['identity']['protocol']
                        if mutation == 'missing':
                            value.pop(field, None)
                        else:
                            value[field] += 1
                            break  # identities share the same in-memory value
                    with self.assertRaises(audit._EvidenceError) as caught:
                        self.validate(contents, records, markers)
                    self.assertEqual('formal-access-transaction-invalid', caught.exception.reason)
        contents, records, markers = self.transaction()
        markers['FORMAL_EVALUATION_PENDING.json']['identity']['protocol']['extra'] = 1
        with self.assertRaises(audit._EvidenceError):
            self.validate(contents, records, markers)

    def test_canonical_missing_options_do_not_default(self):
        for field in ('num_classes', 'er_ace_buffer_size', 'er_ace_batch'):
            changed = copy.deepcopy(self.protocol)
            changed['base_options'].pop(field)
            with self.subTest(field=field), self.assertRaises(KeyError):
                audit._formal_checkpoint_protocol(changed, self.source)

    def test_other_methods_match_producer_without_erace_keys(self):
        for method in ('finetune', 'lwf', 'der_pp', 'er'):
            protocol = protocol_for(spec_for_key(f'cifar100:{method}:42'))
            expected = audit._formal_checkpoint_protocol(protocol, self.source)
            with self.subTest(method=method):
                self.assertEqual(self.producer(protocol), expected)
                self.assertFalse({'num_classes', 'er_ace_buffer_size', 'er_ace_batch'} & expected.keys())


if __name__ == '__main__':
    unittest.main()
