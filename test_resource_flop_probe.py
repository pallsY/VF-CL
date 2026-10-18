"""Profiler inputs are pinned without looking at formal test scores."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from resource_flop_probe import parse_task_classes, verify_profile_inputs


class FlopProbeTests(unittest.TestCase):
    def test_custom_and_uniform_task_orders(self):
        vector = {'num_classes': 4, 'num_tasks': 2,
                  'custom_tasks': '0,1|2,3'}
        image = {'num_classes': 4, 'num_tasks': 2,
                 'classes_per_task': 2, 'custom_tasks': ''}
        self.assertEqual(parse_task_classes(vector), [[0, 1], [2, 3]])
        self.assertEqual(parse_task_classes(image), [[0, 1], [2, 3]])
        with self.assertRaisesRegex(ValueError, 'task'):
            parse_task_classes(dict(vector, custom_tasks='0,1|1,2'))

    def test_source_and_snapshot_are_bound_to_frozen_job(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        job = root / 'job'
        run = job / 'run'
        snapshot = run / 'formal_snapshots/event_0_CIL.pt'
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b'frozen')
        config = {'num_classes': 4, 'num_tasks': 2,
                  'custom_tasks': '0,1|2,3', 'num_parties': 4}
        (run / 'config.json').write_text(json.dumps(config))
        (job / 'JOB_PLAN.json').write_text(json.dumps({
            'code_commit': 'a' * 40,
        }))
        (run / 'FORMAL_STATE_FROZEN.json').write_text(json.dumps({
            'identity': {'snapshots': [
                {'file': {'path': 'formal_snapshots/event_0_CIL.pt',
                          'sha256': hashlib.sha256(b'frozen').hexdigest()}}
            ]},
        }))
        source = root / 'source'
        source.mkdir()
        with mock.patch('resource_flop_probe.subprocess.check_output',
                        side_effect=['a' * 40 + '\n', '']):
            actual, path, commit = verify_profile_inputs(job, source, 1)
        self.assertEqual(actual, config)
        self.assertEqual(path, snapshot)
        self.assertEqual(commit, 'a' * 40)
        with mock.patch('resource_flop_probe.subprocess.check_output',
                        side_effect=['a' * 40 + '\n', '']):
            first, first_path, _ = verify_profile_inputs(job, source, 0)
        self.assertEqual(first, config)
        self.assertIsNone(first_path)
        snapshot.write_bytes(b'changed')
        with mock.patch('resource_flop_probe.subprocess.check_output',
                        side_effect=['a' * 40 + '\n', '']):
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                verify_profile_inputs(job, source, 1)


if __name__ == '__main__':
    unittest.main()
