import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT = Path(__file__).with_name('run_method_shard_queue.py')
METHODS = ('lwf_wa', 'proto_fedspace', 'adaptive')


class MethodShardQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'results'
        self.base.mkdir()
        self.calls = Path(self.temp.name) / 'calls.jsonl'
        self.launcher = Path(self.temp.name) / 'fake_launcher.py'
        self.launcher.write_text(textwrap.dedent(
            '''\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import signal
            import sys
            import time

            root = Path(sys.argv[-1])
            method = os.environ['VFCL_FORMAL_METHOD']
            check = len(sys.argv) == 3 and sys.argv[1] == '--check'
            with open(os.environ['FAKE_CALLS'], 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({'method': method, 'check': check}) + '\\n')
            if check:
                required = (
                    'FORMAL_REGISTRY.json',
                    'COMPATIBILITY_CENSUS.json',
                    'FORMAL_PLAN.json',
                    'MISSING_JOBS.json',
                    'FORMAL_ROOT_IDENTITY.json',
                ) if os.environ.get('REQUIRE_AUTHORITY') == '1' else ()
                if required and not all((root / name).is_file()
                                        for name in required):
                    raise SystemExit(8)
                raise SystemExit(0)
            if method == os.environ.get('FAIL_METHOD'):
                (root / 'FAILED_JOB').write_text('{}', encoding='utf-8')
                raise SystemExit(9)
            records = root / 'records'
            records.mkdir()
            for seed in (42, 43, 44):
                name = f'cifar100%3A{method}%3A{seed}.json'
                (records / name).write_text('{}', encoding='utf-8')
            if method != os.environ.get('OMIT_PHASE_METHOD'):
                (root / 'METHOD_SHARD_PHASE_SUCCESS').write_text(
                    '{}', encoding='utf-8')
            if method != os.environ.get('OMIT_SUCCESS_METHOD'):
                (root / 'METHOD_SHARD_SUCCESS').write_text(
                    '{}', encoding='utf-8')
            if method == os.environ.get('SIGNAL_AT_DRAIN_METHOD'):
                queue_pid = os.getppid()
                launcher_pid = os.getpid()
                if os.fork() == 0:
                    for _ in range(200):
                        if os.getppid() != launcher_pid:
                            os.kill(queue_pid, signal.SIGTERM)
                            break
                        time.sleep(0.01)
                    os._exit(0)
            '''
        ), encoding='utf-8')
        self.launcher.chmod(0o755)

    def run_queue(self, *methods, tag='queue-v1', extra_env=None):
        env = os.environ.copy()
        env['FAKE_CALLS'] = str(self.calls)
        env.update(extra_env or {})
        command = [
            sys.executable,
            str(SCRIPT),
            '--tag', tag,
            '--results-base', str(self.base),
            '--launcher', str(self.launcher),
            *methods,
        ]
        return subprocess.run(
            command, env=env, text=True, capture_output=True, check=False)

    def calls_for(self, *, check):
        if not self.calls.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.calls.read_text(encoding='utf-8').splitlines()
        ]
        return [row['method'] for row in rows if row['check'] is check]

    def create_complete_root(self, method, tag='queue-v1'):
        root = self.base / f'formal-method-{method}-{tag}'
        records = root / 'records'
        records.mkdir(parents=True)
        for seed in (42, 43, 44):
            name = f'cifar100%3A{method}%3A{seed}.json'
            (records / name).write_text('{}', encoding='utf-8')
        (root / 'METHOD_SHARD_PHASE_SUCCESS').write_text(
            '{}', encoding='utf-8')
        (root / 'METHOD_SHARD_SUCCESS').write_text('{}', encoding='utf-8')
        return root

    def test_successful_methods_launch_in_reviewed_order(self):
        result = self.run_queue()

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(list(METHODS), self.calls_for(check=False))
        for method in METHODS:
            root = self.base / f'formal-method-{method}-queue-v1'
            self.assertTrue((root / 'METHOD_SHARD_SUCCESS').is_file())

    def test_new_root_has_driver_authority_before_launcher_check(self):
        result = self.run_queue(
            'adaptive', extra_env={'REQUIRE_AUTHORITY': '1'})

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['adaptive'], self.calls_for(check=True))
        root = self.base / 'formal-method-adaptive-queue-v1'
        for name in (
                'FORMAL_REGISTRY.json', 'COMPATIBILITY_CENSUS.json',
                'FORMAL_PLAN.json', 'MISSING_JOBS.json',
                'FORMAL_ROOT_IDENTITY.json'):
            self.assertTrue((root / name).is_file())

    def test_failure_stops_before_next_method(self):
        result = self.run_queue(
            extra_env={'FAIL_METHOD': 'proto_fedspace'})

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            ['lwf_wa', 'proto_fedspace'], self.calls_for(check=False))
        self.assertFalse(
            (self.base / 'formal-method-adaptive-queue-v1').exists())

    def test_signal_after_child_exit_stops_before_next_method(self):
        result = self.run_queue(
            extra_env={'SIGNAL_AT_DRAIN_METHOD': 'lwf_wa'})

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual(['lwf_wa'], self.calls_for(check=False))
        self.assertFalse(
            (self.base / 'formal-method-proto_fedspace-queue-v1').exists())

    def test_terminal_launcher_failure_is_in_queue_log(self):
        result = self.run_queue(
            'adaptive', extra_env={'FAIL_METHOD': 'adaptive'})

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            'adaptive: launcher exited 9',
            (self.base / 'method-shard-queue-queue-v1.log').read_text(
                encoding='utf-8'))

    def test_subprocess_start_error_is_in_queue_log_with_command(self):
        self.launcher.write_text(
            '#!/definitely/missing/python\n', encoding='utf-8')

        result = self.run_queue('adaptive')

        self.assertNotEqual(0, result.returncode)
        log = (self.base / 'method-shard-queue-queue-v1.log').read_text(
            encoding='utf-8')
        self.assertIn('cannot start', log)
        self.assertIn(str(self.launcher), log)
        self.assertIn('No such file or directory', log)

    def test_completed_roots_are_skipped_on_restart(self):
        for method in METHODS:
            self.create_complete_root(method)

        result = self.run_queue()

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([], self.calls_for(check=False))
        self.assertIn('already complete', result.stdout)

    def test_incomplete_existing_root_fails_closed(self):
        (self.base / 'formal-method-adaptive-queue-v1').mkdir()

        result = self.run_queue('adaptive')

        self.assertNotEqual(0, result.returncode)
        self.assertEqual([], self.calls_for(check=False))
        self.assertIn('pre-existing root is not complete', result.stderr)

    def test_invalid_tag_and_non_suffix_order_are_rejected(self):
        bad_tag = self.run_queue(tag='../unsafe')
        bad_order = self.run_queue('lwf_wa', 'adaptive')

        self.assertNotEqual(0, bad_tag.returncode)
        self.assertNotEqual(0, bad_order.returncode)
        self.assertEqual([], self.calls_for(check=False))

    def test_missing_success_contract_stops_queue(self):
        result = self.run_queue(
            extra_env={'OMIT_PHASE_METHOD': 'lwf_wa'})

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(['lwf_wa'], self.calls_for(check=False))
        self.assertIn('formal success contract failed', result.stderr)

    def test_existing_failure_marker_invalidates_success_markers(self):
        root = self.create_complete_root('adaptive')
        (root / 'FAILED_JOB').write_text('{}', encoding='utf-8')

        result = self.run_queue('adaptive')

        self.assertNotEqual(0, result.returncode)
        self.assertEqual([], self.calls_for(check=False))

    def test_second_controller_for_same_tag_is_rejected(self):
        lock_path = self.base / '.method-shard-queue-queue-v1.lock'
        with lock_path.open('a+', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_queue()

        self.assertEqual(73, result.returncode)
        self.assertEqual([], self.calls_for(check=False))
        self.assertIn('queue is already running', result.stderr)


if __name__ == '__main__':
    unittest.main()
