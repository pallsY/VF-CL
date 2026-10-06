#!/usr/bin/env python3
import argparse
import fcntl
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile


METHODS = ('lwf_wa', 'proto_fedspace', 'adaptive')
SEEDS = (42, 43, 44)
PROFILE = 'single-method-formal-v1'
DATASET = 'cifar100'
REVIEWED_PYTHON = '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'
WORKTREE = Path(__file__).resolve().parent
DEFAULT_RESULTS_BASE = Path('/home/c3080/YangXiaoXiang/VF-CL/results')
DEFAULT_LAUNCHER = WORKTREE / 'run_three_dataset_formal_comparison.sh'
TAG_PATTERN = re.compile(r'[a-z0-9][a-z0-9._-]*\Z')
FAILURE_MARKERS = ('FAILED_JOB', 'FORMAL_STOPPED')
SUCCESS_MARKERS = ('METHOD_SHARD_PHASE_SUCCESS', 'METHOD_SHARD_SUCCESS')


class QueueError(RuntimeError):
    pass


def validate_request(tag, values):
    if not TAG_PATTERN.fullmatch(tag):
        raise QueueError('tag contains unsafe characters')
    methods = tuple(values) or METHODS
    if methods not in tuple(METHODS[index:] for index in range(len(METHODS))):
        raise QueueError(
            'methods must be a non-empty suffix of: ' + ', '.join(METHODS))
    return methods


def expected_records(method):
    return {
        f'{DATASET}%3A{method}%3A{seed}.json'
        for seed in SEEDS
    }


def root_complete(root, method):
    if root.is_symlink() or not root.is_dir():
        return False
    if any((root / marker).exists() for marker in FAILURE_MARKERS):
        return False
    if not all((root / marker).is_file() for marker in SUCCESS_MARKERS):
        return False
    records = root / 'records'
    if records.is_symlink() or not records.is_dir():
        return False
    actual = {
        item.name
        for item in records.iterdir()
        if item.is_file() and not item.is_symlink() and item.suffix == '.json'
    }
    return actual == expected_records(method)


def emit(log, message, *, error=False):
    line = f'method queue: {message}\n'
    target = sys.stderr if error else sys.stdout
    target.write(line)
    target.flush()
    log.write(line)
    log.flush()


def run_logged(command, env, log):
    argv = [str(item) for item in command]
    try:
        process = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except OSError as error:
        raise QueueError(
            f'cannot start {shlex.join(argv)}: {error}') from error
    previous = {}
    interrupted = None

    def forward(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if process.poll() is None:
            process.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, forward)
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if interrupted is not None:
        raise QueueError(
            f'interrupted by {signal.Signals(interrupted).name}')
    return code


def launch_method(method, tag, results_base, launcher, log):
    root = results_base / f'formal-method-{method}-{tag}'
    if os.path.lexists(root):
        if root_complete(root, method):
            emit(log, f'{method}: already complete; skipping')
            return
        raise QueueError(
            f'{method}: pre-existing root is not complete: {root}')
    env = os.environ.copy()
    env.update({
        'VFCL_EXPERIMENT_PROFILE': PROFILE,
        'VFCL_FORMAL_DATASET': DATASET,
        'VFCL_FORMAL_METHOD': method,
        'VFCL_GPU_COUNT': '2',
        'VFCL_PYTHON': REVIEWED_PYTHON,
    })
    declarations = None
    driver = WORKTREE / 'three_dataset_formal_driver.py'
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', suffix='.json',
                prefix='method-shard-declarations-', delete=False) as stream:
            declarations = Path(stream.name)
            stream.write('{}\n')
        emit(log, f'{method}: census {root}')
        code = run_logged((
            REVIEWED_PYTHON, driver, 'census', '--root', root,
            '--declarations', declarations,
        ), env, log)
        if code:
            raise QueueError(f'{method}: driver census exited {code}')
        emit(log, f'{method}: plan {root}')
        code = run_logged((
            REVIEWED_PYTHON, driver, 'plan', '--root', root,
        ), env, log)
        if code:
            raise QueueError(f'{method}: driver plan exited {code}')
    finally:
        if declarations is not None:
            declarations.unlink(missing_ok=True)
    emit(log, f'{method}: preflight {root}')
    code = run_logged((launcher, '--check', root), env, log)
    if code:
        raise QueueError(f'{method}: launcher preflight exited {code}')
    emit(log, f'{method}: starting')
    code = run_logged((launcher, root), env, log)
    if code:
        raise QueueError(f'{method}: launcher exited {code}')
    if not root_complete(root, method):
        raise QueueError(f'{method}: formal success contract failed')
    emit(log, f'{method}: complete')


def run_queue(tag, methods, results_base, launcher):
    if os.path.lexists(results_base):
        if results_base.is_symlink() or not results_base.is_dir():
            raise QueueError('results base must be a regular directory')
    else:
        results_base.mkdir(parents=True)
    lock_path = results_base / f'.method-shard-queue-{tag}.lock'
    log_path = results_base / f'method-shard-queue-{tag}.log'
    with lock_path.open('a+', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('method queue: queue is already running', file=sys.stderr)
            return 73
        lock.seek(0)
        lock.truncate()
        lock.write(f'{os.getpid()}\n')
        lock.flush()
        with log_path.open('a', encoding='utf-8') as log:
            try:
                emit(log, f'begin tag={tag} methods={",".join(methods)}')
                if not launcher.is_file() or not os.access(launcher, os.X_OK):
                    raise QueueError(f'launcher is not executable: {launcher}')
                for method in methods:
                    launch_method(method, tag, results_base, launcher, log)
                emit(log, f'complete tag={tag}')
            except (QueueError, OSError) as error:
                emit(log, f'failed: {error}', error=True)
                raise
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run reviewed CIFAR-100 method shards sequentially.')
    parser.add_argument('--tag', required=True)
    parser.add_argument(
        '--results-base', type=Path, default=DEFAULT_RESULTS_BASE)
    parser.add_argument('--launcher', type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument('methods', nargs='*')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        methods = validate_request(args.tag, args.methods)
        return run_queue(
            args.tag, methods, args.results_base, args.launcher)
    except (QueueError, OSError) as error:
        print(f'method queue: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('method queue: interrupted', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
