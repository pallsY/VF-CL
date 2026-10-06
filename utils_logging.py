"""Lightweight tee utility: write everything from stdout+stderr to a log file
AND keep showing it on the console.

Usage:
    from utils_logging import tee_to_file
    with tee_to_file('/path/to/run.log'):
        ...  # all prints inside go to both screen and file
"""
import sys
import os
from contextlib import contextmanager
from datetime import datetime


class _Tee:
    """File-like object that writes to multiple underlying streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Pretend to be a tty so libraries that check this still print progress bars
        return getattr(self.streams[0], 'isatty', lambda: False)()


@contextmanager
def tee_to_file(log_path, mode='w', also_stderr=True, header=True):
    """Capture all stdout/stderr writes to log_path while still printing to console.

    Args:
        log_path: where to write the log
        mode: 'w' overwrite (default), 'a' append
        also_stderr: also tee stderr into the same file
        header: write a small header with timestamp + command at top
    """
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    f = open(log_path, mode, buffering=1, encoding='utf-8')  # line-buffered

    if header:
        f.write(f"# Log started: {datetime.now().isoformat()}\n")
        f.write(f"# Command: {' '.join(sys.argv)}\n")
        f.write(f"# CWD: {os.getcwd()}\n")
        f.write(f"# Log file: {log_path}\n")
        f.write('-' * 70 + '\n')

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _Tee(orig_stdout, f)
    if also_stderr:
        sys.stderr = _Tee(orig_stderr, f)

    try:
        yield log_path
    finally:
        sys.stdout = orig_stdout
        if also_stderr:
            sys.stderr = orig_stderr
        try:
            f.write('-' * 70 + '\n')
            f.write(f"# Log ended: {datetime.now().isoformat()}\n")
        except Exception:
            pass
        f.close()
