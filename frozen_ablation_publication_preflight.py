"""Run synthetic producer-to-auditor regressions and seal their evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('publication gate requires the real Linux producer fixture')
    code = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip():
        raise ValueError('publication preflight checkout is dirty')
    args.output_root.mkdir(parents=True, exist_ok=False)
    modules = ['test_frozen_ablation_run', 'test_frozen_ablation_audit',
               'test_frozen_ablation_manifest', 'test_frozen_ablation_smoke',
               'test_frozen_fim_application']
    command = [sys.executable, '-m', 'unittest', '-v', *modules]
    result = subprocess.run(command, cwd=code, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = args.output_root / 'publication-tests.log'
    log.write_bytes(result.stdout)
    text = result.stdout.decode('utf-8', errors='replace')
    fixture_passed = bool(re.search(r'^test_full_linux_producer_fixture_passes_formal_seal .* \.\.\. ok$', text, re.M))
    passed = result.returncode == 0 and fixture_passed and 'skipped' not in text.lower()
    report = {'schema_version': 1, 'status': 'passed' if passed else 'failed',
              'orchestration_commit': commit, 'returncode': result.returncode,
              'platform': sys.platform, 'linux_producer_fixture_passed': fixture_passed,
              'skipped_tests': 0 if 'skipped' not in text.lower() else -1,
              'data': 'synthetic fixtures only', 'real_test_access': False,
              'log_sha256': hashlib.sha256(result.stdout).hexdigest(),
              'test_source_sha256': {m: hashlib.sha256((code / (m + '.py')).read_bytes()).hexdigest()
                                     for m in modules}}
    (args.output_root / 'PUBLICATION_PREFLIGHT.json').write_text(
        json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
