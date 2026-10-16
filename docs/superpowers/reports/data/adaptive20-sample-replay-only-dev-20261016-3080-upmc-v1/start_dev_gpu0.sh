#!/usr/bin/env bash
set -euo pipefail

run=${1:?run required}
test "$(whoami)" = yangxx
test "$(hostname)" = c3080.upcbrfed.xyz
case "$run" in
    upmc_seed54_SAMPLE_ONLY) seed=54; predecessor=isolet_seed55_SAMPLE_ONLY; min_bytes=8589934592 ;;
    upmc_seed55_SAMPLE_ONLY) seed=55; predecessor=upmc_seed54_SAMPLE_ONLY; min_bytes=8589934592 ;;
    *) echo "invalid run $run" >&2; exit 2 ;;
esac

root=/brf/users/yangxx/VFCL-results/adaptive20-sample-replay-only-dev-20261016-3080-upmc-v1
source_root=/brf/users/yangxx/VF-CL-worktrees/raw-sample-replay-20261016-standby
python_bin=/home/c3080/YangXiaoXiang/envs/vfcl/bin/python
config="$root/config_$run.json"
test "$(sha256sum "$root/manifest.json" | cut -d ' ' -f1)" = 0a388a144d5335455c06e5da1881271235671a5d34c1cf1b95f7ed553cbc1baf
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED="$seed" TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0

test "$(git -C "$source_root" rev-parse HEAD)" = 99e1da7e05e7eeb8b57587b6d3a75c6eaa490066
test -z "$(git -C "$source_root" status --porcelain)"
test ! -e "$root/$run"
test ! -e "$root/$run.launch.exit"
if [ -n "$predecessor" ]; then
    test -f "$root/ARCHIVED_$predecessor.txt"
    test ! -e "$root/$predecessor"
fi

ROOT="$root" RUN="$run" python3 - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['ROOT'])
name=os.environ['RUN']
manifest=json.loads((root/'manifest.json').read_text())
assert manifest['source_commit']=='99e1da7e05e7eeb8b57587b6d3a75c6eaa490066'
entry=next(item for item in manifest['runs'] if item['run']==name)
config_path=root/f'config_{name}.json'
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==entry['config_sha256']
config=json.loads(config_path.read_text())
dataset=entry['dataset']
if dataset=='cifar':
    paths={key:pathlib.Path(config['data_path'])/'cifar-100-python'/key for key in ('train','test','meta')}
    expected=manifest['data_sha256']['cifar']
else:
    paths={'vector':pathlib.Path(config['vector_npz'])}
    expected={'vector':manifest['data_sha256'][dataset]}
for key,path in paths.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[key],path
print('PREFLIGHT_OK',name)
PY

while true; do
    test ! -e "$root/$run"
    free_bytes=$(df -B1 --output=avail "$root" | tail -1 | tr -d ' ')
    if [ "$free_bytes" -ge "$min_bytes" ] &&
       [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; then
        break
    fi
    sleep 30
done

cd "$source_root"
set +e
"$python_bin" -u run_replay_factorial_pilot.py --config "$config" > "$root/$run.launch.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$root/$run.launch.exit"
exit "$code"
