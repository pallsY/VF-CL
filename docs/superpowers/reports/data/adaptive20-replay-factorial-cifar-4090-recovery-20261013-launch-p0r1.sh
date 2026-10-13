#!/usr/bin/env bash
set -euo pipefail

root=/home/chase/Yangxx/VF-CL/results/adaptive20-replay-factorial-cifar-seed55-4090-recovery-20261013-v1
source_root=/home/chase/Yangxx/VF-CL-worktrees/replay-factorial-phasefix-20261012
python_bin=/home/chase/miniconda3/envs/py310/bin/python
config="$root/config_P0R1.json"
run="$root/cifar_seed55_P0R1"

test "$(git -C "$source_root" rev-parse HEAD)" = 29de04603ae6bdcc0ede8c951eb18318c1a8a055
test -z "$(git -C "$source_root" status --porcelain)"
test ! -e "$run"
test ! -e "$root/cifar_seed55_P0R1.launch.exit"
echo 'c15c83a2d203bf34139f574c288b81f203f7ee74c933d29800b5c9cc6b8a910e  '"$config" | sha256sum -c -
echo '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b  /home/chase/Yangxx/VF-CL/data/cifar-100-python/train' | sha256sum -c -
echo '4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864  /home/chase/Yangxx/VF-CL/data/cifar-100-python/test' | sha256sum -c -
echo 'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9  /home/chase/Yangxx/VF-CL/data/cifar-100-python/meta' | sha256sum -c -
free_bytes=$(df -B1 --output=avail "$root" | tail -1 | tr -d ' ')
test "$free_bytes" -ge 8589934592
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"

cd "$source_root"
set +e
"$python_bin" -u run_replay_factorial_pilot.py --config "$config" > "$root/cifar_seed55_P0R1.launch.log" 2>&1
code=$?
set -e
if [ "$code" -eq 0 ]; then
  test -f "$run/FORMAL_STATE_FROZEN.json"
  test -f "$run/HOLDOUT_READOUT.json"
  test -f "$run/pilot_holdout/manifest.json"
  test ! -e "$run/FORMAL_EVALUATION_PUBLISHED.json"
  test ! -e "$run/FORMAL_EVALUATION_CONSUMING.json"
  test ! -e "$run/results.json"
  tar -czf "$root/cifar_seed55_P0R1_METADATA_ONLY.tar.gz" -C "$run" FORMAL_STATE_FROZEN.json FORMAL_EVALUATION_PENDING.json HOLDOUT_READOUT.json pilot_holdout/manifest.json config.json data_flow_audit.jsonl
fi
printf '%s\n' "$code" > "$root/cifar_seed55_P0R1.launch.exit"
exit "$code"
