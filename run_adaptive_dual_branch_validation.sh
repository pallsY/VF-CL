#!/usr/bin/env bash
set -Eeuo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=42

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >&2
}

die() {
  log "FATAL: $*"
  return 1
}

W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$W"
: "${VFCL_PYTHON:?VFCL_PYTHON must name the reviewed Python interpreter}"
PY=$(realpath -e -- "$VFCL_PYTHON") || \
  { die "VFCL_PYTHON does not exist"; exit 1; }
[[ -x "$PY" ]] || { die "VFCL_PYTHON is not executable"; exit 1; }
COMMON=$(git -C "$W" rev-parse --git-common-dir) || \
  { die "cannot derive git common directory"; exit 1; }
[[ "$COMMON" = /* ]] || COMMON="$W/$COMMON"
COMMON=$(realpath -e -- "$COMMON") || \
  { die "git common directory does not exist"; exit 1; }
[[ "$(basename -- "$COMMON")" == .git ]] || \
  { die "invalid git common directory"; exit 1; }
REPO=$(dirname -- "$COMMON")
"$PY" -c 'import pathlib,sys; raise SystemExit(pathlib.Path(sys.executable).resolve()!=pathlib.Path(sys.argv[1]).resolve())' "$PY" || \
  { die "VFCL_PYTHON does not identify its own interpreter"; exit 1; }
SOURCE_COMMIT=4651243cd8818df4f5790c9e58f3fe4236aad012
RESULTS_BASE=${RESULTS_BASE:-$REPO/results}
RESULTS_BASE=${RESULTS_BASE%/}
SMOKE_MODE=0
if [[ ${1-} == --smoke ]]; then
  SMOKE_MODE=1
  shift
fi
ROOT=${1:-$RESULTS_BASE/adaptive_dual_branch_validation_seed42_$(date +%Y%m%d_%H%M%S)}
DRIVER="$W/adaptive_dual_branch_validation.py"
EXPECTED_LAUNCHER="$W/run_adaptive_dual_branch_validation.sh"
EXECUTING_LAUNCHER=$(realpath -m -s -- "${BASH_SOURCE[0]}")
LAUNCHER_TOKEN=
IMPLEMENTATION_COMMIT=
WORKER_0_PID=
WORKER_1_PID=
WORKER_ID=
CURRENT_GPU=
STOPPING=0

trim() {
  local value=$1
  value=${value#"${value%%[![:space:]]*}"}
  value=${value%"${value##*[![:space:]]}"}
  printf '%s' "$value"
}

safe_target() {
  local target=${1:?target required}
  local base_abs target_abs current parent
  base_abs=$(realpath -m -s -- "$RESULTS_BASE") || return 1
  target_abs=$(realpath -m -s -- "$target") || return 1
  [[ "$target_abs" == "$base_abs"/* ]] || return 1
  current=$base_abs
  while :; do
    [[ ! -L "$current" ]] || return 1
    [[ "$current" == / ]] && break
    parent=${current%/*}
    [[ -n "$parent" ]] || parent=/
    current=$parent
  done
  current=$target_abs
  while [[ "$current" == "$base_abs"/* ]]; do
    [[ ! -L "$current" ]] || return 1
    parent=${current%/*}
    [[ "$parent" != "$current" ]] || break
    current=$parent
  done
}

atomic_text() {
  local target=${1:?target required} value=${2-} temp
  safe_target "$target" || return 1
  mkdir -p -- "${target%/*}"
  [[ ! -e "$target" && ! -L "$target" ]] || return 1
  temp="${target%/*}/.${target##*/}.${LAUNCHER_TOKEN:-preflight}.tmp"
  [[ ! -e "$temp" && ! -L "$temp" ]] || return 1
  (umask 077; printf '%s\n' "$value" >"$temp")
  mv -n -- "$temp" "$target"
  [[ -f "$target" && ! -L "$target" ]]
}

prepare_root() {
  local name=${ROOT##*/}
  [[ "$name" =~ ^adaptive_dual_branch_validation_seed42_[0-9]{8}_[0-9]{6}$ ]] || \
    { die "root basename is not the frozen adaptive identity"; return 1; }
  safe_target "$ROOT" || \
    { die "root is outside results or has a symlink ancestor"; return 1; }
  [[ ! -e "$ROOT" && ! -L "$ROOT" ]] || \
    { die "root already exists; restart is forbidden"; return 1; }
  mkdir -m 700 -- "$ROOT"
  mkdir -m 700 -- "$ROOT/logs"
  safe_target "$ROOT/logs/launcher.log" || \
    { die "unsafe launcher log"; return 1; }
}

driver() {
  "$PY" "$DRIVER" "$@"
}

tracked_driver() {
  local label=${1:?label required}
  shift
  safe_target "$ROOT/logs/$label.log" || return 1
  driver "$@" >>"$ROOT/logs/$label.log" 2>&1
}

preflight() {
  [[ "$PWD" == "$W" ]] || \
    { die "launcher cwd is not its frozen worktree"; return 1; }
  [[ "$EXECUTING_LAUNCHER" == "$EXPECTED_LAUNCHER"
      && -f "$EXPECTED_LAUNCHER" && ! -L "$EXPECTED_LAUNCHER" ]] || \
    { die "executing launcher is not the tracked worktree script"; return 1; }
  [[ -x "$PY" && -f "$DRIVER" && ! -L "$DRIVER" ]] || \
    { die "driver or Python missing"; return 1; }
  [[ -z "$(git status --porcelain)" ]] || \
    { die "worktree is not clean"; return 1; }
  IMPLEMENTATION_COMMIT=$(git rev-parse HEAD)
  git merge-base --is-ancestor "$SOURCE_COMMIT" "$IMPLEMENTATION_COMMIT" || \
    { die "HEAD is outside the reviewed adaptive lineage"; return 1; }
  [[ "$(git rev-parse HEAD)" == "$IMPLEMENTATION_COMMIT" ]] || \
    { die "HEAD changed during preflight"; return 1; }
  driver check >/dev/null
}

process_identity() {
  local pid=${1:?pid required} stat_line after cmdline
  [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/stat" ]] || return 1
  stat_line=$(<"/proc/$pid/stat")
  after=${stat_line##*) }
  cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline") || return 1
  printf '%s\t%s\t%s\n' \
    "$(trim "$(cut -d' ' -f3 <<<"$after")")" \
    "$(trim "$(cut -d' ' -f20 <<<"$after")")" \
    "$cmdline"
}

claim_payload() {
  local role=${1:?role required} pid=${2:?pid required} identity state pgid start_ticks cmdline
  identity=$(process_identity "$pid") || return 1
  IFS=$'\t' read -r pgid start_ticks cmdline <<<"$identity"
  state=$(awk '{print $3}' "/proc/$pid/stat") || return 1
  [[ "$state" != Z && "$pgid" =~ ^[1-9][0-9]*$ && "$start_ticks" =~ ^[0-9]+$ ]] || return 1
  printf 'token=%s\nrole=%s\npid=%s\npgid=%s\nstart_ticks=%s\ncmdline=%s\n' \
    "$LAUNCHER_TOKEN" "$role" "$pid" "$pgid" "$start_ticks" "$cmdline"
}

claim_launcher() {
  local claim="$ROOT/launcher_claim" owner="$ROOT/launcher_claim/owner"
  safe_target "$claim" || die "unsafe launcher claim"
  mkdir -- "$claim" || die "launcher root is already claimed"
  atomic_text "$owner" "$(claim_payload launcher "$$")" || die "cannot bind launcher identity"
}

claim_matches() {
  local owner=${1:?owner required} role=${2:?role required} pid=${3:?pid required}
  local token actual_role actual_pid expected actual
  [[ -f "$owner" && ! -L "$owner" ]] || return 1
  token=$(sed -n 's/^token=//p' "$owner")
  actual_role=$(sed -n 's/^role=//p' "$owner")
  actual_pid=$(sed -n 's/^pid=//p' "$owner")
  [[ "$token" == "$LAUNCHER_TOKEN" && "$actual_role" == "$role" && "$actual_pid" == "$pid" ]] || return 1
  expected=$(sed -n -e 's/^pgid=//p' -e 's/^start_ticks=//p' -e 's/^cmdline=//p' "$owner")
  actual=$(process_identity "$pid" | tr '\t' '\n') || return 1
  [[ "$expected" == "$actual" ]]
}

claim_job() {
  local dataset=${1:?dataset required} worker=${2:?worker required}
  local claim="$ROOT/claims/$dataset" owner="$ROOT/claims/$dataset/owner" section=primary
  safe_target "$claim" || return 12
  mkdir -p -- "$ROOT/claims"
  if mkdir -- "$claim" 2>/dev/null; then
    atomic_text "$owner" "$(claim_payload "worker_$worker:$dataset" "$$")" || return 12
    return 0
  fi
  (( SMOKE_MODE == 0 )) || section=smoke
  [[ -f "$ROOT/$section/$dataset/SUCCESS" && ! -L "$ROOT/$section/$dataset/SUCCESS" ]] && return 10
  return 12
}

release_job() {
  local dataset=${1:?dataset required} worker=${2:?worker required}
  local claim="$ROOT/claims/$dataset" owner="$ROOT/claims/$dataset/owner"
  claim_matches "$owner" "worker_$worker:$dataset" "$$" || return 1
  rm -- "$owner"
  rmdir -- "$claim"
}

gpu_available() {
  local gpu=${1:?gpu required} free util processes
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null) || return 1
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null) || return 1
  free=$(trim "$free"); util=$(trim "$util")
  [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
  (( free >= 6000 && util <= 20 )) || return 1
  processes=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader -i "$gpu" 2>/dev/null) || return 1
  [[ "$processes" != *python* && "$processes" != *main* ]]
}

gpu_claim_path() {
  local gpu=${1:?gpu required}
  printf '%s/gpu_claims/gpu_%s' "$ROOT" "$gpu"
}

claim_gpu() {
  local gpu=${1:?gpu required} worker=${2:?worker required}
  local claim owner
  claim=$(gpu_claim_path "$gpu")
  owner="$claim/owner"
  safe_target "$claim" || return 12
  mkdir -p -- "$ROOT/gpu_claims"
  mkdir -- "$claim" 2>/dev/null || return 10
  if ! atomic_text "$owner" "$(claim_payload "worker_$worker:gpu_$gpu" "$$")"; then
    rmdir -- "$claim" 2>/dev/null || true
    return 12
  fi
}

release_gpu() {
  local gpu=${1:?gpu required} worker=${2:?worker required}
  local claim owner
  claim=$(gpu_claim_path "$gpu")
  owner="$claim/owner"
  claim_matches "$owner" "worker_$worker:gpu_$gpu" "$$" || return 1
  rm -- "$owner"
  rmdir -- "$claim"
}

wait_gpu() {
  local worker=${1:?worker required} gpu
  local -a candidates=("$worker" "$((1 - worker))")
  while :; do
    for gpu in "${candidates[@]}"; do
      if gpu_available "$gpu" && claim_gpu "$gpu" "$worker"; then
        printf '%s\n' "$gpu"
        return 0
      fi
    done
    [[ ! -e "$ROOT/STOPPED" ]] || return 1
    sleep 30
  done
}

record_failure() {
  local worker=${1:?worker required} dataset=${2:?dataset required} code=${3:?code required}
  local target="$ROOT/FAILED_JOB"
  [[ -e "$target" || -L "$target" ]] || \
    atomic_text "$target" "worker=$worker dataset=$dataset exit=$code token=$LAUNCHER_TOKEN" || true
  [[ -e "$ROOT/STOPPED" ]] || atomic_text "$ROOT/STOPPED" "hard stop" || true
}

worker_loop() {
  local worker=${1:?worker required} dataset code gpu
  local -a datasets command
  if [[ "$worker" == 0 ]]; then
    datasets=(cifar100 upmc_food101)
  elif [[ "$worker" == 1 ]]; then
    datasets=(isolet)
    (( SMOKE_MODE == 0 )) || datasets+=(synthetic_tinyimagenet)
  else
    return 64
  fi
  for dataset in "${datasets[@]}"; do
    [[ ! -e "$ROOT/STOPPED" ]] || return 1
    if claim_job "$dataset" "$worker"; then
      :
    else
      code=$?
      [[ "$code" == 10 ]] && continue
      record_failure "$worker" "$dataset" "$code"
      return "$code"
    fi
    gpu=$(wait_gpu "$worker") || { release_job "$dataset" "$worker" || true; return 1; }
    CURRENT_GPU=$gpu
    if [[ "$dataset" == synthetic_tinyimagenet ]]; then
      command=(run-tiny-smoke --root "$ROOT" --device "cuda:$gpu")
    else
      command=(run-job --root "$ROOT" --dataset "$dataset" --seed 42 --device "cuda:$gpu")
      (( SMOKE_MODE == 0 )) || command+=(--smoke)
    fi
    if tracked_driver "${dataset}_worker_${worker}" "${command[@]}"; then
      release_gpu "$gpu" "$worker" || return 12
      CURRENT_GPU=
      release_job "$dataset" "$worker" || return 12
    else
      code=$?
      record_failure "$worker" "$dataset" "$code"
      if release_gpu "$gpu" "$worker"; then CURRENT_GPU=; else code=12; fi
      release_job "$dataset" "$worker" || true
      return "$code"
    fi
  done
}

smoke_main() {
  preflight
  prepare_root
  LAUNCHER_TOKEN=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
  atomic_text "$ROOT/launcher_token" "$LAUNCHER_TOKEN" || die "cannot freeze launcher token"
  atomic_text "$ROOT/IMPLEMENTATION_COMMIT" "$IMPLEMENTATION_COMMIT" || die "cannot freeze exact HEAD"
  tracked_driver smoke_plan plan-smoke --root "$ROOT"
  claim_launcher
  trap cleanup EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  trap 'on_signal 129' HUP
  start_worker 0
  start_worker 1
  local worker_status=0
  wait_workers || worker_status=$?
  (( worker_status == 0 )) || \
    { die "bounded synthetic smoke failure: $worker_status"; return 1; }
  [[ "$(git rev-parse HEAD)" == "$IMPLEMENTATION_COMMIT" ]] || die "HEAD changed during smoke"
  [[ -z "$(git status --porcelain)" ]] || die "worktree changed during smoke"
  tracked_driver smoke_audit audit-smoke --root "$ROOT"
  [[ -f "$ROOT/SMOKE_EXECUTION_SUCCESS" && ! -L "$ROOT/SMOKE_EXECUTION_SUCCESS" ]] || \
    die "smoke audit did not produce execution evidence"
  for forbidden in EXECUTION_SUCCESS GATE_SUCCESS GATE_FAILED PRIMARY_GATE.json; do
    [[ ! -e "$ROOT/$forbidden" && ! -L "$ROOT/$forbidden" ]] || \
      die "scientific gate evidence appeared during smoke"
  done
}

worker_claim_path() {
  local worker=${1:?worker required}
  printf '%s/worker_%s.owner' "$ROOT" "$worker"
}

worker_entry() {
  local worker=${1:?worker required} expected_root=${2:?root required} token=${3:?token required}
  local internal_smoke_mode=${4:?internal smoke mode required}
  [[ "$internal_smoke_mode" == 0 || "$internal_smoke_mode" == 1 ]] || return 64
  SMOKE_MODE=$internal_smoke_mode
  ROOT=$expected_root
  LAUNCHER_TOKEN=$token
  WORKER_ID=$worker
  trap worker_gpu_cleanup EXIT
  local owner
  owner=$(worker_claim_path "$worker")
  atomic_text "$owner" "$(claim_payload "worker_$worker" "$$")" || return 12
  worker_loop "$worker"
}

worker_gpu_cleanup() {
  local status=$?
  [[ -z "$CURRENT_GPU" ]] || release_gpu "$CURRENT_GPU" "$WORKER_ID" || true
  return "$status"
}

start_worker() {
  local worker=${1:?worker required} pid
  setsid bash "$EXPECTED_LAUNCHER" --worker "$worker" "$ROOT" "$LAUNCHER_TOKEN" "$SMOKE_MODE" &
  pid=$!
  if [[ "$worker" == 0 ]]; then WORKER_0_PID=$pid; else WORKER_1_PID=$pid; fi
}

terminate_group() {
  local worker=${1:?worker required} pid owner pgid current i
  if [[ "$worker" == 0 ]]; then pid=$WORKER_0_PID; else pid=$WORKER_1_PID; fi
  [[ -n "$pid" ]] || return 0
  owner=$(worker_claim_path "$worker")
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
  claim_matches "$owner" "worker_$worker" "$pid" || return 1
  pgid=$(sed -n 's/^pgid=//p' "$owner")
  [[ "$pgid" == "$pid" ]] || return 1
  # verified process groups only: kill -- -PGID
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for i in {1..20}; do
    kill -0 -- "-$pgid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 -- "-$pgid" 2>/dev/null; then
    current=$(process_identity "$pid" || true)
    [[ -n "$current" ]] && claim_matches "$owner" "worker_$worker" "$pid" || return 1
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  ! kill -0 -- "-$pgid" 2>/dev/null
}

worker_running() {
  local pid=${1:?pid required} state
  kill -0 "$pid" 2>/dev/null || return 1
  state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [[ "$state" != Z ]]
}

wait_workers() {
  local status0=-1 status1=-1 code
  while :; do
    if (( status0 == -1 )) && ! worker_running "$WORKER_0_PID"; then
      code=0
      wait "$WORKER_0_PID" || code=$?
      status0=$code
    fi
    if (( status1 == -1 )) && ! worker_running "$WORKER_1_PID"; then
      code=0
      wait "$WORKER_1_PID" || code=$?
      status1=$code
    fi
    if (( status0 > 0 )); then
      (( status1 != -1 )) || terminate_group 1 || return 125
      return "$status0"
    fi
    if (( status1 > 0 )); then
      (( status0 != -1 )) || terminate_group 0 || return 125
      return "$status1"
    fi
    (( status0 == 0 && status1 == 0 )) && return 0
    sleep 1
  done
}

cleanup() {
  local status=$?
  (( STOPPING == 0 )) || exit "$status"
  STOPPING=1
  terminate_group 0 || true
  terminate_group 1 || true
  exit "$status"
}

on_signal() {
  local status=${1:?status required}
  trap - INT TERM HUP
  STOPPING=1
  terminate_group 0 || true
  terminate_group 1 || true
  exit "$status"
}

main() {
  preflight
  prepare_root
  LAUNCHER_TOKEN=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
  atomic_text "$ROOT/launcher_token" "$LAUNCHER_TOKEN" || die "cannot freeze launcher token"
  atomic_text "$ROOT/IMPLEMENTATION_COMMIT" "$IMPLEMENTATION_COMMIT" || die "cannot freeze exact HEAD"
  # PRIMARY_PLAN.json is exclusively written by the driver before workers.
  tracked_driver primary_plan plan --root "$ROOT"
  claim_launcher
  trap cleanup EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  trap 'on_signal 129' HUP

  # worker_loop 0
  # worker_loop 1
  # bounded identities: worker_0 worker_1
  start_worker 0
  start_worker 1
  local worker_status=0
  wait_workers || worker_status=$?
  (( worker_status == 0 )) || \
    { die "bounded worker failure: $worker_status"; return 1; }
  [[ "$(git rev-parse HEAD)" == "$IMPLEMENTATION_COMMIT" ]] || die "HEAD changed during execution"
  [[ -z "$(git status --porcelain)" ]] || die "worktree changed during execution"
  tracked_driver primary_gate summarize --root "$ROOT"
  [[ -f "$ROOT/EXECUTION_SUCCESS" && ! -L "$ROOT/EXECUTION_SUCCESS" ]] || \
    die "summarize did not produce execution evidence"
}

if [[ ${1-} == --worker ]]; then
  shift
  worker_entry "$@"
elif [[ ${ADAPTIVE_LAUNCHER_SOURCE_ONLY:-0} != 1 ]]; then
  if (( SMOKE_MODE == 1 )); then smoke_main "$@"; else main "$@"; fi
fi
