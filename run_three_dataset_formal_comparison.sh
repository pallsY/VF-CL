#!/usr/bin/env bash
set -uo pipefail

readonly REVIEWED_BRANCH='codex/three-dataset-seed42-pilot'
readonly REVIEWED_PYTHON='/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'
readonly MIN_AVAILABLE_KB=10485760
readonly PILOT_MIN_MEMAVAILABLE_KB=41943040
readonly GPU_QUERY_TIMEOUT_SECONDS=2
readonly SMOKE_GPU_WAIT_SECONDS=1
readonly CHILD_TERM_GRACE_SECONDS=4
readonly CHILD_CLEANUP_OVERHEAD_SECONDS=4
readonly PARENT_WORKER_GRACE_SECONDS=12
readonly PARENT_CHILD_KILL_WAIT_SECONDS=2
readonly FORMAL_TEMP_ROOT="${TMPDIR:-/tmp}"
readonly SCRIPT_REAL="$(realpath -e -- "${BASH_SOURCE[0]}")"
readonly WORKTREE="$(dirname -- "$SCRIPT_REAL")"
readonly DRIVER="$WORKTREE/three_dataset_formal_driver.py"
readonly RETENTION="$WORKTREE/prune_completed_runs.py"
readonly MARKERS=(
  FAILED_JOB FORMAL_STOPPED FORMAL_PHASE_SUCCESS
  EXPLANATION_PHASE_SUCCESS FORMAL_EXECUTION_SUCCESS
  PILOT_PHASE_SUCCESS PILOT_EXECUTION_SUCCESS
  RECOVERY_PHASE_SUCCESS RECOVERY_EXECUTION_SUCCESS
  FULL_MATRIX_PHASE_SUCCESS FULL_MATRIX_EXECUTION_SUCCESS
  DATASET_PHASE_SUCCESS DATASET_EXECUTION_SUCCESS
  DATASET_CONTINUATION_PHASE_SUCCESS DATASET_CONTINUATION_SUCCESS
  METHOD_SHARD_PHASE_SUCCESS METHOD_SHARD_SUCCESS
)
readonly SMOKE_MARKERS=(
  SMOKE_FAILED SMOKE_STOPPED SMOKE_AUDIT.json SMOKE_EXECUTION_SUCCESS
)

die() {
  printf 'formal launcher: %s\n' "$*" >&2
  return 1
}

reviewed_branch() {
  case ${VFCL_EXPERIMENT_PROFILE:-formal} in
    seed42-adaptive-recovery) printf '%s\n' 'codex/adaptive-bic-corpus-identity-fix' ;;
    full-public-matrix) printf '%s\n' 'codex/full-matrix-scientific-correctness-fix' ;;
    single-dataset-full-matrix) printf '%s\n' 'codex/dual-gpu-dataset-formal' ;;
    single-dataset-verified-continuation-v1) printf '%s\n' 'codex/gpm-audit-low-memory' ;;
    single-method-formal-v1) printf '%s\n' 'codex/gpm-audit-low-memory' ;;
    *) printf '%s\n' "$REVIEWED_BRANCH" ;;
  esac
}

retention_enabled() {
  case ${VFCL_EXPERIMENT_PROFILE:-formal} in
    seed42-adaptive-recovery|full-public-matrix|single-dataset-full-matrix|single-dataset-verified-continuation-v1|single-method-formal-v1) return 0 ;;
    *) return 1 ;;
  esac
}

disk_ready() {
  [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == full-public-matrix ||
     ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
     ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
     ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]] || return 0
  local payload slots=1
  if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]]; then
    slots=${VFCL_GPU_COUNT:-1}
    [[ $slots == 1 || $slots == 2 ]] || return 2
  fi
  payload=$("$REVIEWED_PYTHON" "$DRIVER" disk-status \
    --root "$FORMAL_ROOT" --requested-slots "$slots") || return 2
  "$REVIEWED_PYTHON" -c '
import json, sys
try:
    payload = json.loads(sys.argv[1])
    if type(payload) is not dict or type(payload.get("safe")) is not bool:
        raise ValueError("invalid disk status")
except (ValueError, TypeError):
    raise SystemExit(2)
raise SystemExit(0 if payload["safe"] else 1)
' "$payload"
}

recovery_resources_ready() {
  [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == seed42-adaptive-recovery ]] || return 0
  local available
  pilot_memory_ready || return 1
  available=$(df -Pk -- "$FORMAL_ROOT" | awk 'NR == 2 {print $4}') || return 1
  [[ $available =~ ^[0-9]+$ && $available -ge 31457280 ]]
}

proc_start() {
  local pid=$1 stat_line
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_line=${stat_line##*) }
  set -- $stat_line
  printf '%s\n' "${20}"
}

proc_pgid_from_stat() {
  local pid=$1 stat_line
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_line=${stat_line##*) }
  set -- $stat_line
  [[ ${3:-} =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$3"
}

proc_state() {
  local pid=$1 stat_line
  IFS= read -r stat_line < "/proc/$pid/stat" || return 1
  stat_line=${stat_line##*) }
  printf '%s\n' "${stat_line%% *}"
}

proc_pgid() {
  local value
  value=$(ps -o pgid= -p "$1" 2>/dev/null) || return 1
  value=${value//[[:space:]]/}
  [[ $value =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$value"
}

validate_root_path() {
  local raw=$1 current resolved
  if [[ $raw != /* || $raw == / ]]; then
    die 'ROOT must be an absolute child directory'
    return 1
  fi
  if [[ "/$raw/" == *'/../'* || "/$raw/" == *'/./'* ]]; then
    die 'ROOT must not contain dot traversal'
    return 1
  fi
  if [[ ! -d $raw || -L $raw ]]; then
    die 'ROOT must be an existing regular directory'
    return 1
  fi
  current=$raw
  while [[ $current != / ]]; do
    if [[ -L $current ]]; then
      die 'ROOT must not have a symlink ancestor'
      return 1
    fi
    current=$(dirname -- "$current")
  done
  if ! resolved=$(realpath -e -- "$raw"); then
    die 'ROOT cannot be resolved'
    return 1
  fi
  if [[ $resolved != "$raw" ]]; then
    die 'ROOT must already be canonical'
    return 1
  fi
  printf '%s\n' "$resolved"
}

root_identity() {
  stat -Lc '%d:%i' -- "$1"
}

check_root_identity() {
  [[ $(root_identity "$FORMAL_ROOT") == "$FORMAL_ROOT_ID" ]] ||
    die 'formal root identity changed'
}

check_frozen_state() {
  local head py_real py_hash
  if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} != "$FORMAL_FROZEN_PROFILE" ]]; then
    die 'experiment profile changed'
    return 1
  fi
  if [[ ${VFCL_FORMAL_DATASET:-} != "${FORMAL_FROZEN_DATASET:-}" ]]; then
    die 'formal dataset changed'
    return 1
  fi
  if [[ ${VFCL_FORMAL_METHOD:-} != "${FORMAL_FROZEN_METHOD:-}" ]]; then
    die 'formal method changed'
    return 1
  fi
  if [[ ${VFCL_GPU_COUNT:-2} != "${FORMAL_FROZEN_GPU_COUNT:-}" ]]; then
    die 'GPU count changed'
    return 1
  fi
  if ! check_root_identity; then return 1; fi
  if [[ -n $(git -C "$WORKTREE" status --porcelain=v1) ]]; then
    die 'implementation worktree became dirty'
    return 1
  fi
  if ! head=$(git -C "$WORKTREE" rev-parse HEAD); then return 1; fi
  if [[ $head != "$FORMAL_FROZEN_HEAD" ]]; then
    die 'implementation HEAD changed'
    return 1
  fi
  if ! py_real=$(realpath -e -- "$REVIEWED_PYTHON"); then return 1; fi
  if ! py_hash=$(sha256sum -- "$REVIEWED_PYTHON"); then return 1; fi
  py_hash=${py_hash%% *}
  if [[ $py_real != "$FORMAL_FROZEN_PY_REAL" || $py_hash != "$FORMAL_FROZEN_PY_SHA" ]]; then
    die 'reviewed Python identity changed'
    return 1
  fi
  return 0
}

retention_control_state() {
  local path=${WORKER_RETENTION_CONTROL:-}
  local pending=${WORKER_RETENTION_PENDING:-}
  local owner=${WORKER_OWNER[2]:-}
  [[ -n $path && -n $pending && -n $owner &&
     $path == "$WORKER_GATE_ROOT/retention-active" &&
     $pending == "$WORKER_GATE_ROOT/.retention-active" ]] || return 2
  if [[ -L $path || -L $pending ]]; then return 2; fi
  if [[ ! -e $path && ! -e $pending ]]; then return 1; fi
  [[ -f $path && ! -e $pending ]] || return 2
  "$REVIEWED_PYTHON" -c '
import json, os, stat, sys
from pathlib import Path

path, expected_raw, root_raw = sys.argv[1:]
try:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(fd)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600):
            raise ValueError
        content = os.read(fd, 65537)
        if len(content) > 65536:
            raise ValueError
    finally:
        os.close(fd)
    payload = json.loads(content)
    expected = json.loads(expected_raw)
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(payload).encode() + b"\n" != content:
        raise ValueError
    if (type(payload) is not dict
            or set(payload) != {"kind", "owner", "spec_key"}
            or payload["kind"] != "formal_retention_active"
            or type(payload["spec_key"]) is not str or not payload["spec_key"]
            or payload["owner"] != expected):
        raise ValueError
    root = Path(root_raw)
    active = json.loads((root / "audit_queue" / "active.json").read_text())
    audit_owner = json.loads((root / "audit-owner.json").read_text())
    if (type(active) is not dict or active.get("spec_key") != payload["spec_key"]
            or audit_owner != expected):
        raise ValueError
except (OSError, TypeError, ValueError):
    raise SystemExit(2)
' "$path" "$owner" "$FORMAL_ROOT" >/dev/null 2>&1
}

retention_source_ready() {
  [[ -f $RETENTION && ! -L $RETENTION ]] &&
    git -C "$WORKTREE" ls-files --error-unmatch -- "${RETENTION##*/}" >/dev/null
}

preflight() {
  local mode=$1 root_arg=$2 branch head status available py_real py_hash name
  if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]]; then
    export VFCL_GPU_COUNT=${VFCL_GPU_COUNT:-1}
  fi
  if ((PARENT_WORKER_GRACE_SECONDS <
       CHILD_TERM_GRACE_SECONDS + CHILD_CLEANUP_OVERHEAD_SECONDS)); then
    die 'parent worker grace is shorter than child cleanup invariant'
    return 1
  fi
  if [[ ${SMOKE_MODE+x} ]]; then
    die 'external SMOKE_MODE is not supported'
    return 1
  fi
  if [[ ! -f $SCRIPT_REAL || -L $SCRIPT_REAL ]]; then
    die 'launcher must be a regular file'
    return 1
  fi
  if [[ ! -f $DRIVER || -L $DRIVER ]]; then
    die 'formal driver is unavailable'
    return 1
  fi
  if ! retention_source_ready; then
    die 'retention source is unavailable or untracked'
    return 1
  fi
  if ! branch=$(git -C "$WORKTREE" branch --show-current); then return 1; fi
  if [[ $branch != "$(reviewed_branch)" ]]; then
    die 'implementation branch differs'
    return 1
  fi
  if ! head=$(git -C "$WORKTREE" rev-parse HEAD); then return 1; fi
  if [[ ! $head =~ ^[0-9a-f]{40}$ || $head == 0000000000000000000000000000000000000000 ]]; then
    die 'implementation HEAD is invalid'
    return 1
  fi
  if ! status=$(git -C "$WORKTREE" status --porcelain=v1); then return 1; fi
  if [[ -n $status ]]; then
    die 'implementation worktree is dirty'
    return 1
  fi
  if [[ ${VFCL_PYTHON+x} && ${VFCL_PYTHON} != "$REVIEWED_PYTHON" ]]; then
    die 'VFCL_PYTHON differs from the reviewed interpreter'
    return 1
  fi
  if [[ ! -f $REVIEWED_PYTHON || ! -x $REVIEWED_PYTHON ]]; then
    die 'reviewed Python is unavailable'
    return 1
  fi
  if ! py_real=$(realpath -e -- "$REVIEWED_PYTHON"); then return 1; fi
  if ! py_hash=$(sha256sum -- "$REVIEWED_PYTHON"); then return 1; fi
  py_hash=${py_hash%% *}
  if ! FORMAL_ROOT=$(validate_root_path "$root_arg"); then return 1; fi
  if ! FORMAL_ROOT_ID=$(root_identity "$FORMAL_ROOT"); then return 1; fi
  recovery_resources_ready || { die 'insufficient recovery memory or disk space'; return 1; }
  for name in "${MARKERS[@]}"; do
    if [[ -e $FORMAL_ROOT/$name || -L $FORMAL_ROOT/$name ]]; then
      die "existing control marker rejects launch: $name"
      return 1
    fi
  done
  if ! available=$(df -Pk -- "$FORMAL_ROOT" | awk 'NR == 2 {print $4}'); then return 1; fi
  if [[ ! $available =~ ^[0-9]+$ || $available -lt $MIN_AVAILABLE_KB ]]; then
    die 'insufficient formal-root disk space'
    return 1
  fi
  export VFCL_PYTHON=$REVIEWED_PYTHON
  if [[ ( ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
          ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
          ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ) ]] &&
      ! disk_ready; then
    die 'insufficient single-dataset disk capacity'
    return 1
  fi
  if ! "$REVIEWED_PYTHON" "$DRIVER" check --root "$FORMAL_ROOT" >/dev/null; then
    die 'installed formal authority rejected'
    return 1
  fi
  if ! check_root_identity; then return 1; fi
  FORMAL_FROZEN_HEAD=$head
  FORMAL_FROZEN_PY_REAL=$py_real
  FORMAL_FROZEN_PY_SHA=$py_hash
  FORMAL_FROZEN_PROFILE=${VFCL_EXPERIMENT_PROFILE:-formal}
  FORMAL_FROZEN_DATASET=${VFCL_FORMAL_DATASET:-}
  FORMAL_FROZEN_METHOD=${VFCL_FORMAL_METHOD:-}
  FORMAL_FROZEN_GPU_COUNT=${VFCL_GPU_COUNT:-2}
  export FORMAL_ROOT FORMAL_ROOT_ID FORMAL_FROZEN_HEAD
  export FORMAL_FROZEN_PY_REAL FORMAL_FROZEN_PY_SHA
  export FORMAL_FROZEN_PROFILE FORMAL_FROZEN_DATASET FORMAL_FROZEN_GPU_COUNT
  export FORMAL_FROZEN_METHOD
  if [[ $mode == check ]]; then printf 'formal launcher check passed\n'; fi
  return 0
}

smoke_preflight() {
  local root_arg=$1 branch head status available py_real py_hash name
  if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]]; then
    export VFCL_GPU_COUNT=${VFCL_GPU_COUNT:-1}
  fi
  if [[ ${SMOKE_MODE+x} ]]; then
    die 'external SMOKE_MODE is not supported'
    return 1
  fi
  if [[ ! -f $SCRIPT_REAL || -L $SCRIPT_REAL || ! -f $DRIVER || -L $DRIVER ]] ||
      ! retention_source_ready; then
    die 'smoke launcher source is unavailable'
    return 1
  fi
  branch=$(git -C "$WORKTREE" branch --show-current) || return 1
  [[ $branch == "$(reviewed_branch)" ]] || {
    die 'implementation branch differs'
    return 1
  }
  head=$(git -C "$WORKTREE" rev-parse HEAD) || return 1
  [[ $head =~ ^[0-9a-f]{40}$ && $head != 0000000000000000000000000000000000000000 ]] || {
    die 'implementation HEAD is invalid'
    return 1
  }
  status=$(git -C "$WORKTREE" status --porcelain=v1) || return 1
  [[ -z $status ]] || {
    die 'implementation worktree is dirty'
    return 1
  }
  if [[ ${VFCL_PYTHON+x} && ${VFCL_PYTHON} != "$REVIEWED_PYTHON" ]]; then
    die 'VFCL_PYTHON differs from the reviewed interpreter'
    return 1
  fi
  [[ -f $REVIEWED_PYTHON && -x $REVIEWED_PYTHON ]] || {
    die 'reviewed Python is unavailable'
    return 1
  }
  py_real=$(realpath -e -- "$REVIEWED_PYTHON") || return 1
  py_hash=$(sha256sum -- "$REVIEWED_PYTHON") || return 1
  py_hash=${py_hash%% *}
  FORMAL_ROOT=$(validate_root_path "$root_arg") || return 1
  FORMAL_ROOT_ID=$(root_identity "$FORMAL_ROOT") || return 1
  recovery_resources_ready || { die 'insufficient recovery memory or disk space'; return 1; }
  for name in "${MARKERS[@]}" "${SMOKE_MARKERS[@]}"; do
    if [[ -e $FORMAL_ROOT/$name || -L $FORMAL_ROOT/$name ]]; then
      die "existing control marker rejects smoke launch: $name"
      return 1
    fi
  done
  if [[ -n $(find "$FORMAL_ROOT" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    die 'generated smoke root must be empty'
    return 1
  fi
  available=$(df -Pk -- "$FORMAL_ROOT" | awk 'NR == 2 {print $4}') || return 1
  [[ $available =~ ^[0-9]+$ && $available -ge $MIN_AVAILABLE_KB ]] || {
    die 'insufficient smoke-root disk space'
    return 1
  }
  export VFCL_PYTHON=$REVIEWED_PYTHON
  "$REVIEWED_PYTHON" "$DRIVER" smoke-plan --root "$FORMAL_ROOT" >/dev/null || {
    die 'generated smoke planning failed'
    return 1
  }
  FORMAL_FROZEN_HEAD=$head
  FORMAL_FROZEN_PY_REAL=$py_real
  FORMAL_FROZEN_PY_SHA=$py_hash
  FORMAL_FROZEN_PROFILE=${VFCL_EXPERIMENT_PROFILE:-formal}
  FORMAL_FROZEN_DATASET=${VFCL_FORMAL_DATASET:-}
  FORMAL_FROZEN_GPU_COUNT=${VFCL_GPU_COUNT:-2}
  export FORMAL_ROOT FORMAL_ROOT_ID FORMAL_FROZEN_HEAD
  export FORMAL_FROZEN_PY_REAL FORMAL_FROZEN_PY_SHA
  export FORMAL_FROZEN_PROFILE FORMAL_FROZEN_DATASET FORMAL_FROZEN_GPU_COUNT
  "$REVIEWED_PYTHON" "$DRIVER" smoke-check --root "$FORMAL_ROOT" >/dev/null || {
    die 'generated smoke plan rejected'
    return 1
  }
}

marker_payload() {
  local kind=$1 role=${2:-launcher} code=${3:-0} spec=${4:-}
  printf '{"exit_code":%d,"kind":"%s","role":"%s","spec_key":"%s"}' \
    "$code" "$kind" "$role" "$spec"
}

memory_available_kb() {
  awk '$1 == "MemAvailable:" {print $2; found=1} END {exit !found}' /proc/meminfo
}

pilot_memory_ready() {
  case ${VFCL_EXPERIMENT_PROFILE:-formal} in
    seed42-pilot|seed42-adaptive-recovery|full-public-matrix|single-dataset-full-matrix|single-dataset-verified-continuation-v1|single-method-formal-v1) ;;
    *) return 0 ;;
  esac
  local available
  available=$(memory_available_kb) || return 1
  [[ $available =~ ^[0-9]+$ && $available -ge $PILOT_MIN_MEMAVAILABLE_KB ]]
}

pilot_parent_running() {
  case ${VFCL_EXPERIMENT_PROFILE:-formal} in
    seed42-pilot|seed42-adaptive-recovery|full-public-matrix|single-dataset-full-matrix|single-dataset-verified-continuation-v1|single-method-formal-v1) ;;
    *) return 0 ;;
  esac
  local current state
  [[ ! -e $FORMAL_ROOT/FORMAL_STOPPED && ! -e $FORMAL_ROOT/FAILED_JOB ]] || return 143
  current=$(proc_start "$1" 2>/dev/null) || return 143
  state=$(proc_state "$1" 2>/dev/null) || return 143
  [[ $current == "${FORMAL_PARENT_START:-}" && $state != Z ]] || return 143
}

mark_once() {
  local name=$1 payload=$2
  "$REVIEWED_PYTHON" "$DRIVER" mark --root "$FORMAL_ROOT" \
    --name "$name" --payload-json "$payload" >/dev/null
}

trim() {
  local value=$1
  value=${value#"${value%%[![:space:]]*}"}
  value=${value%"${value##*[![:space:]]}"}
  printf '%s' "$value"
}

gpu_eligible() {
  local wanted=$1 allow_foreign=${2:-0}
  local line index uuid free util extra app_uuid app_pid app_name
  local gpu_output compute_output matched= matched_free= matched_util=
  local uuid0= uuid1= seen0=0 seen1=0
  [[ $allow_foreign == 0 || $allow_foreign == 1 ]] || return 2
  if ! gpu_output=$(timeout --kill-after=1s "${GPU_QUERY_TIMEOUT_SECONDS}s" \
      nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null); then
    return 2
  fi
  while IFS= read -r line; do
    [[ -n $line ]] || continue
    extra=
    IFS=',' read -r index uuid free util extra <<< "$line"
    index=$(trim "${index:-}"); uuid=$(trim "${uuid:-}")
    free=$(trim "${free:-}"); util=$(trim "${util:-}")
    extra=$(trim "${extra:-}")
    [[ -z $extra && -n $uuid && $free =~ ^[0-9]+$ &&
       $util =~ ^[0-9]+$ ]] || return 2
    if [[ $index == 0 && $seen0 -eq 0 ]]; then
      uuid0=$uuid; seen0=1
    elif [[ $index == 1 && $seen1 -eq 0 ]]; then
      uuid1=$uuid; seen1=1
    else
      return 2
    fi
    if [[ $index == "$wanted" ]]; then
      matched=$uuid; matched_free=$free; matched_util=$util
    fi
  done <<< "$gpu_output"
  if [[ ${VFCL_GPU_COUNT:-2} == 1 ]]; then
    [[ $seen0 -eq 1 && $seen1 -eq 0 ]] || return 2
  elif [[ ${VFCL_GPU_COUNT:-2} == 2 ]]; then
    [[ $seen0 -eq 1 && $seen1 -eq 1 && $uuid0 != "$uuid1" ]] || return 2
  else
    return 2
  fi
  [[ -n $matched ]] || return 1
  [[ $matched_free -ge 6000 && $matched_util -le 20 ]] || return 1
  if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == full-public-matrix ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
        ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]]; then
    [[ $matched_util -eq 0 ]] || return 1
  fi
  if ! compute_output=$(timeout --kill-after=1s "${GPU_QUERY_TIMEOUT_SECONDS}s" \
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
      --format=csv,noheader,nounits 2>/dev/null); then
    return 2
  fi
  while IFS= read -r line; do
    [[ -n $line ]] || continue
    extra=
    IFS=',' read -r app_uuid app_pid app_name extra <<< "$line"
    app_uuid=$(trim "${app_uuid:-}"); app_pid=$(trim "${app_pid:-}")
    app_name=$(trim "${app_name:-}"); extra=$(trim "${extra:-}")
    [[ -z $extra && -n $app_name && $app_pid =~ ^[1-9][0-9]*$ &&
       ( $app_uuid == "$uuid0" || $app_uuid == "$uuid1" ) ]] || return 2
    if [[ ( ${VFCL_EXPERIMENT_PROFILE:-formal} == full-public-matrix ||
            ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
            ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
            ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ) &&
          $app_uuid == "$matched" ]]; then
      return 1
    fi
    if [[ $allow_foreign == 0 && $app_uuid == "$matched" &&
          ${app_name,,} =~ (python|main) ]]; then
      return 1
    fi
  done <<< "$compute_output"
  FORMAL_GPU_UUID=$matched
  export FORMAL_GPU_UUID
  return 0
}

claim_gpu_for_worker() {
  local index=$1 owner_json=$2 first second gpu status initial_uuid
  if [[ $index -eq 0 ]]; then first=0; second=1; else first=1; second=0; fi
  for gpu in "$first" "$second"; do
    gpu_eligible "$gpu" 1; status=$?
    [[ $status -ne 2 ]] || return 2
    [[ $status -eq 0 ]] || continue
    initial_uuid=$FORMAL_GPU_UUID
    FORMAL_CLAIMED_GPU=$gpu
    FORMAL_CLAIMED_GPU_UUID=$initial_uuid
    if "$REVIEWED_PYTHON" "$DRIVER" gpu-claim --root "$FORMAL_ROOT" \
        --physical-gpu "$gpu" --owner-json "$owner_json" >/dev/null 2>&1; then
      gpu_eligible "$gpu" 1; status=$?
      if [[ $status -eq 0 && $FORMAL_GPU_UUID == "$initial_uuid" ]]; then
        return 0
      fi
      "$REVIEWED_PYTHON" "$DRIVER" gpu-release --root "$FORMAL_ROOT" \
        --physical-gpu "$gpu" --owner-json "$owner_json" >/dev/null 2>&1 ||
        return 2
      FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
      [[ $status -eq 1 ]] || return 2
    else
      status=$?
      if [[ $status -eq 4 ]]; then
        FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
      else
        return 2
      fi
    fi
  done
  return 1
}

monitor_peak_mib() {
  local job_pid=$1 uuid=$2 line app_uuid app_pid used extra peak=0 rows
  while kill -0 "$job_pid" 2>/dev/null; do
    if ! rows=$(timeout --kill-after=1s "${GPU_QUERY_TIMEOUT_SECONDS}s" nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null); then
      return 1
    fi
    while IFS= read -r line; do
      [[ -n $line ]] || continue
      extra=
      IFS=',' read -r app_uuid app_pid used extra <<< "$line"
      app_uuid=$(trim "${app_uuid:-}"); app_pid=$(trim "${app_pid:-}")
      used=$(trim "${used:-}"); extra=$(trim "${extra:-}")
      [[ -z $extra && -n $app_uuid && $app_pid =~ ^[1-9][0-9]*$ &&
         $used =~ ^[0-9]+$ ]] || return 1
      if [[ $app_uuid == "$uuid" && $app_pid == "$job_pid" && $used =~ ^[0-9]+$ && $used -gt $peak ]]; then
        peak=$used
      fi
    done <<< "$rows"
    sleep 0.05
  done
  printf '%s\n' "$peak"
}

signal_verified_child_group() {
  local leader_pid=$1 leader_start=$2 expected_pgid=$3 token=$4 signal_number=$5
  [[ $leader_pid =~ ^[1-9][0-9]*$ && $leader_start =~ ^[0-9]+$ &&
     $expected_pgid == "$leader_pid" && $token =~ ^[0-9a-f]{64}$ &&
     $signal_number =~ ^(0|9|15)$ ]] || return 1
  "$REVIEWED_PYTHON" -c '
import ctypes, errno, os, pathlib, sys

leader_pid, leader_start, expected_pgid, raw_token, signal_number = sys.argv[1:]
leader_pid = int(leader_pid)
leader_start = int(leader_start)
expected_pgid = int(expected_pgid)
signal_number = int(signal_number)
token = b"FORMAL_CHILD_TOKEN=" + raw_token.encode()
proc = pathlib.Path("/proc")

def identity(pid):
    try:
        parts = (proc / str(pid) / "stat").read_bytes().rsplit(b") ", 1)
    except FileNotFoundError:
        return None
    if len(parts) != 2:
        raise ValueError
    fields = parts[1].split()
    if len(fields) < 20:
        raise ValueError
    return int(fields[19]), int(fields[2]), fields[0]

def owns(pid):
    try:
        values = (proc / str(pid) / "environ").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    return token in values

try:
    leader = identity(leader_pid)
    if leader is not None:
        if leader[0] != leader_start or leader[1] != expected_pgid:
            raise ValueError
        if leader[2] != b"Z":
            leader_owned = owns(leader_pid)
            if leader_owned is None:
                leader = None
            elif not leader_owned:
                raise ValueError

    # ponytail: glibc 2.35 and this Python build lack pidfd wrappers;
    # use the reviewed x86_64 kernel syscall numbers, or fail closed.
    if os.uname().machine != "x86_64":
        raise ValueError
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long

    candidates = []
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        current = identity(pid)
        if current is None or current[1] != expected_pgid or current[2] == b"Z":
            continue
        owned = owns(pid)
        if owned is None:
            continue
        if owned:
            candidates.append((pid != leader_pid, pid, current[0]))

    found = False
    for _, pid, start in sorted(candidates):
        fd = syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
        if fd < 0:
            if ctypes.get_errno() == errno.ESRCH:
                continue
            raise OSError(ctypes.get_errno(), "pidfd_open")
        try:
            current = identity(pid)
            if current is None:
                continue
            if current[0] != start or current[1] != expected_pgid:
                raise ValueError
            owned = owns(pid)
            if owned is None:
                continue
            if not owned:
                raise ValueError
            if syscall(ctypes.c_long(424), ctypes.c_int(fd),
                       ctypes.c_int(signal_number), ctypes.c_void_p(0),
                       ctypes.c_uint(0)) < 0:
                if ctypes.get_errno() == errno.ESRCH:
                    continue
                raise OSError(ctypes.get_errno(), "pidfd_send_signal")
            found = True
        finally:
            os.close(fd)
except (AttributeError, OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(10 if found else 0)
' "$leader_pid" "$leader_start" "$expected_pgid" "$token" "$signal_number"
}

WORKER_RETENTION_ACTIVE=0
WORKER_RETENTION_CONTROL=
WORKER_RETENTION_PENDING=

worker_main() {
  local root=$1 phase=$2 index=$3 token=$4 gate_path=$5 parent_pid=$6
  local pid pgid owner_template gate_mode child_registered
  local start_ns end_ns runtime peak_mib peak_bytes rc=0 monitor_rc=0 gpu_status
  local job_seed expected_seed option_index seed_count
  local audit_payload audit_fields audit_ready producers_done
  local memory_status retention_mode disk_status
  local -a command=() retention_sftp_option=()
  role=; key=; owner_json=; gpu=; gpu_uuid=; run_dir=; started=0
  worker_index=$index; auditor_owner=
  FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
  job_pid=; job_start=; job_pgid=; monitor_pid=; peak_file=; command_file=
  child_gate_root=; child_gate=; child_gate_pending=; child_token=
  child_control=; child_control_pending=
  child_released=0; PENDING_CHILD_SIGNAL=0; WORKER_RETENTION_ACTIVE=0
  payload=; failure_marked=0; job_log_open=0
  if [[ ${FORMAL_INTERNAL_TOKEN:-} != "$token" ]]; then
    die 'invalid internal worker token'
    return 1
  fi
  FORMAL_ROOT=$root
  pid=$BASHPID
  if ! pgid=$(proc_pgid "$pid"); then return 1; fi
  if [[ $pgid -ne $pid ]]; then
    die 'worker is not an owned process-group leader'
    return 1
  fi
  role="$phase-worker-$index"

  worker_mark_failure() {
    local kind=$1 failure_rc=$2
    [[ -n $key && $failure_rc -ne 0 && $failure_marked -eq 0 ]] || return 0
    payload=$(marker_payload "$kind" "$role" "$failure_rc" "$key")
    mark_once FAILED_JOB "$payload" >/dev/null 2>&1 || true
    failure_marked=1
  }
  worker_begin_retention_control() {
    local control_root=${gate_path%/*} control_mode
    WORKER_RETENTION_CONTROL="$control_root/retention-active"
    WORKER_RETENTION_PENDING="$control_root/.retention-active"
    [[ $control_root == "$FORMAL_TEMP_ROOT"/formal-worker-gates.* &&
       -d $control_root && ! -L $control_root ]] || return 1
    control_mode=$(stat -Lc '%a' -- "$control_root") || return 1
    [[ $control_mode == 700 && ! -e $WORKER_RETENTION_CONTROL &&
       ! -L $WORKER_RETENTION_CONTROL && ! -e $WORKER_RETENTION_PENDING &&
       ! -L $WORKER_RETENTION_PENDING ]] || return 1
    if ! (set -o noclobber; umask 077; : > "$WORKER_RETENTION_PENDING"); then
      return 1
    fi
    chmod 0600 -- "$WORKER_RETENTION_PENDING" || return 1
    WORKER_RETENTION_ACTIVE=1
  }
  worker_install_retention_control() {
    local payload
    [[ $WORKER_RETENTION_ACTIVE -eq 1 &&
       -n $WORKER_RETENTION_CONTROL && -n $WORKER_RETENTION_PENDING &&
       -f $WORKER_RETENTION_PENDING && ! -L $WORKER_RETENTION_PENDING &&
       ! -e $WORKER_RETENTION_CONTROL && ! -L $WORKER_RETENTION_CONTROL ]] || return 1
    payload=$("$REVIEWED_PYTHON" -c '
import json, sys
owner = json.loads(sys.argv[2])
print(json.dumps({"kind": "formal_retention_active", "owner": owner,
                  "spec_key": sys.argv[1]}, sort_keys=True,
                 separators=(",", ":"), allow_nan=False))
' "$key" "$owner_template") || return 1
    printf '%s\n' "$payload" > "$WORKER_RETENTION_PENDING" || return 1
    mv -T -- "$WORKER_RETENTION_PENDING" "$WORKER_RETENTION_CONTROL" || return 1
  }
  worker_clear_retention_control() {
    [[ -n $WORKER_RETENTION_CONTROL && -n $WORKER_RETENTION_PENDING &&
       ! -L $WORKER_RETENTION_CONTROL && ! -L $WORKER_RETENTION_PENDING ]] || return 1
    if [[ -f $WORKER_RETENTION_CONTROL && ! -e $WORKER_RETENTION_PENDING ]]; then
      rm -f -- "$WORKER_RETENTION_CONTROL" || return 1
    elif [[ -f $WORKER_RETENTION_PENDING && ! -e $WORKER_RETENTION_CONTROL ]]; then
      rm -f -- "$WORKER_RETENTION_PENDING" || return 1
    else
      return 1
    fi
    WORKER_RETENTION_CONTROL=; WORKER_RETENTION_PENDING=
  }
  child_environment_owned() {
    local current_start current_pgid
    [[ -n $job_pid && -n $job_start && -n $job_pgid &&
       -n $child_token ]] || return 1
    current_start=$(proc_start "$job_pid" 2>/dev/null) || return 1
    current_pgid=$(proc_pgid_from_stat "$job_pid" 2>/dev/null) || return 1
    [[ $current_start == "$job_start" && $current_pgid == "$job_pgid" &&
       $job_pgid == "$job_pid" ]] || return 1
    "$REVIEWED_PYTHON" -c '
import pathlib, sys
pid, token = sys.argv[1], sys.argv[2]
items = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
raise SystemExit(0 if b"FORMAL_CHILD_TOKEN=" + token.encode() in items else 1)
' "$job_pid" "$child_token" >/dev/null 2>&1
  }
  child_release_identity_valid() {
    local current_pgid
    child_environment_owned || return 1
    current_pgid=$(proc_pgid "$job_pid" 2>/dev/null) || return 1
    [[ $current_pgid == "$job_pgid" ]] || return 1
    "$REVIEWED_PYTHON" -c '
import pathlib, sys
pid, gate, token = sys.argv[1:]
parts = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
required = (b"formal-child-wrapper", gate.encode(), token.encode())
raise SystemExit(0 if all(value in parts for value in required) else 1)
' "$job_pid" "$child_gate" "$child_token" >/dev/null 2>&1
  }
  cleanup_child_gate() {
    [[ -n $child_gate_root ]] || return 0
    if [[ $child_gate_root == "$FORMAL_TEMP_ROOT"/formal-child-gates.* &&
          -d $child_gate_root && ! -L $child_gate_root ]]; then
      rm -f -- "$child_gate_root/gate" "$child_gate_root/.gate"
      rmdir -- "$child_gate_root" 2>/dev/null || true
    fi
    if [[ -n $child_control &&
          $child_control == "$FORMAL_TEMP_ROOT"/formal-worker-gates.*/child-[012] ]]; then
      rm -f -- "$child_control" "${child_control%/*}/.${child_control##*/}"
    fi
    child_gate_root=; child_gate=; child_gate_pending=; child_token=
    child_control=; child_control_pending=
    child_released=0
  }
  worker_stop_child() {
    local deadline status
    [[ -n $job_pid ]] || return 0
    [[ -n $job_start && -n $job_pgid && -n $child_token ]] || return 1
    signal_verified_child_group \
      "$job_pid" "$job_start" "$job_pgid" "$child_token" 15
    status=$?
    [[ $status -eq 0 || $status -eq 10 ]] || return 1
    deadline=$((SECONDS + CHILD_TERM_GRACE_SECONDS))
    while [[ $SECONDS -lt $deadline ]]; do
      signal_verified_child_group \
        "$job_pid" "$job_start" "$job_pgid" "$child_token" 0
      status=$?
      [[ $status -ne 0 ]] || break
      [[ $status -eq 10 ]] || return 1
      sleep 0.05
    done
    if [[ $status -eq 10 ]]; then
      signal_verified_child_group \
        "$job_pid" "$job_start" "$job_pgid" "$child_token" 9
      status=$?
      [[ $status -eq 0 || $status -eq 10 ]] || return 1
      deadline=$((SECONDS + CHILD_CLEANUP_OVERHEAD_SECONDS))
      while [[ $SECONDS -lt $deadline ]]; do
        signal_verified_child_group \
          "$job_pid" "$job_start" "$job_pgid" "$child_token" 0
        status=$?
        [[ $status -ne 0 ]] || break
        [[ $status -eq 10 ]] || return 1
        sleep 0.05
      done
    fi
    [[ $status -eq 0 ]] || return 1
    wait "$job_pid" 2>/dev/null || true
    job_pid=; job_start=; job_pgid=
  }
  worker_cleanup() {
    local child_cleanup_ok=1
    if [[ -n $job_pid ]]; then worker_stop_child || child_cleanup_ok=0; fi
    if [[ -n $monitor_pid ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      monitor_pid=
    fi
    if [[ $job_log_open -eq 1 ]]; then
      exec 9>&-
      job_log_open=0
    fi
    if [[ -n $FORMAL_CLAIMED_GPU && -n $owner_json ]]; then
      if "$REVIEWED_PYTHON" "$DRIVER" gpu-release --root "$FORMAL_ROOT" \
          --physical-gpu "$FORMAL_CLAIMED_GPU" \
          --owner-json "$owner_json" >/dev/null 2>&1; then
        FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
      fi
    fi
    if [[ $worker_index != 2 && -n $key && $started -eq 0 && -n $owner_json ]]; then
      "$REVIEWED_PYTHON" "$DRIVER" release-claim --root "$FORMAL_ROOT" \
        --spec "$key" --owner-json "$owner_json" >/dev/null 2>&1 || true
    fi
    [[ -z $peak_file ]] || rm -f -- "$peak_file"
    [[ -z $command_file ]] || rm -f -- "$command_file"
    if [[ $child_cleanup_ok -eq 1 ]]; then cleanup_child_gate; fi
    if [[ $worker_index == 2 && -n $auditor_owner && $WORKER_RETENTION_ACTIVE -eq 0 &&
          ( -e $FORMAL_ROOT/FAILED_JOB || -e $FORMAL_ROOT/FORMAL_STOPPED ) ]]; then
      "$REVIEWED_PYTHON" "$DRIVER" cancel-audit --root "$FORMAL_ROOT" \
        --owner-json "$auditor_owner" >/dev/null 2>&1 || true
    fi
  }
  worker_spawn_child() {
    local child_stderr=${1:-9}
    child_token=$("$REVIEWED_PYTHON" -c \
      'import secrets; print(secrets.token_hex(32))') || return 1
    child_gate_root=$(mktemp -d \
      "$FORMAL_TEMP_ROOT/formal-child-gates.XXXXXXXX") || return 1
    chmod 0700 -- "$child_gate_root" || return 1
    child_gate="$child_gate_root/gate"
    child_gate_pending="$child_gate_root/.gate"
    if ! (set -o noclobber; printf '%s\n' "$child_token" \
        > "$child_gate_pending"); then
      return 1
    fi
    chmod 0600 -- "$child_gate_pending" || return 1

    PENDING_CHILD_SIGNAL=0
    trap 'PENDING_CHILD_SIGNAL=130' INT
    trap 'PENDING_CHILD_SIGNAL=143' TERM
    trap 'PENDING_CHILD_SIGNAL=129' HUP
    FORMAL_CHILD_TOKEN=$child_token CUDA_VISIBLE_DEVICES=$gpu \
      setsid bash -c '
set -uo pipefail
gate=$1
expected=$2
parent_pid=$3
shift 3
trap "exit 130" INT
trap "exit 143" TERM
trap "exit 129" HUP
deadline=$((SECONDS + 12))
while [[ ! -e $gate && ! -L $gate ]]; do
  kill -0 "$parent_pid" 2>/dev/null || exit 143
  [[ $SECONDS -lt $deadline ]] || exit 124
  sleep 0.01
done
[[ -f $gate && ! -L $gate ]] || exit 125
[[ $(stat -Lc "%a" -- "$gate") == 600 ]] || exit 125
actual=$(<"$gate")
[[ $actual == "$expected" ]] || exit 125
exec "$@"
' formal-child-wrapper "$child_gate" "$child_token" "$pid" \
      "${command[@]}" >&9 2>&"$child_stderr" &
    job_pid=$!
    job_pgid=$job_pid
    for _ in {1..100}; do
      job_start=$(proc_start "$job_pid" 2>/dev/null || true)
      [[ -n $job_start ]] && break
      sleep 0.01
    done
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    if [[ $PENDING_CHILD_SIGNAL -ne 0 ]]; then
      exit "$PENDING_CHILD_SIGNAL"
    fi
    [[ -n $job_start ]] || return 1
    child_registered=0
    for _ in {1..100}; do
      if child_release_identity_valid; then
        child_registered=1
        break
      fi
      kill -0 "$job_pid" 2>/dev/null || break
      sleep 0.01
    done
    [[ $child_registered -eq 1 ]] || return 1
    if ! (set -o noclobber; printf '%s\t%s\t%s\t%s\t%s\n' \
        "$job_pid" "$job_start" "$job_pgid" "$child_token" "$child_gate" \
        > "$child_control_pending"); then
      return 1
    fi
    chmod 0600 -- "$child_control_pending" || return 1
    mv -T -- "$child_control_pending" "$child_control" || return 1
    mv -T -- "$child_gate_pending" "$child_gate" || return 1
    child_released=1
  }
  worker_on_exit() {
    local exit_rc=$?
    trap - EXIT
    # A sibling failure may send TERM while this worker is already cleaning up.
    trap '' INT TERM HUP
    if [[ $exit_rc -ne 0 ]]; then
      if [[ $WORKER_RETENTION_ACTIVE -eq 1 ]]; then
        worker_mark_failure failed_retention "$exit_rc"
      else
        worker_mark_failure failed_setup "$exit_rc"
      fi
    fi
    worker_cleanup
    exit "$exit_rc"
  }
  trap worker_on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  if [[ $gate_path != "$FORMAL_TEMP_ROOT"/formal-worker-gates.*/gate-[012] ]]; then
    return 1
  fi
  while [[ ! -e $gate_path && ! -L $gate_path ]]; do
    [[ ! -e $FORMAL_ROOT/FORMAL_STOPPED ]] || return 143
    kill -0 "$parent_pid" 2>/dev/null || return 143
    sleep 0.02
  done
  [[ -f $gate_path && ! -L $gate_path ]] || return 1
  gate_mode=$(stat -Lc '%a' -- "$gate_path") || return 1
  [[ $gate_mode == 600 ]] || return 1
  owner_template=$(<"$gate_path") || return 1
  [[ -n $owner_template ]] || return 1
  if [[ $index == 2 ]]; then auditor_owner=$owner_template; fi
  [[ ! -e $FORMAL_ROOT/FORMAL_STOPPED ]] || return 143
  check_frozen_state || return 1
  "$REVIEWED_PYTHON" "$DRIVER" check --root "$FORMAL_ROOT" >/dev/null || return 1

  while :; do
    key=; owner_json=; gpu=; gpu_uuid=; run_dir=; started=0
    job_seed=; expected_seed=; option_index=0; seed_count=0
    failure_marked=0; peak_file=; command_file=; command=()
    child_gate_root=; child_gate=; child_gate_pending=; child_token=
    child_control="${gate_path%/*}/child-$index"
    child_control_pending="${gate_path%/*}/.child-$index"
    child_released=0
    check_frozen_state || return 1
    if retention_enabled; then
      pilot_parent_running "$parent_pid" || return 143
    fi
    if [[ $index == 2 ]]; then
      # Snapshot before selection so null cannot race the final handoff.
      producers_done=0
      if [[ -e ${gate_path%/*}/producers-done || -L ${gate_path%/*}/producers-done ]]; then
        [[ -f ${gate_path%/*}/producers-done && ! -L ${gate_path%/*}/producers-done &&
           $(stat -Lc '%a' -- "${gate_path%/*}/producers-done") == 600 &&
           $(<"${gate_path%/*}/producers-done") == "$token" ]] || return 1
        producers_done=1
      fi
      if retention_enabled; then
        worker_begin_retention_control || return 1
      fi
      audit_payload=$("$REVIEWED_PYTHON" "$DRIVER" next-audit --root "$FORMAL_ROOT" \
        --phase "$phase" --owner-json "$owner_template") || return 1
      if retention_enabled; then
        pilot_parent_running "$parent_pid" || return 143
      fi
      if [[ $audit_payload == null ]]; then
        if retention_enabled; then
          worker_clear_retention_control || return 1
          WORKER_RETENTION_ACTIVE=0
        fi
        audit_ready=$("$REVIEWED_PYTHON" "$DRIVER" audit-phase-ready \
          --root "$FORMAL_ROOT" --phase "$phase") || return 1
        [[ $audit_ready != true ]] || return 0
        [[ $audit_ready == false && $producers_done -eq 0 ]] || return 1
        sleep 0.1
        continue
      fi
      audit_fields=$("$REVIEWED_PYTHON" -c '
import json, pathlib, re, sys
payload = json.loads(sys.argv[1])
root = pathlib.Path(sys.argv[2])
if type(payload) is not dict or set(payload) != {"spec_key", "seed", "physical_gpu", "run_dir"}:
    raise SystemExit("invalid audit payload fields")
key, seed, gpu, run = (payload[name] for name in ("spec_key", "seed", "physical_gpu", "run_dir"))
if (type(key) is not str or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", key)
        or type(seed) is not int
        or seed not in (42, 43, 44) or key.rsplit(":", 1)[-1] != str(seed)
        or type(gpu) is not int or gpu not in (0, 1) or type(run) is not str
        or any(ord(c) < 32 or ord(c) == 127 for c in run)):
    raise SystemExit("invalid audit payload types or identity")
path = pathlib.Path(run)
if not path.is_dir() or path.is_symlink() or path.resolve() != path or path.parent != root / "runs":
    raise SystemExit("invalid audit run directory")
print(key, seed, gpu, run, sep="\t")
' "$audit_payload" "$FORMAL_ROOT") || return 1
      IFS=$'\t' read -r key job_seed gpu run_dir <<< "$audit_fields"
      started=1
      if retention_enabled; then
        worker_install_retention_control || return 1
        pilot_parent_running "$parent_pid" || return 143
      fi
      # Reuse strict registered spec/run-name validation without execution.
      "$REVIEWED_PYTHON" "$DRIVER" command --spec "$key" \
        --run-dir "$run_dir" --format nul >/dev/null || return 1
      export PYTHONHASHSEED=$job_seed CUDA_VISIBLE_DEVICES=$gpu
      command=("$REVIEWED_PYTHON" "$DRIVER" audit-run --root "$FORMAL_ROOT"
               --spec-key "$key" --run-dir "$run_dir")
      exec 9>/dev/null || return 1
      job_log_open=1
      worker_spawn_child 2 || return 1
      wait "$job_pid"; rc=$?
      worker_stop_child || return 1
      exec 9>&-
      job_log_open=0
      cleanup_child_gate
      if [[ $rc -ne 0 ]]; then
        if retention_enabled; then
          worker_clear_retention_control || return 1
          WORKER_RETENTION_ACTIVE=0
        fi
        worker_mark_failure failed_audit "$rc"
        return "$rc"
      fi
      if retention_enabled; then
        pilot_parent_running "$parent_pid" || return 143
        if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-full-matrix ||
              ${VFCL_EXPERIMENT_PROFILE:-formal} == single-dataset-verified-continuation-v1 ||
              ${VFCL_EXPERIMENT_PROFILE:-formal} == single-method-formal-v1 ]]; then
          retention_sftp_option=(--terminate-blocking-sftp)
        fi
        for retention_mode in --dry-run --apply; do
          pilot_parent_running "$parent_pid" || return 143
          check_frozen_state || return 1
          if "$REVIEWED_PYTHON" "$RETENTION" --root "$FORMAL_ROOT" \
              --worktree "$WORKTREE" --expected-head "$FORMAL_FROZEN_HEAD" \
              --spec-key "$key" --owner-json "$owner_template" \
              "${retention_sftp_option[@]}" "$retention_mode"; then
            rc=0
          else
            rc=$?
            worker_mark_failure failed_retention "$rc"
            return "$rc"
          fi
        done
        pilot_parent_running "$parent_pid" || return 143
        check_frozen_state || return 1
      fi
      "$REVIEWED_PYTHON" "$DRIVER" complete-audit --root "$FORMAL_ROOT" \
        --spec "$key" --owner-json "$owner_template" >/dev/null || return 1
      if retention_enabled; then
        worker_clear_retention_control || return 1
      fi
      WORKER_RETENTION_ACTIVE=0
      key=
      continue
    fi
    while :; do
      pilot_memory_ready && recovery_resources_ready; memory_status=$?
      pilot_parent_running "$parent_pid" || return 143
      disk_ready; disk_status=$?
      if [[ $disk_status -eq 2 ]]; then
        mark_once FAILED_JOB "$(marker_payload failed_setup "$role" 1)" \
          >/dev/null 2>&1 || true
        return 1
      fi
      pilot_parent_running "$parent_pid" || return 143
      [[ $memory_status -ne 0 || $disk_status -ne 0 ]] || break
      check_frozen_state || return 1
      audit_ready=$("$REVIEWED_PYTHON" "$DRIVER" audit-phase-ready \
        --root "$FORMAL_ROOT" --phase "$phase") || return 1
      pilot_parent_running "$parent_pid" || return 143
      [[ $audit_ready != true ]] || return 0
      [[ $audit_ready == false ]] || return 1
      sleep 1
    done
    key=$("$REVIEWED_PYTHON" "$DRIVER" claim --root "$FORMAL_ROOT" \
      --phase "$phase" --owner-json "$owner_template" --pipeline --format text)
    rc=$?
    if [[ $rc -eq 75 ]]; then
      [[ -z $key ]] || return 1
      sleep 0.1
      continue
    fi
    [[ $rc -eq 0 ]] || return "$rc"
    if [[ -z $key ]]; then
      if [[ ${VFCL_EXPERIMENT_PROFILE:-formal} == full-public-matrix ]]; then
        # Reservations remain owner-bound until CPU audit and retention settle.
        audit_ready=$("$REVIEWED_PYTHON" "$DRIVER" audit-phase-ready \
          --root "$FORMAL_ROOT" --phase "$phase") || return 1
        [[ $audit_ready != true ]] || return 0
        [[ $audit_ready == false ]] || return 1
        sleep 0.1
        continue
      fi
      return 0
    fi
    owner_json=$("$REVIEWED_PYTHON" "$DRIVER" claim-owner \
      --root "$FORMAL_ROOT" --spec "$key") || return 1
    while :; do
      claim_gpu_for_worker "$index" "$owner_json"; gpu_status=$?
      [[ $gpu_status -ne 0 ]] || break
      [[ $gpu_status -ne 2 ]] || return 95
      if retention_enabled; then
        "$REVIEWED_PYTHON" "$DRIVER" release-claim --root "$FORMAL_ROOT" \
          --spec "$key" --owner-json "$owner_json" >/dev/null || return 1
        key=; owner_json=
        sleep 0.1
        continue 2
      fi
      [[ ! -e $FORMAL_ROOT/FORMAL_STOPPED ]] || return 1
      sleep 0.1
    done
    gpu=$FORMAL_CLAIMED_GPU
    gpu_uuid=$FORMAL_CLAIMED_GPU_UUID
    pilot_memory_ready && recovery_resources_ready; memory_status=$?
    pilot_parent_running "$parent_pid" || return 143
    if [[ $memory_status -ne 0 ]]; then
      "$REVIEWED_PYTHON" "$DRIVER" gpu-release --root "$FORMAL_ROOT" \
        --physical-gpu "$gpu" --owner-json "$owner_json" >/dev/null || return 1
      FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
      "$REVIEWED_PYTHON" "$DRIVER" release-claim --root "$FORMAL_ROOT" \
        --spec "$key" --owner-json "$owner_json" >/dev/null || return 1
      key=; owner_json=; gpu=; gpu_uuid=
      continue
    fi
    run_dir=$("$REVIEWED_PYTHON" "$DRIVER" prepare-run --root "$FORMAL_ROOT" \
      --spec "$key" --owner-json "$owner_json") || return 1
    started=1
    command_file=$(mktemp "${TMPDIR:-/tmp}/formal-command.XXXXXXXX") || return 96
    "$REVIEWED_PYTHON" "$DRIVER" command --spec "$key" \
      --run-dir "$run_dir" --format nul > "$command_file" || return 1
    mapfile -d '' -t command < "$command_file" || return 1
    [[ ${#command[@]} -gt 0 ]] || return 1
    expected_seed=${key##*:}
    for ((option_index=0; option_index<${#command[@]}; option_index++)); do
      if [[ ${command[$option_index]} == --seed ]]; then
        seed_count=$((seed_count + 1))
        ((option_index + 1 < ${#command[@]})) || return 1
        job_seed=${command[$((option_index + 1))]}
      fi
    done
    [[ $seed_count -eq 1 && $expected_seed =~ ^(42|43|44)$ &&
       $job_seed == "$expected_seed" ]] || return 1
    export PYTHONHASHSEED=$job_seed
    export CUDA_VISIBLE_DEVICES=$gpu
    rm -f -- "$command_file" || return 1
    command_file=
    peak_file=$(mktemp "${TMPDIR:-/tmp}/formal-peak.XXXXXXXX") || return 96
    if ! (set -o noclobber; : > "$run_dir/job.log") 2>/dev/null; then
      return 1
    fi
    exec 9>> "$run_dir/job.log" || return 1
    job_log_open=1
    start_ns=$("$REVIEWED_PYTHON" -c \
      'import time; print(time.monotonic_ns())') || return 1

    worker_spawn_child || return 1
    if child_environment_owned; then
      monitor_peak_mib "$job_pid" "$gpu_uuid" > "$peak_file" &
      monitor_pid=$!
      wait "$job_pid"; rc=$?
      worker_stop_child || return 1
      wait "$monitor_pid" 2>/dev/null; monitor_rc=$?
      monitor_pid=
    elif [[ -z $(proc_state "$job_pid" 2>/dev/null || true) ||
            $(proc_state "$job_pid" 2>/dev/null || true) == Z ]]; then
      wait "$job_pid"; rc=$?
      worker_stop_child || return 1
      monitor_rc=0
      printf '0\n' > "$peak_file"
    else
      return 1
    fi
    exec 9>&-
    job_log_open=0
    cleanup_child_gate
    end_ns=$("$REVIEWED_PYTHON" -c \
      'import time; print(time.monotonic_ns())') || return 1
    chmod 0444 -- "$run_dir/job.log" || return 1
    if [[ $rc -ne 0 ]]; then
      worker_mark_failure failed_job "$rc"
      return "$rc"
    fi
    if [[ $monitor_rc -ne 0 ]]; then
      worker_mark_failure failed_monitor "$monitor_rc"
      return "$monitor_rc"
    fi
    peak_mib=$(<"$peak_file")
    [[ $peak_mib =~ ^[0-9]+$ ]] || return 1
    peak_bytes=$((peak_mib * 1024 * 1024))
    runtime=$("$REVIEWED_PYTHON" -c \
      'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1_000_000_000.0)' \
      "$start_ns" "$end_ns") || return 1
    "$REVIEWED_PYTHON" "$DRIVER" resource-record --root "$FORMAL_ROOT" \
      --spec "$key" --run-dir "$run_dir" --runtime-seconds "$runtime" \
      --peak-gpu-memory-bytes "$peak_bytes" >/dev/null
    rc=$?
    if [[ $rc -ne 0 ]]; then
      worker_mark_failure failed_resource "$rc"
      return "$rc"
    fi
    "$REVIEWED_PYTHON" "$DRIVER" queue-audit --root "$FORMAL_ROOT" \
      --spec "$key" --physical-gpu "$gpu" --owner-json "$owner_json" >/dev/null || return 1
    "$REVIEWED_PYTHON" "$DRIVER" gpu-release --root "$FORMAL_ROOT" \
      --physical-gpu "$gpu" --owner-json "$owner_json" >/dev/null || return 1
    FORMAL_CLAIMED_GPU=; FORMAL_CLAIMED_GPU_UUID=
    gpu=; key=; owner_json=; started=0
    rm -f -- "$peak_file"; peak_file=
  done
}

declare -a WORKER_PID=() WORKER_START=() WORKER_PGID=() WORKER_ACTIVE=()
declare -a WORKER_TERM_SENT=() WORKER_GATE=() WORKER_OWNER_PENDING=()
declare -a WORKER_CHILD_CONTROL=() WORKER_OWNER=()
WORKER_GATE_ROOT=
PENDING_SPAWN_SIGNAL=0
PARENT_CHILD_PID=; PARENT_CHILD_START=; PARENT_CHILD_PGID=
PARENT_CHILD_TOKEN=; PARENT_CHILD_GATE=

registered_group() {
  local index=$1 pid=${WORKER_PID[$index]} current_start current_pgid expected_start
  expected_start=${WORKER_START[$index]:-}
  [[ -n $expected_start ]] || return 1
  current_start=$(proc_start "$pid" 2>/dev/null) || return 1
  current_pgid=$(proc_pgid "$pid" 2>/dev/null) || return 1
  [[ $current_start == "$expected_start" &&
     $current_pgid == "${WORKER_PGID[$index]}" &&
     $current_pgid == "$pid" ]]
}

parse_child_control() {
  local index=$1 path mode extra
  path=${WORKER_CHILD_CONTROL[$index]:-}
  [[ -n $path &&
     $path == "$FORMAL_TEMP_ROOT"/formal-worker-gates.*/child-[012] &&
     -f $path && ! -L $path ]] || return 1
  mode=$(stat -Lc '%a' -- "$path") || return 1
  [[ $mode == 600 ]] || return 1
  extra=
  IFS=$'\t' read -r PARENT_CHILD_PID PARENT_CHILD_START \
    PARENT_CHILD_PGID PARENT_CHILD_TOKEN PARENT_CHILD_GATE extra < "$path" ||
    return 1
  [[ -z $extra && $PARENT_CHILD_PID =~ ^[1-9][0-9]*$ &&
     $PARENT_CHILD_START =~ ^[0-9]+$ &&
     $PARENT_CHILD_PGID == "$PARENT_CHILD_PID" &&
     $PARENT_CHILD_TOKEN =~ ^[0-9a-f]{64}$ &&
     $PARENT_CHILD_GATE == "$FORMAL_TEMP_ROOT"/formal-child-gates.*/gate ]] || return 1
}

parent_cleanup_child() {
  local index=$1 deadline status control_dir child_root
  parse_child_control "$index" || return 1
  signal_verified_child_group "$PARENT_CHILD_PID" "$PARENT_CHILD_START" \
    "$PARENT_CHILD_PGID" "$PARENT_CHILD_TOKEN" 9
  status=$?
  [[ $status -eq 0 || $status -eq 10 ]] || return 1
  if [[ $status -eq 10 ]]; then
    deadline=$((SECONDS + PARENT_CHILD_KILL_WAIT_SECONDS))
    while [[ $SECONDS -lt $deadline ]]; do
      signal_verified_child_group "$PARENT_CHILD_PID" "$PARENT_CHILD_START" \
        "$PARENT_CHILD_PGID" "$PARENT_CHILD_TOKEN" 0
      status=$?
      [[ $status -ne 0 ]] || break
      [[ $status -eq 10 ]] || return 1
      sleep 0.05
    done
  fi
  if [[ $status -eq 0 ]]; then
    child_root=${PARENT_CHILD_GATE%/*}
    if [[ $child_root == "$FORMAL_TEMP_ROOT"/formal-child-gates.* &&
          -d $child_root && ! -L $child_root ]]; then
      rm -f -- "$child_root/gate" "$child_root/.gate"
      rmdir -- "$child_root" 2>/dev/null || true
    fi
    control_dir=${WORKER_CHILD_CONTROL[$index]%/*}
    rm -f -- "${WORKER_CHILD_CONTROL[$index]}" \
      "$control_dir/.child-$index"
    return 0
  fi
  return 1
}

cleanup_gate_root() {
  [[ -n $WORKER_GATE_ROOT ]] || return 0
  if [[ $WORKER_GATE_ROOT == "$FORMAL_TEMP_ROOT"/formal-worker-gates.* &&
        -d $WORKER_GATE_ROOT && ! -L $WORKER_GATE_ROOT ]]; then
    rm -f -- "$WORKER_GATE_ROOT/gate-0" "$WORKER_GATE_ROOT/gate-1" \
      "$WORKER_GATE_ROOT/.owner-0" "$WORKER_GATE_ROOT/.owner-1" \
      "$WORKER_GATE_ROOT/.child-0" "$WORKER_GATE_ROOT/.child-1" \
      "$WORKER_GATE_ROOT/gate-2" "$WORKER_GATE_ROOT/.owner-2" \
      "$WORKER_GATE_ROOT/.child-2" \
      "$WORKER_GATE_ROOT/producers-done" "$WORKER_GATE_ROOT/.producers-done" \
      "$WORKER_GATE_ROOT/retention-active" "$WORKER_GATE_ROOT/.retention-active"
    rmdir -- "$WORKER_GATE_ROOT" 2>/dev/null || return 1
  fi
  WORKER_GATE_ROOT=; WORKER_RETENTION_CONTROL=; WORKER_RETENTION_PENDING=
}

parent_cleanup_gpu() {
  local index=$1 candidates physical_gpu candidate_owner
  [[ $index != 2 && -n ${WORKER_OWNER[$index]:-} ]] || return 0
  check_root_identity || return 1
  candidates=$("$REVIEWED_PYTHON" -c '
import contextlib, json, os, re, stat, sys
root, raw_template = sys.argv[1:]
template = json.loads(raw_template)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
for gpu in (0, 1):
    try:
        with contextlib.ExitStack() as stack:
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            for name in ("gpu_claims", f"gpu-{gpu}"):
                fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                stack.callback(os.close, fd)
            fd = os.open("owner.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            stream = stack.enter_context(os.fdopen(fd, "rb"))
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                continue
            content = stream.read(65537)
            if len(content) > 65536:
                continue
            installed = json.loads(content)
        if type(installed) is not dict or canonical(installed).encode() + b"\n" != content:
            continue
        job = installed.get("job")
        if type(job) is not str or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", job):
            continue
        owner = {**template, "job": job}
        expected = {**owner, "kind": "formal_gpu_claim", "physical_gpu": gpu}
        if canonical(installed) == canonical(expected):
            print(gpu, canonical(owner), sep="\t")
    except (OSError, ValueError, TypeError):
        continue
' "$FORMAL_ROOT" "${WORKER_OWNER[$index]}") || return 1
  [[ -n $candidates ]] || return 0
  while IFS=$'\t' read -r physical_gpu candidate_owner; do
    # The driver rechecks terminal authority and the exact current slot under lock.
    timeout --kill-after=1s "${CHILD_CLEANUP_OVERHEAD_SECONDS}s" \
      "$REVIEWED_PYTHON" "$DRIVER" gpu-release --root "$FORMAL_ROOT" \
      --physical-gpu "$physical_gpu" --owner-json "$candidate_owner" >/dev/null 2>&1 || return 1
  done <<< "$candidates"
}

terminate_registered() {
  local index pid deadline retention_state
  for index in "${!WORKER_PID[@]}"; do
    [[ ${WORKER_ACTIVE[$index]:-0} -ne 0 ]] || continue
    pid=${WORKER_PID[$index]}
    if registered_group "$index"; then
      kill -TERM -- "-${WORKER_PGID[$index]}" 2>/dev/null || true
      WORKER_TERM_SENT[$index]=1
    fi
  done
  deadline=$((SECONDS + PARENT_WORKER_GRACE_SECONDS))
  for index in "${!WORKER_PID[@]}"; do
    [[ ${WORKER_ACTIVE[$index]:-0} -ne 0 ]] || continue
    pid=${WORKER_PID[$index]}
    while [[ $SECONDS -lt $deadline ]] && kill -0 "$pid" 2>/dev/null; do
      if [[ ${WORKER_TERM_SENT[$index]:-0} -eq 0 ]] && registered_group "$index"; then
        kill -TERM -- "-${WORKER_PGID[$index]}" 2>/dev/null || true
        WORKER_TERM_SENT[$index]=1
      fi
      [[ $(proc_state "$pid" 2>/dev/null || true) != Z ]] || break
      sleep 0.05
    done
    if kill -0 "$pid" 2>/dev/null; then
      parent_cleanup_child "$index" || true
      if registered_group "$index"; then
        kill -KILL -- "-${WORKER_PGID[$index]}" 2>/dev/null || true
      fi
    fi
    wait "$pid" 2>/dev/null || true
    parent_cleanup_child "$index" || true
    WORKER_ACTIVE[$index]=0
  done
  for index in 0 1; do
    parent_cleanup_gpu "$index" || die "owned GPU cleanup failed for worker $index"
  done
  if [[ -n ${WORKER_OWNER[2]:-} ]]; then
    if retention_control_state; then
      retention_state=0
    else
      retention_state=$?
    fi
    if [[ $retention_state -eq 1 ]]; then
      "$REVIEWED_PYTHON" "$DRIVER" cancel-audit --root "$FORMAL_ROOT" \
        --owner-json "${WORKER_OWNER[2]}" >/dev/null 2>&1 || true
    fi
  fi
}

install_stopped() {
  local code=$1 phase=$2
  mark_once FORMAL_STOPPED "$(marker_payload formal_stopped "$phase" "$code")" \
    >/dev/null 2>&1 || true
}

run_phase() {
  local phase=$1 token=$2 index pid pgid start state rc remaining=3
  local owner_json pending parent_pid=$BASHPID
  FORMAL_PARENT_START=$(proc_start "$parent_pid") || return 1
  export FORMAL_PARENT_START
  WORKER_PID=(); WORKER_START=(); WORKER_PGID=(); WORKER_ACTIVE=()
  WORKER_TERM_SENT=(); WORKER_GATE=(); WORKER_OWNER_PENDING=()
  WORKER_CHILD_CONTROL=(); WORKER_OWNER=()
  WORKER_GATE_ROOT=$(mktemp -d \
    "$FORMAL_TEMP_ROOT/formal-worker-gates.XXXXXXXX") || return 1
  chmod 0700 -- "$WORKER_GATE_ROOT" || {
    cleanup_gate_root
    return 1
  }
  WORKER_RETENTION_CONTROL="$WORKER_GATE_ROOT/retention-active"
  WORKER_RETENTION_PENDING="$WORKER_GATE_ROOT/.retention-active"
  for index in 0 1 2; do
    start=; pgid=
    WORKER_GATE[$index]="$WORKER_GATE_ROOT/gate-$index"
    WORKER_OWNER_PENDING[$index]="$WORKER_GATE_ROOT/.owner-$index"
    WORKER_CHILD_CONTROL[$index]="$WORKER_GATE_ROOT/child-$index"
    PENDING_SPAWN_SIGNAL=0
    trap 'PENDING_SPAWN_SIGNAL=130' INT
    trap 'PENDING_SPAWN_SIGNAL=143' TERM
    trap 'PENDING_SPAWN_SIGNAL=129' HUP
    FORMAL_INTERNAL_TOKEN=$token setsid "$SCRIPT_REAL" --internal-worker \
      "$FORMAL_ROOT" "$phase" "$index" "$token" \
      "${WORKER_GATE[$index]}" "$parent_pid" &
    pid=$!
    WORKER_PID[$index]=$pid
    WORKER_PGID[$index]=$pid
    WORKER_ACTIVE[$index]=2
    WORKER_TERM_SENT[$index]=0
    trap 'main_signal 130' INT
    trap 'main_signal 143' TERM
    trap 'main_signal 129' HUP
    if [[ $PENDING_SPAWN_SIGNAL -ne 0 ]]; then
      main_signal "$PENDING_SPAWN_SIGNAL"
    fi
    for _ in {1..100}; do
      start=$(proc_start "$pid" 2>/dev/null || true)
      pgid=$(proc_pgid "$pid" 2>/dev/null || true)
      [[ -n $start && $pgid == "$pid" ]] && break
      sleep 0.01
    done
    [[ -n ${start:-} && $pgid == "$pid" ]] || {
      install_stopped 1 "$phase-registration"
      terminate_registered
      cleanup_gate_root
      die 'worker process group could not be registered'
      return 1
    }
    WORKER_START[$index]=$start
    WORKER_ACTIVE[$index]=1
  done
  for index in 0 1 2; do
    owner_json=$("$REVIEWED_PYTHON" "$DRIVER" owner --root "$FORMAL_ROOT" \
      --phase "$phase" --launcher-token "$token" \
      --worker-role "$phase-worker-$index" --pid "${WORKER_PID[$index]}" \
      --pgid "${WORKER_PGID[$index]}") || {
        install_stopped 1 "$phase-registration"
        terminate_registered
        cleanup_gate_root
        return 1
      }
    WORKER_OWNER[$index]=$owner_json
    if ! (set -o noclobber; printf '%s\n' "$owner_json" \
        > "${WORKER_OWNER_PENDING[$index]}"); then
      install_stopped 1 "$phase-registration"
      terminate_registered
      cleanup_gate_root
      return 1
    fi
    chmod 0600 -- "${WORKER_OWNER_PENDING[$index]}" || {
      install_stopped 1 "$phase-registration"
      terminate_registered
      cleanup_gate_root
      return 1
    }
  done
  check_frozen_state || {
    install_stopped 1 "$phase-pre-release"
    terminate_registered
    cleanup_gate_root
    return 1
  }
  "$REVIEWED_PYTHON" "$DRIVER" check --root "$FORMAL_ROOT" >/dev/null || {
    install_stopped 1 "$phase-pre-release"
    terminate_registered
    cleanup_gate_root
    return 1
  }
  for index in 0 1 2; do
    if [[ -e $FORMAL_ROOT/FORMAL_STOPPED ]] || ! registered_group "$index"; then
      install_stopped 1 "$phase-pre-release"
      terminate_registered
      cleanup_gate_root
      return 1
    fi
  done
  for index in 0 1 2; do
    mv -T -- "${WORKER_OWNER_PENDING[$index]}" "${WORKER_GATE[$index]}" || {
      install_stopped 1 "$phase-gate-release"
      terminate_registered
      cleanup_gate_root
      return 1
    }
  done
  while [[ $remaining -gt 0 ]]; do
    for index in 0 1 2; do
      [[ ${WORKER_ACTIVE[$index]} -eq 1 ]] || continue
      pid=${WORKER_PID[$index]}
      state=$(proc_state "$pid" 2>/dev/null || true)
      if [[ -z $state || $state == Z ]]; then
        wait "$pid"; rc=$?
        remaining=$((remaining - 1))
        if [[ $rc -ne 0 ]]; then
          install_stopped "$rc" "$phase"
          terminate_registered
          cleanup_gate_root
          return "$rc"
        fi
        check_frozen_state || {
          install_stopped 1 "$phase-post-wait"
          terminate_registered
          cleanup_gate_root
          return 1
        }
        "$REVIEWED_PYTHON" "$DRIVER" check --root "$FORMAL_ROOT" \
          >/dev/null || {
            install_stopped 1 "$phase-post-wait"
            terminate_registered
            cleanup_gate_root
            return 1
          }
        WORKER_ACTIVE[$index]=0
      fi
    done
    if [[ ${WORKER_ACTIVE[0]} -eq 0 && ${WORKER_ACTIVE[1]} -eq 0 &&
          ! -e $WORKER_GATE_ROOT/producers-done ]]; then
      (set -o noclobber; umask 077; printf '%s\n' "$token" \
        > "$WORKER_GATE_ROOT/.producers-done") &&
        mv -T -- "$WORKER_GATE_ROOT/.producers-done" "$WORKER_GATE_ROOT/producers-done" || {
          install_stopped 1 "$phase-producers-done"
          terminate_registered
          cleanup_gate_root
          return 1
        }
    fi
    [[ $remaining -eq 0 ]] || sleep 0.05
  done
  cleanup_gate_root || return 1
  return 0
}

main_signal() {
  local code=$1
  trap - INT TERM HUP
  install_stopped "$code" signal
  terminate_registered
  cleanup_gate_root
  exit "$code"
}

SMOKE_CHILD_PID=; SMOKE_CHILD_START=; SMOKE_CHILD_PGID=
SMOKE_CHILD_TOKEN=; SMOKE_COMMAND_FILE=; PENDING_SMOKE_SIGNAL=0
SMOKE_ENUM_PID=; SMOKE_ENUM_START=; SMOKE_ENUM_PGID=; SMOKE_ENUM_TOKEN=
SMOKE_JOBS_FILE=; SMOKE_SELECTED_GPU=; SMOKE_SELECTED_GPU_UUID=

select_smoke_gpu() {
  local preferred=$1 alternate gpu status
  [[ $preferred == 0 || $preferred == 1 ]] || return 2
  alternate=$((1 - preferred))
  for gpu in "$preferred" "$alternate"; do
    gpu_eligible "$gpu"; status=$?
    [[ $status -ne 2 ]] || return 2
    if [[ $status -eq 0 ]]; then
      SMOKE_SELECTED_GPU=$gpu
      SMOKE_SELECTED_GPU_UUID=$FORMAL_GPU_UUID
      return 0
    fi
  done
  return 1
}

wait_for_smoke_gpu() {
  local preferred=$1 status
  while :; do
    check_frozen_state || return 3
    "$REVIEWED_PYTHON" "$DRIVER" smoke-check \
      --root "$FORMAL_ROOT" >/dev/null || return 3
    select_smoke_gpu "$preferred"; status=$?
    [[ $status -eq 1 ]] || return "$status"
    sleep "$SMOKE_GPU_WAIT_SECONDS"
  done
}

smoke_child_owned() {
  local current_start current_pgid
  [[ -n $SMOKE_CHILD_PID && -n $SMOKE_CHILD_START &&
     -n $SMOKE_CHILD_PGID && -n $SMOKE_CHILD_TOKEN ]] || return 1
  current_start=$(proc_start "$SMOKE_CHILD_PID" 2>/dev/null) || return 1
  current_pgid=$(proc_pgid_from_stat "$SMOKE_CHILD_PID" 2>/dev/null) || return 1
  [[ $current_start == "$SMOKE_CHILD_START" &&
     $current_pgid == "$SMOKE_CHILD_PGID" &&
     $SMOKE_CHILD_PGID == "$SMOKE_CHILD_PID" ]] || return 1
  "$REVIEWED_PYTHON" -c '
import pathlib, sys
pid, token = sys.argv[1:]
items = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
raise SystemExit(0 if b"FORMAL_CHILD_TOKEN=" + token.encode() in items else 1)
' "$SMOKE_CHILD_PID" "$SMOKE_CHILD_TOKEN" >/dev/null 2>&1
}

smoke_stop_child() {
  local deadline state
  [[ -n $SMOKE_CHILD_PID ]] || return 0
  if smoke_child_owned; then
    kill -TERM -- "-$SMOKE_CHILD_PGID" 2>/dev/null || true
  fi
  deadline=$((SECONDS + CHILD_TERM_GRACE_SECONDS))
  while [[ $SECONDS -lt $deadline ]] && kill -0 "$SMOKE_CHILD_PID" 2>/dev/null; do
    state=$(proc_state "$SMOKE_CHILD_PID" 2>/dev/null || true)
    [[ $state != Z ]] || break
    sleep 0.05
  done
  if kill -0 "$SMOKE_CHILD_PID" 2>/dev/null && smoke_child_owned; then
    kill -KILL -- "-$SMOKE_CHILD_PGID" 2>/dev/null || true
  fi
  wait "$SMOKE_CHILD_PID" 2>/dev/null || true
  SMOKE_CHILD_PID=; SMOKE_CHILD_START=; SMOKE_CHILD_PGID=; SMOKE_CHILD_TOKEN=
}

smoke_enumerator_owned() {
  local current_start current_pgid
  [[ -n $SMOKE_ENUM_PID && -n $SMOKE_ENUM_START &&
     -n $SMOKE_ENUM_PGID && -n $SMOKE_ENUM_TOKEN ]] || return 1
  current_start=$(proc_start "$SMOKE_ENUM_PID" 2>/dev/null) || return 1
  current_pgid=$(proc_pgid_from_stat "$SMOKE_ENUM_PID" 2>/dev/null) || return 1
  [[ $current_start == "$SMOKE_ENUM_START" &&
     $current_pgid == "$SMOKE_ENUM_PGID" &&
     $SMOKE_ENUM_PGID == "$SMOKE_ENUM_PID" ]] || return 1
  "$REVIEWED_PYTHON" -c '
import pathlib, sys
pid, token = sys.argv[1:]
items = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
raise SystemExit(0 if b"FORMAL_CHILD_TOKEN=" + token.encode() in items else 1)
' "$SMOKE_ENUM_PID" "$SMOKE_ENUM_TOKEN" >/dev/null 2>&1
}

smoke_stop_enumerator() {
  local deadline state
  [[ -n $SMOKE_ENUM_PID ]] || return 0
  if smoke_enumerator_owned; then
    kill -TERM -- "-$SMOKE_ENUM_PGID" 2>/dev/null || true
  else
    kill -TERM "$SMOKE_ENUM_PID" 2>/dev/null || true
  fi
  deadline=$((SECONDS + CHILD_TERM_GRACE_SECONDS))
  while [[ $SECONDS -lt $deadline ]] && kill -0 "$SMOKE_ENUM_PID" 2>/dev/null; do
    state=$(proc_state "$SMOKE_ENUM_PID" 2>/dev/null || true)
    [[ $state != Z ]] || break
    sleep 0.05
  done
  if kill -0 "$SMOKE_ENUM_PID" 2>/dev/null; then
    if smoke_enumerator_owned; then
      kill -KILL -- "-$SMOKE_ENUM_PGID" 2>/dev/null || true
    else
      kill -KILL "$SMOKE_ENUM_PID" 2>/dev/null || true
    fi
  fi
  wait "$SMOKE_ENUM_PID" 2>/dev/null || true
  SMOKE_ENUM_PID=; SMOKE_ENUM_START=; SMOKE_ENUM_PGID=; SMOKE_ENUM_TOKEN=
}

smoke_mark_failure() {
  local code=$1
  "$REVIEWED_PYTHON" "$DRIVER" smoke-mark --root "$FORMAL_ROOT" \
    --name SMOKE_FAILED --kind failed --exit-code "$code" >/dev/null 2>&1 || true
  "$REVIEWED_PYTHON" "$DRIVER" smoke-mark --root "$FORMAL_ROOT" \
    --name SMOKE_STOPPED --kind stopped --exit-code "$code" >/dev/null 2>&1 || true
}

smoke_signal() {
  local code=$1
  trap - INT TERM HUP
  smoke_stop_enumerator
  smoke_stop_child
  [[ -z $SMOKE_COMMAND_FILE ]] || rm -f -- "$SMOKE_COMMAND_FILE"
  SMOKE_COMMAND_FILE=
  [[ -z $SMOKE_JOBS_FILE ]] || rm -f -- "$SMOKE_JOBS_FILE"
  SMOKE_JOBS_FILE=
  smoke_mark_failure "$code"
  exit "$code"
}

run_smoke() {
  local token=$1 job run_dir log preferred gpu status rc start pgid state index=0
  local expected_jobs=4
  [[ ${VFCL_EXPERIMENT_PROFILE:-formal} != full-public-matrix ]] || expected_jobs=42
  local -a jobs=() command=()
  SMOKE_JOBS_FILE=$(mktemp "${TMPDIR:-/tmp}/formal-smoke-jobs.XXXXXXXX") || {
    smoke_mark_failure 96
    return 96
  }
  SMOKE_ENUM_TOKEN=$(
    "$REVIEWED_PYTHON" -c 'import secrets; print(secrets.token_hex(32))'
  ) || {
    rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
    smoke_mark_failure 1
    return 1
  }
  PENDING_SMOKE_SIGNAL=0
  trap 'PENDING_SMOKE_SIGNAL=130' INT
  trap 'PENDING_SMOKE_SIGNAL=143' TERM
  trap 'PENDING_SMOKE_SIGNAL=129' HUP
  FORMAL_CHILD_TOKEN=$SMOKE_ENUM_TOKEN setsid \
    "$REVIEWED_PYTHON" "$DRIVER" smoke-jobs --root "$FORMAL_ROOT" \
    > "$SMOKE_JOBS_FILE" &
  SMOKE_ENUM_PID=$!
  SMOKE_ENUM_PGID=$SMOKE_ENUM_PID
  start=; pgid=
  for _ in {1..100}; do
    start=$(proc_start "$SMOKE_ENUM_PID" 2>/dev/null || true)
    pgid=$(proc_pgid "$SMOKE_ENUM_PID" 2>/dev/null || true)
    [[ -n $start && $pgid == "$SMOKE_ENUM_PID" ]] && break
    state=$(proc_state "$SMOKE_ENUM_PID" 2>/dev/null || true)
    [[ -z $state || $state == Z ]] && break
    sleep 0.01
  done
  SMOKE_ENUM_START=$start
  trap 'smoke_signal 130' INT
  trap 'smoke_signal 143' TERM
  trap 'smoke_signal 129' HUP
  if [[ $PENDING_SMOKE_SIGNAL -ne 0 ]]; then
    smoke_signal "$PENDING_SMOKE_SIGNAL"
  fi
  if [[ -n $start && $pgid != "$SMOKE_ENUM_PID" ]]; then
    smoke_stop_enumerator
    rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
    smoke_mark_failure 1
    return 1
  fi
  if [[ -n $start ]] && ! smoke_enumerator_owned; then
    state=$(proc_state "$SMOKE_ENUM_PID" 2>/dev/null || true)
    if [[ -n $state && $state != Z ]]; then
      smoke_stop_enumerator
      rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
      smoke_mark_failure 1
      return 1
    fi
  fi
  wait "$SMOKE_ENUM_PID"; rc=$?
  SMOKE_ENUM_PID=; SMOKE_ENUM_START=; SMOKE_ENUM_PGID=; SMOKE_ENUM_TOKEN=
  if [[ $rc -ne 0 ]]; then
    rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
    smoke_mark_failure "$rc"
    return "$rc"
  fi
  mapfile -t jobs < "$SMOKE_JOBS_FILE" || {
    rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
    smoke_mark_failure 1
    return 1
  }
  rm -f -- "$SMOKE_JOBS_FILE"; SMOKE_JOBS_FILE=
  if [[ ${#jobs[@]} -ne $expected_jobs ]]; then
    if [[ $expected_jobs -eq 4 ]]; then
      die 'generated smoke plan must contain exactly four jobs'
    else
      die 'generated smoke plan must contain exactly 42 jobs'
    fi
    smoke_mark_failure 1
    return 1
  fi
  if [[ $(printf '%s\n' "${jobs[@]}" | sort -u | wc -l) -ne $expected_jobs ]]; then
    die 'generated smoke jobs must be unique'
    smoke_mark_failure 1
    return 1
  fi
  for job in "${jobs[@]}"; do
    [[ $job =~ ^[a-z0-9-]+$ ]] || {
      smoke_mark_failure 1
      return 1
    }
    preferred=$((index % 2))
    run_dir="$FORMAL_ROOT/runs/$job"
    log="$FORMAL_ROOT/logs/$job.log"
    check_frozen_state || {
      smoke_mark_failure 1
      return 1
    }
    "$REVIEWED_PYTHON" "$DRIVER" smoke-check --root "$FORMAL_ROOT" >/dev/null || {
      smoke_mark_failure 1
      return 1
    }
    SMOKE_COMMAND_FILE=$(mktemp "${TMPDIR:-/tmp}/formal-smoke-command.XXXXXXXX") || {
      smoke_mark_failure 96
      return 96
    }
    "$REVIEWED_PYTHON" "$DRIVER" smoke-command --root "$FORMAL_ROOT" \
      --job "$job" --run-dir "$run_dir" --format nul > "$SMOKE_COMMAND_FILE" || {
      rm -f -- "$SMOKE_COMMAND_FILE"; SMOKE_COMMAND_FILE=
      smoke_mark_failure 1
      return 1
    }
    mapfile -d '' -t command < "$SMOKE_COMMAND_FILE" || {
      rm -f -- "$SMOKE_COMMAND_FILE"; SMOKE_COMMAND_FILE=
      smoke_mark_failure 1
      return 1
    }
    rm -f -- "$SMOKE_COMMAND_FILE"; SMOKE_COMMAND_FILE=
    [[ ${#command[@]} -gt 0 ]] || {
      smoke_mark_failure 1
      return 1
    }
    while :; do
      wait_for_smoke_gpu "$preferred"; status=$?
      if [[ $status -ne 0 ]]; then
        rc=$([[ $status -eq 2 ]] && printf '95' || printf '1')
        smoke_mark_failure "$rc"
        return "$rc"
      fi
      gpu=$SMOKE_SELECTED_GPU
      check_frozen_state || {
        smoke_mark_failure 1
        return 1
      }
      "$REVIEWED_PYTHON" "$DRIVER" smoke-check \
        --root "$FORMAL_ROOT" >/dev/null || {
        smoke_mark_failure 1
        return 1
      }
      gpu_eligible "$gpu"; status=$?
      if [[ $status -eq 0 ]]; then
        SMOKE_SELECTED_GPU_UUID=$FORMAL_GPU_UUID
        break
      fi
      if [[ $status -eq 2 ]]; then
        smoke_mark_failure 95
        return 95
      fi
    done
    "$REVIEWED_PYTHON" "$DRIVER" smoke-begin --root "$FORMAL_ROOT" \
      --job "$job" --physical-gpu "$gpu" >/dev/null || {
      smoke_mark_failure 1
      return 1
    }
    if ! (set -o noclobber; : > "$log") 2>/dev/null; then
      smoke_mark_failure 1
      return 1
    fi
    SMOKE_CHILD_TOKEN=$(
      "$REVIEWED_PYTHON" -c 'import secrets; print(secrets.token_hex(32))'
    ) || {
      smoke_mark_failure 1
      return 1
    }
    PENDING_SMOKE_SIGNAL=0
    trap 'PENDING_SMOKE_SIGNAL=130' INT
    trap 'PENDING_SMOKE_SIGNAL=143' TERM
    trap 'PENDING_SMOKE_SIGNAL=129' HUP
    FORMAL_CHILD_TOKEN=$SMOKE_CHILD_TOKEN CUDA_VISIBLE_DEVICES=$gpu \
      setsid "${command[@]}" > "$log" 2>&1 &
    SMOKE_CHILD_PID=$!
    SMOKE_CHILD_PGID=$SMOKE_CHILD_PID
    start=; pgid=
    for _ in {1..100}; do
      start=$(proc_start "$SMOKE_CHILD_PID" 2>/dev/null || true)
      pgid=$(proc_pgid "$SMOKE_CHILD_PID" 2>/dev/null || true)
      [[ -n $start && $pgid == "$SMOKE_CHILD_PID" ]] && break
      state=$(proc_state "$SMOKE_CHILD_PID" 2>/dev/null || true)
      if [[ $state == Z ]] ||
          { [[ -z $state ]] && ! kill -0 "$SMOKE_CHILD_PID" 2>/dev/null; }; then
        break
      fi
      sleep 0.01
    done
    SMOKE_CHILD_START=$start
    trap 'smoke_signal 130' INT
    trap 'smoke_signal 143' TERM
    trap 'smoke_signal 129' HUP
    if [[ $PENDING_SMOKE_SIGNAL -ne 0 ]]; then
      smoke_signal "$PENDING_SMOKE_SIGNAL"
    fi
    if [[ -z $start ]]; then
      state=$(proc_state "$SMOKE_CHILD_PID" 2>/dev/null || true)
      if [[ $state != Z ]] &&
          { [[ -n $state ]] || kill -0 "$SMOKE_CHILD_PID" 2>/dev/null; }; then
        kill -TERM "$SMOKE_CHILD_PID" 2>/dev/null || true
        wait "$SMOKE_CHILD_PID" 2>/dev/null || true
        SMOKE_CHILD_PID=; SMOKE_CHILD_PGID=; SMOKE_CHILD_TOKEN=
        chmod 0444 -- "$log" 2>/dev/null || true
        smoke_mark_failure 1
        return 1
      fi
      wait "$SMOKE_CHILD_PID"; rc=$?
    else
      if [[ $pgid != "$SMOKE_CHILD_PID" ]] || ! smoke_child_owned; then
        state=$(proc_state "$SMOKE_CHILD_PID" 2>/dev/null || true)
        if [[ $state != Z ]] &&
            { [[ -n $state ]] || kill -0 "$SMOKE_CHILD_PID" 2>/dev/null; }; then
          smoke_stop_child
          chmod 0444 -- "$log" 2>/dev/null || true
          smoke_mark_failure 1
          return 1
        fi
      fi
      wait "$SMOKE_CHILD_PID"; rc=$?
    fi
    SMOKE_CHILD_PID=; SMOKE_CHILD_START=; SMOKE_CHILD_PGID=; SMOKE_CHILD_TOKEN=
    chmod 0444 -- "$log" || {
      smoke_mark_failure 1
      return 1
    }
    if [[ $rc -ne 0 ]]; then
      smoke_mark_failure "$rc"
      return "$rc"
    fi
    "$REVIEWED_PYTHON" "$DRIVER" smoke-record --root "$FORMAL_ROOT" \
      --job "$job" --run-dir "$run_dir" >/dev/null || {
      smoke_mark_failure 1
      return 1
    }
    index=$((index + 1))
  done
  check_frozen_state || {
    smoke_mark_failure 1
    return 1
  }
  "$REVIEWED_PYTHON" "$DRIVER" audit-smoke --root "$FORMAL_ROOT" >/dev/null || {
    smoke_mark_failure 1
    return 1
  }
}

main() {
  local mode root token
  if [[ ${1:-} == --internal-worker ]]; then
    if [[ $# -ne 7 ]]; then
      die 'invalid internal worker arguments'
      return 64
    fi
    worker_main "$2" "$3" "$4" "$5" "$6" "$7"
    return
  fi
  if [[ ${1:-} == --smoke ]]; then
    [[ $# -eq 2 ]] || { die 'usage: launcher --smoke ROOT'; return 64; }
    root=$2
    smoke_preflight "$root" || return
    export CUBLAS_WORKSPACE_CONFIG=':4096:8'
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export PYTHONHASHSEED=42
    token=$("$REVIEWED_PYTHON" -c 'import secrets; print(secrets.token_hex(32))') || return
    trap 'smoke_signal 130' INT
    trap 'smoke_signal 143' TERM
    trap 'smoke_signal 129' HUP
    run_smoke "$token"
    return
  elif [[ ${1:-} == --check ]]; then
    [[ $# -eq 2 ]] || { die 'usage: launcher --check ROOT'; return 64; }
    mode=check; root=$2
  elif [[ $# -eq 1 && ${1:-} != --* ]]; then
    mode=launch; root=$1
  else
    die 'usage: launcher [--check|--smoke] ROOT'
    return 64
  fi
  preflight "$mode" "$root" || return
  [[ $mode == check ]] && return 0
  export CUBLAS_WORKSPACE_CONFIG=':4096:8'
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export PYTHONHASHSEED=42
  token=$("$REVIEWED_PYTHON" -c 'import secrets; print(secrets.token_hex(32))') || return
  export FORMAL_INTERNAL_TOKEN=$token
  trap 'main_signal 130' INT
  trap 'main_signal 143' TERM
  trap 'main_signal 129' HUP
  check_frozen_state || return
  run_phase formal "$token" || return
  if [[ $FORMAL_FROZEN_PROFILE == full-public-matrix ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once FULL_MATRIX_PHASE_SUCCESS \
      "$(marker_payload full_matrix_phase_success launcher 0)" || return
    mark_once FULL_MATRIX_EXECUTION_SUCCESS \
      "$(marker_payload full_matrix_execution_success launcher 0)"
    return
  fi
  if [[ $FORMAL_FROZEN_PROFILE == single-dataset-full-matrix ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once DATASET_PHASE_SUCCESS \
      "$(marker_payload dataset_phase_success launcher 0)" || return
    mark_once DATASET_EXECUTION_SUCCESS \
      "$(marker_payload dataset_execution_success launcher 0)"
    return
  fi
  if [[ $FORMAL_FROZEN_PROFILE == single-dataset-verified-continuation-v1 ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once DATASET_CONTINUATION_PHASE_SUCCESS \
      "$(marker_payload dataset_continuation_phase_success launcher 0)" || return
    mark_once DATASET_CONTINUATION_SUCCESS \
      "$(marker_payload dataset_continuation_success launcher 0)"
    return
  fi
  if [[ $FORMAL_FROZEN_PROFILE == single-method-formal-v1 ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once METHOD_SHARD_PHASE_SUCCESS \
      "$(marker_payload method_shard_phase_success launcher 0)" || return
    mark_once METHOD_SHARD_SUCCESS \
      "$(marker_payload method_shard_success launcher 0)"
    return
  fi
  if [[ $FORMAL_FROZEN_PROFILE == seed42-adaptive-recovery ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once RECOVERY_PHASE_SUCCESS \
      "$(marker_payload recovery_phase_success launcher 0)" || return
    mark_once RECOVERY_EXECUTION_SUCCESS \
      "$(marker_payload recovery_execution_success launcher 0)"
    return
  fi
  if [[ $FORMAL_FROZEN_PROFILE == seed42-pilot ]]; then
    "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
      install_stopped 1 finalize
      return 1
    }
    mark_once PILOT_PHASE_SUCCESS \
      "$(marker_payload pilot_phase_success launcher 0)" || return
    mark_once PILOT_EXECUTION_SUCCESS \
      "$(marker_payload pilot_execution_success launcher 0)"
    return
  fi
  "$REVIEWED_PYTHON" "$DRIVER" phase-ready --root "$FORMAL_ROOT" \
    --phase explanation >/dev/null || {
      install_stopped 1 formal_barrier
      return 1
    }
  mark_once FORMAL_PHASE_SUCCESS \
    "$(marker_payload formal_phase_success launcher 0)" || return
  run_phase explanation "$token" || return
  mark_once EXPLANATION_PHASE_SUCCESS \
    "$(marker_payload explanation_phase_success launcher 0)" || return
  "$REVIEWED_PYTHON" "$DRIVER" finalize --root "$FORMAL_ROOT" >/dev/null || {
    install_stopped 1 finalize
    return 1
  }
  mark_once FORMAL_EXECUTION_SUCCESS \
    "$(marker_payload formal_execution_success launcher 0)" || return
}

main "$@"
