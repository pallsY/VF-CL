#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    log "ERROR: $*" >&2
    return 1
}

CANONICAL_W=/home/chase/Yangxx/VF-CL/.worktrees/unified-head-consolidation-factorial
CANONICAL_PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python
CANONICAL_REPO=/home/chase/Yangxx/VF-CL
CANONICAL_RESULTS_BASE=/home/chase/Yangxx/VF-CL/results/
W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
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
SOURCE_COMMIT=a60d429c91966c5d42d057c8bd5a380777cc69cb
RESULTS_BASE=$REPO/results/
if [[ "${LAUNCHER_SOURCE_ONLY:-0}" == 1 && -n "${LAUNCHER_TEST_RESULTS_ROOT:-}" ]]; then
    RESULTS_BASE=$LAUNCHER_TEST_RESULTS_ROOT
fi
RESULTS_BASE=${RESULTS_BASE%/}
ROOT=${1:-$RESULTS_BASE/unified_head_consolidation_factorial_seed42_$(date +%Y%m%d_%H%M%S)}
MATRIX_IDENTITY=unified_head_consolidation_factorial:seed42:v1
if [[ "${LAUNCHER_SOURCE_ONLY:-0}" == 1 ]]; then
    VFCL_PROC_ROOT=${VFCL_PROC_ROOT:-/proc}
    POLL_SECONDS=${GPU_POLL_SECONDS:-30}
else
    VFCL_PROC_ROOT=/proc
    POLL_SECONDS=30
fi
WORKER_0_PID=
WORKER_1_PID=
ACTIVE_MAIN_PID=
CLAIM_COMPLETE=10
CLAIM_LIVE=11
CLAIM_UNSAFE=12
declare -A CLAIM_TOKENS=()
LAUNCHER_TOKEN=

safe_target() {
    local target=$1
    [[ ! -L "$target" ]] || die "symlinked target is not allowed: $target"
}

atomic_text() {
    local target=$1 text=$2 temporary
    safe_target "$target" || return 1
    temporary=$(mktemp "${target}.tmp.XXXXXX")
    printf '%s\n' "$text" > "$temporary"
    mv -T -- "$temporary" "$target"
}

prepare_root() {
    local resolved_base resolved_root relative component cursor target existing head recorded identity
    [[ "$ROOT" = /* ]] || die "ROOT must be absolute"
    [[ "${ROOT##*/}" =~ ^unified_head_consolidation_factorial_seed42_[0-9]{8}_[0-9]{6}$ ]] || \
        die "ROOT basename is outside the frozen matrix contract"
    resolved_base=$(realpath -e -- "$RESULTS_BASE") || die "results root is unavailable"
    resolved_root=$(realpath -m -- "$ROOT") || die "ROOT cannot be resolved"
    [[ "$resolved_root" == "$resolved_base"/* ]] || die "ROOT is outside $RESULTS_BASE"

    relative=${ROOT#"$RESULTS_BASE"/}
    cursor=$RESULTS_BASE
    IFS=/ read -ra components <<< "$relative"
    for component in "${components[@]}"; do
        cursor=$cursor/$component
        [[ ! -L "$cursor" ]] || die "symlinked ROOT component is not allowed: $cursor"
    done
    existing=0
    [[ ! -e "$ROOT" ]] || existing=1
    if (( existing )); then
        [[ -d "$ROOT" && ! -L "$ROOT" ]] || die "ROOT is unsafe"
        [[ -f "$ROOT/CODE_COMMIT.txt" && ! -L "$ROOT/CODE_COMMIT.txt" ]] || \
            die "existing ROOT has no immutable code identity"
        [[ -f "$ROOT/MATRIX_IDENTITY.txt" && ! -L "$ROOT/MATRIX_IDENTITY.txt" ]] || \
            die "existing ROOT has no immutable matrix identity"
        identity=$(<"$ROOT/MATRIX_IDENTITY.txt")
        [[ "$identity" == "$MATRIX_IDENTITY" ]] || die "ROOT matrix identity mismatch"
        head=$(git -C "$W" rev-parse HEAD) || return 1
        recorded=$(<"$ROOT/CODE_COMMIT.txt")
        [[ "$recorded" == "$head" ]] || die "ROOT belongs to implementation $recorded"
    else
        mkdir -- "$ROOT"
        head=$(git -C "$W" rev-parse HEAD) || return 1
        [[ "$head" =~ ^[0-9a-f]{40}$ ]] || die "HEAD is not a full commit"
        atomic_text "$ROOT/CODE_COMMIT.txt" "$head"
        atomic_text "$ROOT/MATRIX_IDENTITY.txt" "$MATRIX_IDENTITY"
    fi
    [[ -d "$ROOT" && ! -L "$ROOT" ]] || die "ROOT is unsafe"
    for target in \
        "$ROOT/logs" "$ROOT/claims" "$ROOT/reuse_audit" \
        "$ROOT/runs" "$ROOT/smoke" "$ROOT/formal_report" \
        "$ROOT/logs/launcher.log" \
        "$ROOT/logs/worker_0.log" "$ROOT/logs/worker_1.log" \
        "$ROOT/CODE_COMMIT.txt" "$ROOT/MATRIX_IDENTITY.txt" \
        "$ROOT/required_jobs.json" "$ROOT/SMOKE_SUCCESS" \
        "$ROOT/FAILED_JOB" "$ROOT/STOPPED" \
        "$ROOT/worker_0.pid" "$ROOT/worker_1.pid" "$ROOT/WORKERS_READY" \
        "$ROOT/EXECUTION_SUCCESS" "$ROOT/GATE_SUCCESS" "$ROOT/GATE_FAILED"; do
        safe_target "$target"
    done
    mkdir -p -- "$ROOT/logs" "$ROOT/claims"
}

driver() {
    "$PY" "$W/unified_head_consolidation_factorial.py" "$@"
}

tracked_driver() {
    local status
    setsid "$PY" "$W/unified_head_consolidation_factorial.py" "$@" &
    ACTIVE_MAIN_PID=$!
    if wait "$ACTIVE_MAIN_PID"; then status=0; else status=$?; fi
    ACTIVE_MAIN_PID=
    return "$status"
}

trim() {
    local value=$1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

preflight() {
    local head recorded
    [[ "$W" = "$(git -C "$W" rev-parse --show-toplevel)" ]] || die "wrong worktree"
    [[ -z "$(git -C "$W" status --porcelain)" ]] || die "worktree is not clean"
    git -C "$W" merge-base --is-ancestor "$SOURCE_COMMIT" HEAD || \
        die "HEAD is not a descendant of SOURCE_COMMIT"
    head=$(git -C "$W" rev-parse HEAD)
    [[ "$head" =~ ^[0-9a-f]{40}$ ]] || die "HEAD is not a full commit"
    recorded=$(<"$ROOT/CODE_COMMIT.txt")
    [[ "$recorded" == "$head" ]] || die "ROOT belongs to implementation $recorded"
    [[ "$(<"$ROOT/MATRIX_IDENTITY.txt")" == "$MATRIX_IDENTITY" ]] || \
        die "ROOT matrix identity mismatch"
}

gpu_available() {
    local device=$1 gpu_rows app_rows row index uuid free app_uuid pid cmdline cwd
    local uuid_candidate free_candidate extra
    gpu_rows=$(nvidia-smi \
        --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits 2>/dev/null) || {
        die "GPU $device unavailable: GPU query failed"
        return 1
    }
    app_rows=$(nvidia-smi \
        --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null) || {
        die "GPU $device unavailable: process query failed"
        return 1
    }

    uuid=
    free=
    while IFS= read -r row; do
        [[ -n "$row" ]] || continue
        IFS=, read -r index uuid_candidate free_candidate extra <<< "$row"
        index=$(trim "$index")
        uuid_candidate=$(trim "$uuid_candidate")
        free_candidate=$(trim "$free_candidate")
        [[ -z "${extra:-}" && "$index" =~ ^[0-9]+$ && \
            "$uuid_candidate" =~ ^GPU-[A-Za-z0-9-]+$ && \
            "$free_candidate" =~ ^[0-9]+$ ]] || {
            die "GPU $device unavailable: malformed GPU evidence"
            return 1
        }
        if [[ "$index" == "$device" ]]; then
            uuid=$uuid_candidate
            free=$free_candidate
        fi
    done <<< "$gpu_rows"
    [[ -n "$uuid" && "$free" =~ ^[0-9]+$ ]] || {
        die "GPU $device unavailable: missing GPU evidence"
        return 1
    }
    (( free >= 3500 )) || {
        die "GPU $device unavailable: ${free} MiB free is below 3500 MiB"
        return 1
    }

    while IFS= read -r row; do
        [[ -n "$row" ]] || continue
        IFS=, read -r app_uuid pid extra <<< "$row"
        app_uuid=$(trim "$app_uuid")
        pid=$(trim "$pid")
        [[ -z "${extra:-}" && "$app_uuid" =~ ^GPU-[A-Za-z0-9-]+$ && "$pid" =~ ^[0-9]+$ ]] || {
            die "GPU $device unavailable: malformed process evidence"
            return 1
        }
        [[ "$app_uuid" == "$uuid" ]] || continue
        [[ -r "$VFCL_PROC_ROOT/$pid/cmdline" ]] || {
            die "GPU $device unavailable: missing process evidence for PID $pid"
            return 1
        }
        cmdline=$(tr '\0' ' ' < "$VFCL_PROC_ROOT/$pid/cmdline") || {
            die "GPU $device unavailable: unreadable process evidence for PID $pid"
            return 1
        }
        if [[ "$cmdline" == *"$CANONICAL_REPO/"*main.py* ||
              "$cmdline" == *"$REPO/"*main.py* ]]; then
            die "GPU $device unavailable: active VFCL main.py PID $pid"
            return 1
        fi
        if [[ "$cmdline" == *main.py* ]]; then
            [[ -L "$VFCL_PROC_ROOT/$pid/cwd" ]] || {
                die "GPU $device unavailable: missing cwd evidence for PID $pid"
                return 1
            }
            cwd=$(readlink -f -- "$VFCL_PROC_ROOT/$pid/cwd") || {
                die "GPU $device unavailable: unreadable VFCL cwd evidence for PID $pid"
                return 1
            }
            if [[ "$cwd" == "$CANONICAL_REPO" || "$cwd" == "$CANONICAL_REPO"/* ||
                  "$cwd" == "$REPO" || "$cwd" == "$REPO"/* ]]; then
                die "GPU $device unavailable: active VFCL main.py PID $pid"
                return 1
            fi
        fi
    done <<< "$app_rows"
    log "GPU $device eligible: ${free} MiB free" >&2
}

wait_gpu() {
    local device=$1
    until gpu_available "$device"; do
        log "waiting ${POLL_SECONDS}s for GPU $device"
        sleep "$POLL_SECONDS"
    done
}

wait_any_gpu() {
    while true; do
        if gpu_available 0; then printf '0\n'; return; fi
        if gpu_available 1; then printf '1\n'; return; fi
        log "waiting ${POLL_SECONDS}s for an eligible GPU" >&2
        sleep "$POLL_SECONDS"
    done
}

spec_safe() {
    printf '%s\n' "${1//:/_}"
}

completed_job() {
    local spec=$1 kind=${2:-full} safe section root success record expected_smoke
    safe=$(spec_safe "$spec")
    section=runs
    expected_smoke=false
    if [[ "$kind" == smoke ]]; then
        section=smoke
        expected_smoke=true
    fi
    root=$ROOT/$section/$safe
    success=$root/SUCCESS
    record=$root/record.json
    [[ -f "$success" && ! -L "$success" && -f "$record" && ! -L "$record" ]] || return 1
    "$PY" -c '
import sys
sys.path.insert(0, sys.argv[1])
import unified_head_consolidation_factorial as driver
raise SystemExit(not driver._valid_record(
    sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "true",
))
' "$W" "$root" "$spec" "$ROOT" "$expected_smoke"
}

claim_job() {
    local spec=$1 safe claim lock owner identity pid start token owner_token owner_pid=$BASHPID
    completed_job "$spec" full && return "$CLAIM_COMPLETE"
    safe=$(spec_safe "$spec")
    claim=$ROOT/claims/$safe.claim
    lock=$claim.lock
    safe_target "$claim" || return "$CLAIM_UNSAFE"
    safe_target "$lock" || return "$CLAIM_UNSAFE"
    mkdir -- "$lock" 2>/dev/null || return "$CLAIM_UNSAFE"
    completed_job "$spec" full && {
        rmdir -- "$lock"
        return "$CLAIM_COMPLETE"
    }
    owner=$claim/owner
    safe_target "$owner" || {
        rmdir -- "$lock"
        return "$CLAIM_UNSAFE"
    }
    if [[ -e "$claim" ]]; then
        [[ -d "$claim" && ! -L "$claim" && -f "$owner" ]] || {
            rmdir -- "$lock"
            return "$CLAIM_UNSAFE"
        }
        identity=$(<"$owner")
        IFS=: read -r pid start owner_token <<< "$identity"
        [[ "$pid" =~ ^[0-9]+$ && "$start" =~ ^[0-9]+$ && -n "$owner_token" ]] || {
            rmdir -- "$lock"
            return "$CLAIM_UNSAFE"
        }
        if [[ "$(process_identity "$pid" 2>/dev/null || true)" == "$pid:$start" ]]; then
            log "claim held for $spec by PID $pid"
            rmdir -- "$lock"
            return "$CLAIM_LIVE"
        fi
        rm -f -- "$owner"
        rmdir -- "$claim" 2>/dev/null || {
            rmdir -- "$lock"
            return "$CLAIM_UNSAFE"
        }
    fi
    mkdir -- "$claim" || {
        rmdir -- "$lock"
        return "$CLAIM_UNSAFE"
    }
    token=$(claim_token) || {
        rmdir -- "$claim" "$lock" 2>/dev/null || true
        return "$CLAIM_UNSAFE"
    }
    identity=$(process_identity "$owner_pid") || {
        rmdir -- "$claim" "$lock" 2>/dev/null || true
        return "$CLAIM_UNSAFE"
    }
    atomic_text "$owner" "$identity:$token" || {
        rmdir -- "$claim" "$lock" 2>/dev/null || true
        return "$CLAIM_UNSAFE"
    }
    CLAIM_TOKENS[$safe]=$token
    rmdir -- "$lock"
}

release_claim() {
    local safe claim lock owner token expected actual owner_pid=$BASHPID
    safe=$(spec_safe "$1")
    claim=$ROOT/claims/$safe.claim
    lock=$claim.lock
    token=${CLAIM_TOKENS[$safe]-}
    [[ -n "$token" ]] || return 1
    safe_target "$claim" || return 1
    safe_target "$lock" || return 1
    mkdir -- "$lock" 2>/dev/null || return 1
    owner=$claim/owner
    expected=$(process_identity "$owner_pid"):$token || {
        rmdir -- "$lock"
        return 1
    }
    actual=
    [[ ! -f "$owner" ]] || actual=$(<"$owner")
    if [[ "$actual" == "$expected" ]]; then
        rm -f -- "$owner"
        rmdir -- "$claim" 2>/dev/null || true
        unset 'CLAIM_TOKENS[$safe]'
    else
        rmdir -- "$lock"
        return 1
    fi
    rmdir -- "$lock"
}

process_identity() {
    local pid=$1 start
    [[ -r "/proc/$pid/stat" ]] || return 1
    start=$(awk '{print $22}' "/proc/$pid/stat") || return 1
    [[ "$start" =~ ^[0-9]+$ ]] || return 1
    printf '%s:%s\n' "$pid" "$start"
}

claim_token() {
    local token
    token=$(</proc/sys/kernel/random/uuid) || return 1
    [[ "$token" =~ ^[0-9a-f-]+$ ]] || return 1
    printf '%s\n' "$token"
}

worker_identity() {
    local pid=$1 worker=$2 identity pgid cwd command_hash expected_wrapper
    local -a arguments=()
    identity=$(process_identity "$pid") || return 1
    pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]') || return 1
    [[ "$pgid" == "$pid" ]] || return 1
    [[ -r "/proc/$pid/cmdline" && -L "/proc/$pid/cwd" ]] || return 1
    mapfile -d '' -t arguments < "/proc/$pid/cmdline" || return 1
    expected_wrapper='source "$1"; ROOT=$2; worker_loop '"$worker"' "${@:3}"'
    [[ ${#arguments[@]} -ge 6 \
        && "${arguments[1]}" == -c \
        && "${arguments[2]}" == "$expected_wrapper" \
        && "${arguments[3]}" == bash \
        && "${arguments[4]}" == "$W/run_unified_head_consolidation_factorial.sh" \
        && "${arguments[5]}" == "$ROOT" ]] || return 1
    cwd=$(readlink -f -- "/proc/$pid/cwd") || return 1
    [[ "$cwd" == "$W" ]] || return 1
    command_hash=$(sha256sum "/proc/$pid/cmdline" | awk '{print $1}') || return 1
    [[ "$command_hash" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s:%s:worker_%s:%s\n' "$identity" "$pgid" "$worker" "$command_hash"
}

claim_launcher() {
    local claim=$ROOT/claims/launcher.claim lock=$ROOT/claims/launcher.claim.lock
    local owner identity pid start token owner_token owner_pid=$BASHPID
    safe_target "$claim" || return 1
    safe_target "$lock" || return 1
    mkdir -- "$lock" 2>/dev/null || return 1
    owner=$claim/owner
    safe_target "$owner" || { rmdir -- "$lock"; return 1; }
    if [[ -e "$claim" ]]; then
        [[ -d "$claim" && ! -L "$claim" && -f "$owner" ]] || {
            rmdir -- "$lock"
            return 1
        }
        identity=$(<"$owner")
        IFS=: read -r pid start owner_token <<< "$identity"
        [[ "$pid" =~ ^[0-9]+$ && "$start" =~ ^[0-9]+$ && -n "$owner_token" ]] || {
            rmdir -- "$lock"
            return 1
        }
        if [[ "$(process_identity "$pid" 2>/dev/null || true)" == "$pid:$start" ]]; then
            die "launcher already active for ROOT as PID $pid"
            rmdir -- "$lock"
            return 1
        fi
        rm -f -- "$owner"
        rmdir -- "$claim" 2>/dev/null || { rmdir -- "$lock"; return 1; }
    fi
    mkdir -- "$claim" || { rmdir -- "$lock"; return 1; }
    token=$(claim_token) || { rmdir -- "$claim" "$lock"; return 1; }
    identity=$(process_identity "$owner_pid") || { rmdir -- "$claim" "$lock"; return 1; }
    atomic_text "$owner" "$identity:$token" || {
        rmdir -- "$claim" "$lock" 2>/dev/null || true
        return 1
    }
    LAUNCHER_TOKEN=$token
    rmdir -- "$lock"
}

release_launcher() {
    local claim=$ROOT/claims/launcher.claim lock=$ROOT/claims/launcher.claim.lock
    local owner expected actual owner_pid=$BASHPID
    [[ -n "$LAUNCHER_TOKEN" ]] || return 1
    safe_target "$claim" || return 1
    safe_target "$lock" || return 1
    mkdir -- "$lock" 2>/dev/null || return 1
    owner=$claim/owner
    expected=$(process_identity "$owner_pid"):$LAUNCHER_TOKEN || {
        rmdir -- "$lock"
        return 1
    }
    actual=
    [[ ! -f "$owner" ]] || actual=$(<"$owner")
    if [[ "$actual" != "$expected" ]]; then
        rmdir -- "$lock"
        return 1
    fi
    rm -f -- "$owner"
    rmdir -- "$claim" 2>/dev/null || true
    LAUNCHER_TOKEN=
    rmdir -- "$lock"
}

prepare_workers() {
    local target
    for target in "$ROOT/worker_0.pid" "$ROOT/worker_1.pid" "$ROOT/WORKERS_READY"; do
        safe_target "$target" || return 1
        rm -f -- "$target"
    done
}

terminate_group() {
    local group=$1 attempts=${2:-50} attempt
    [[ "$group" =~ ^[0-9]+$ ]] || return 0
    kill -TERM -- "-$group" 2>/dev/null || true
    for ((attempt=0; attempt<attempts; attempt++)); do
        kill -0 -- "-$group" 2>/dev/null || break
        sleep 0.1
    done
    kill -KILL -- "-$group" 2>/dev/null || true
    wait "$group" 2>/dev/null || true
}

run_smoke() {
    local spec=$1 device
    device=$(wait_any_gpu)
    tracked_driver run-job "$spec" --device "cuda:$device" --matrix-root "$ROOT" --smoke
    completed_job "$spec" smoke || die "smoke validation failed for $spec"
}

run_smokes() {
    run_smoke "cifar100:C:42" || return 1
    run_smoke "cifar100:D:42" || return 1
    safe_target "$ROOT/SMOKE_SUCCESS" || return 1
    touch -- "$ROOT/SMOKE_SUCCESS"
}

load_jobs() {
    local spec count expected worker
    declare -g -a REQUIRED_JOBS WORKER_0_JOBS WORKER_1_JOBS ALL_SPECS
    declare -A allowed=() seen=() partition=()
    mapfile -t REQUIRED_JOBS < <(driver jobs --matrix-root "$ROOT")
    mapfile -t WORKER_0_JOBS < <(driver jobs --matrix-root "$ROOT" --worker 0 --workers 2)
    mapfile -t WORKER_1_JOBS < <(driver jobs --matrix-root "$ROOT" --worker 1 --workers 2)
    mapfile -t ALL_SPECS < <("$PY" -c \
        "import sys; sys.path.insert(0, '$W'); import unified_head_consolidation_factorial as f; print(*f.all_specs(), sep='\\n')")
    expected=$("$PY" -c \
        'import json,sys; print(len(json.load(open(sys.argv[1]))["required_jobs"]))' \
        "$ROOT/required_jobs.json")
    [[ ${#REQUIRED_JOBS[@]} -eq $expected ]] || die "driver jobs count differs from required_jobs.json"
    for spec in "${ALL_SPECS[@]}"; do allowed[$spec]=1; done
    for spec in "${REQUIRED_JOBS[@]}"; do
        [[ "$spec" == *:42 && ${allowed[$spec]+yes} ]] || die "invalid required spec: $spec"
        [[ ! ${seen[$spec]+yes} ]] || die "duplicate required spec: $spec"
        seen[$spec]=1
    done
    for worker in 0 1; do
        if (( worker == 0 )); then local -n jobs_ref=WORKER_0_JOBS; else local -n jobs_ref=WORKER_1_JOBS; fi
        for spec in "${jobs_ref[@]}"; do
            [[ ${seen[$spec]+yes} && ! ${partition[$spec]+yes} ]] || die "invalid worker partition: $spec"
            partition[$spec]=1
        done
        unset -n jobs_ref
    done
    [[ ${#partition[@]} -eq ${#REQUIRED_JOBS[@]} ]] || die "worker partitions are incomplete"
}

record_failure() {
    local spec=$1 worker=$2 status=$3 payload
    payload=$(printf '{"spec":"%s","worker":%d,"exit":%d}' "$spec" "$worker" "$status")
    if mkdir -- "$ROOT/claims/FAILED_JOB.claim" 2>/dev/null; then
        atomic_text "$ROOT/FAILED_JOB" "$payload"
    fi
    safe_target "$ROOT/STOPPED"
    touch -- "$ROOT/STOPPED"
}

terminate_sibling() {
    local worker=$1 sibling_file evidence sibling_pid start pgid role command_hash actual
    sibling_file=$ROOT/worker_$((1-worker)).pid
    [[ -f "$sibling_file" && ! -L "$sibling_file" ]] || return 0
    evidence=$(<"$sibling_file")
    IFS=: read -r sibling_pid start pgid role command_hash <<< "$evidence"
    [[ "$sibling_pid" =~ ^[0-9]+$ && "$start" =~ ^[0-9]+$ \
        && "$pgid" == "$sibling_pid" \
        && "$role" == "worker_$((1-worker))" \
        && "$command_hash" =~ ^[0-9a-f]{64}$ ]] || return 0
    actual=$(worker_identity "$sibling_pid" "$((1-worker))" 2>/dev/null) || return 0
    [[ "$actual" == "$evidence" ]] || return 0
    terminate_group "$pgid"
}

worker_loop() {
    local worker=$1 spec status claim_status
    shift
    ACTIVE_JOB_PID=
    trap 'terminate_group "$ACTIVE_JOB_PID" 10; exit 143' TERM INT
    while [[ ! -f "$ROOT/WORKERS_READY" ]]; do sleep 1; done
    for spec in "$@"; do
        [[ ! -e "$ROOT/STOPPED" ]] || return 1
        if completed_job "$spec" full; then
            log "worker $worker skips complete $spec"
            continue
        fi
        if claim_job "$spec"; then
            claim_status=0
        else
            claim_status=$?
        fi
        case $claim_status in
            0) ;;
            "$CLAIM_COMPLETE")
                log "worker $worker skips strictly complete $spec"
                continue
                ;;
            *)
                record_failure "$spec" "$worker" "$claim_status"
                terminate_sibling "$worker"
                return "$claim_status"
                ;;
        esac
        wait_gpu "$worker"
        set +e
        setsid "$PY" "$W/unified_head_consolidation_factorial.py" run-job \
            "$spec" --device "cuda:$worker" --matrix-root "$ROOT" &
        ACTIVE_JOB_PID=$!
        wait "$ACTIVE_JOB_PID"
        status=$?
        ACTIVE_JOB_PID=
        set -e
        if (( status != 0 )) || ! completed_job "$spec" full; then
            (( status != 0 )) || status=92
            record_failure "$spec" "$worker" "$status"
            release_claim "$spec"
            terminate_sibling "$worker"
            return "$status"
        fi
        release_claim "$spec"
    done
}

validate_required_jobs() {
    local spec
    for spec in "${REQUIRED_JOBS[@]}"; do
        completed_job "$spec" full || {
            die "required job is not strictly complete: $spec"
            return 1
        }
    done
}

terminate_children() {
    local pid
    for pid in "$ACTIVE_MAIN_PID" "$WORKER_0_PID" "$WORKER_1_PID"; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        terminate_group "$pid"
    done
}

summarize_gate() {
    local status
    if tracked_driver summarize --matrix-root "$ROOT"; then status=0; else status=$?; fi
    case $status in
        0) return 0 ;;
        3) return 3 ;;
        *) return "$status" ;;
    esac
}

main() {
    local status0 status1 summary_status
    prepare_root
    claim_launcher
    trap 'terminate_children; release_launcher' EXIT
    prepare_workers
    exec > >(tee -a "$ROOT/logs/launcher.log") 2>&1
    trap 'exit 130' INT TERM
    preflight
    cd "$W"
    log "driver check"
    tracked_driver check
    log "driver audit-reuse"
    tracked_driver audit-reuse --matrix-root "$ROOT"
    run_smokes
    load_jobs

    LAUNCHER_SOURCE_ONLY=1 LAUNCHER_TEST_RESULTS_ROOT= VFCL_PROC_ROOT=/proc \
        GPU_POLL_SECONDS=30 setsid bash -c 'source "$1"; ROOT=$2; worker_loop 0 "${@:3}"' bash \
        "$W/run_unified_head_consolidation_factorial.sh" "$ROOT" \
        "${WORKER_0_JOBS[@]}" > >(tee -a "$ROOT/logs/worker_0.log") 2>&1 &
    WORKER_0_PID=$!
    LAUNCHER_SOURCE_ONLY=1 LAUNCHER_TEST_RESULTS_ROOT= VFCL_PROC_ROOT=/proc \
        GPU_POLL_SECONDS=30 setsid bash -c 'source "$1"; ROOT=$2; worker_loop 1 "${@:3}"' bash \
        "$W/run_unified_head_consolidation_factorial.sh" "$ROOT" \
        "${WORKER_1_JOBS[@]}" > >(tee -a "$ROOT/logs/worker_1.log") 2>&1 &
    WORKER_1_PID=$!
    atomic_text "$ROOT/worker_0.pid" "$(worker_identity "$WORKER_0_PID" 0)"
    atomic_text "$ROOT/worker_1.pid" "$(worker_identity "$WORKER_1_PID" 1)"
    touch -- "$ROOT/WORKERS_READY"
    set +e
    wait "$WORKER_0_PID"; status0=$?
    wait "$WORKER_1_PID"; status1=$?
    set -e
    WORKER_0_PID=
    WORKER_1_PID=
    if (( status0 != 0 || status1 != 0 )) || [[ -e "$ROOT/STOPPED" ]]; then
        return 1
    fi

    validate_required_jobs
    summarize_gate
}

if [[ "${LAUNCHER_SOURCE_ONLY:-0}" != 1 ]]; then
    main "$@"
fi
