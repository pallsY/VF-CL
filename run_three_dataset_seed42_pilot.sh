#!/usr/bin/env bash
set -euo pipefail
if [[ ${VFCL_EXPERIMENT_PROFILE+x} ]]; then
  printf '%s\n' 'pilot launcher: external profile override is forbidden' >&2
  exit 64
fi
export VFCL_EXPERIMENT_PROFILE=seed42-pilot
exec "$(dirname -- "$(realpath -e -- "${BASH_SOURCE[0]}")")/run_three_dataset_formal_comparison.sh" "$@"
