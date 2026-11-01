#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
    printf 'Usage: %s PYTHON COMMAND [ARGS...]\n' "$0" >&2
    exit 2
fi
python_bin=$1
shift
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=42 "$python_bin" "$script_dir/frozen_ablation_fim_diagnosis.py" "$@"
