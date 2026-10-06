#!/bin/bash
# Targeted gap-fill: Phase-1 (Oracle + P0) for ONLY the unfinished tin cells.
# Mirrors run_fragmentation.sh params exactly; idempotent on aggregated.json,
# so already-complete methods (tin_2p: oracle/finetune_retrain/der_pp_luv) are skipped.
# Does NOT touch Phase 2/3. Runs:
#   tin_2p : proto_evolve:luv, proto_evolve:fedosd
#   tin_4p : oracle + finetune:retrain, der_pp:luv, proto_evolve:luv, proto_evolve:fedosd
set -u
source /home/chase/miniconda3/etc/profile.d/conda.sh
conda activate FedEMoE
cd /home/chase/ct/VF-CUL

P0_METHODS=(
    "finetune:retrain"
    "der_pp:luv"
    "proto_evolve:luv"
    "proto_evolve:fedosd"
)

SEEDS=(--seeds 42,43,44 --device cuda:0
       --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01
       --aggregation sum --model_type resnet18)

TIN=(--data tinyimagenet --num_classes 200 --num_tasks 10 --classes_per_task 20
     --unlearn_after_tasks "3,7" --unlearn_classes "20;100")

run_combo() {
    local cl=$1 ul=$2 dsname=$3 P=$4
    local OUT="./results/frag/${dsname}_${P}p/${cl}_x_${ul}"
    # depth-agnostic: main.py multiseed puts aggregated.json 2 levels deep, oracle 1 level
    if [ -d "$OUT" ] && find "$OUT" -name aggregated.json 2>/dev/null | grep -q .; then
        echo "  SKIP ${dsname}-${P}p :: ${cl} x ${ul} (done)"; return
    fi
    mkdir -p "$OUT"
    echo "===== ${dsname}-${P}p :: ${cl} x ${ul} $(date) ====="
    python -u main.py --cl_method "$cl" --ul_method "$ul" --num_parties "$P" \
                      "${SEEDS[@]}" "${TIN[@]}" --results_dir "$OUT"
}

run_oracle() {
    local dsname=$1 P=$2
    local OUT="./results/frag/${dsname}_${P}p/oracle"
    if [ -d "$OUT" ] && find "$OUT" -name aggregated.json 2>/dev/null | grep -q .; then
        echo "  SKIP ${dsname}-${P}p :: Oracle (done)"; return
    fi
    mkdir -p "$OUT"
    echo "===== ${dsname}-${P}p :: Oracle $(date) ====="
    python -u run_oracle_solo.py --num_parties "$P" "${SEEDS[@]}" "${TIN[@]}" \
                                  --results_dir "$OUT"
}

echo "===== TIN GAP-FILL START $(date) ====="
for cell in "tin 2" "tin 4"; do
    set -- $cell; dsname=$1; P=$2
    run_oracle "$dsname" "$P"
    for m in "${P0_METHODS[@]}"; do
        run_combo "${m%:*}" "${m#*:}" "$dsname" "$P"
    done
done
echo "===== TIN GAP-FILL DONE $(date) ====="
