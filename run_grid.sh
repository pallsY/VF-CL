#!/bin/bash
# Grid-fill: DER++/ER-ACE × {GA,MoDe,FUCRT,FedUP,FedOSD,FedAU,FUDP}
#   × 4 configs (c10_o1, c10_o2, c100_o1, c100_o2) × 3 seeds.
# Plus V-LETO × retrain on C10 O1 (crashed before v4.23 KD-shape fix).
# Idempotent: skips combos whose aggregated.json already exists.
set -u
source /home/chase/miniconda3/etc/profile.d/conda.sh
conda activate FedEMoE
cd /home/chase/ct/VF-CUL

CLs=(der_pp er_ace)
ULs=(gradient_ascent mode fucrt fedup fedosd fedau fudp)

# Common flags (everything except dataset-specific task spec / unlearn spec).
COMMON_C10=( --num_parties 2 --aggregation sum --model_type resnet18
             --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01
             --seeds 42,43,44 --device cuda:0
             --data cifar10 --num_classes 10
             --custom_tasks "0,1|2,3|4,5|6,7|8,9" )
COMMON_C100=( --num_parties 2 --aggregation sum --model_type resnet18
              --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01
              --seeds 42,43,44 --device cuda:0
              --data cifar100 --num_classes 100
              --num_tasks 5 --classes_per_task 20 )

run_combo() {
    local cl=$1 ul=$2 cfg=$3
    shift 3
    local OUT="./results/grid/${cfg}/${cl}_x_${ul}"
    if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then
        echo "  SKIP ${cfg} :: ${cl} x ${ul} (already done)"
        return
    fi
    mkdir -p "$OUT"
    echo "===== ${cfg} :: ${cl} x ${ul} $(date) ====="
    python -u main.py --cl_method "$cl" --ul_method "$ul" "$@" --results_dir "$OUT"
}

# --- 0. Fill the missing V-LETO × retrain cell on C10 O1 ---
run_combo proto_evolve retrain c10_o1 \
    "${COMMON_C10[@]}" --unlearn_after_tasks "1,3" --unlearn_classes "0;5"

# --- 1. Grid fill: 2 CL × 7 UL × 4 configs ---
for cl in "${CLs[@]}"; do
    for ul in "${ULs[@]}"; do
        run_combo "$cl" "$ul" c10_o1 \
            "${COMMON_C10[@]}" --unlearn_after_tasks "1,3" --unlearn_classes "0;5"
        run_combo "$cl" "$ul" c10_o2 \
            "${COMMON_C10[@]}" --unlearn_after_tasks "2,3" --unlearn_classes "1;2,3"
        run_combo "$cl" "$ul" c100_o1 \
            "${COMMON_C100[@]}" --unlearn_after_tasks "1,3" --unlearn_classes "20;60"
        run_combo "$cl" "$ul" c100_o2 \
            "${COMMON_C100[@]}" --unlearn_after_tasks "2,3" --unlearn_classes "0;40,41"
    done
done

echo "===== GRID FILL DONE $(date) ====="
