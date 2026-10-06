#!/bin/bash
# Table 1 (fragmentation sweep) — PRIORITY-FIRST ordering.
#
# 13 rows total. Sweep #parties on 3 datasets:
#   CIFAR-10/100 : P ∈ {1, 4}     (P=2 reused from results/main_*_o1/)
#   TinyImagenet : P ∈ {1, 2, 4}
# → 9 fragmentation cells per row.
#
# Phase 1 (highest priority, ~5 days): Oracle + 4 P0 methods × 9 cells.
#   After phase 1: a full 5-row Table 1 across all 9 cells.
# Phase 2 (~3 days): 3 P1 methods × 9 cells → 8 rows.
# Phase 3 (~5 days): 5 P2 methods × 9 cells → 13 rows.
# Idempotent on aggregated.json; safe to re-launch.
set -u
source /home/chase/miniconda3/etc/profile.d/conda.sh
conda activate FedEMoE
cd /home/chase/ct/VF-CUL

P0_METHODS=(
    "finetune:retrain"             # lower bound
    "der_pp:luv"                   # best replay × VFL-UL
    "proto_evolve:luv"             # canonical VFL pipeline
    "proto_evolve:fedosd"          # VFL × best-KL UL
)
P1_METHODS=(
    "er_ace:luv"                   # C100-best CL × VFL-UL
    "proto_evolve:gradient_ascent" # GA family
    "proto_evolve:fedup"           # prototype/feature UL family
)
P2_METHODS=(
    "proto_evolve:fucrt"           # distillation UL family
    "proto_evolve:fedau"           # linear UL family
    "proto_evolve:fudp"            # pruning UL family
    "er:luv"                       # basic replay × VFL-UL
    "proto_evolve:mode"            # momentum-degradation UL family
)

SEEDS=(--seeds 42,43,44 --device cuda:0
       --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01
       --aggregation sum --model_type resnet18)

C10=(--data cifar10 --num_classes 10
     --custom_tasks "0,1|2,3|4,5|6,7|8,9"
     --unlearn_after_tasks "1,3" --unlearn_classes "0;5")
C100=(--data cifar100 --num_classes 100 --num_tasks 5 --classes_per_task 20
      --unlearn_after_tasks "1,3" --unlearn_classes "20;60")
TIN=(--data tinyimagenet --num_classes 200 --num_tasks 10 --classes_per_task 20
     --unlearn_after_tasks "3,7" --unlearn_classes "20;100")

# args_for_dataset c10 -> echoes the dataset's flag array
args_for() {
    case "$1" in
        c10)  echo "${C10[@]}";;
        c100) echo "${C100[@]}";;
        tin)  echo "${TIN[@]}";;
    esac
}

run_combo() {
    local cl=$1 ul=$2 dsname=$3 P=$4
    local OUT="./results/frag/${dsname}_${P}p/${cl}_x_${ul}"
    if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then
        echo "  SKIP ${dsname}-${P}p :: ${cl} x ${ul} (done)"; return
    fi
    mkdir -p "$OUT"
    local ds_args
    read -ra ds_args <<< "$(args_for $dsname)"
    echo "===== ${dsname}-${P}p :: ${cl} x ${ul} $(date) ====="
    python -u main.py --cl_method "$cl" --ul_method "$ul" --num_parties "$P" \
                      "${SEEDS[@]}" "${ds_args[@]}" --results_dir "$OUT"
}

run_oracle() {
    local dsname=$1 P=$2
    local OUT="./results/frag/${dsname}_${P}p/oracle"
    if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then
        echo "  SKIP ${dsname}-${P}p :: Oracle (done)"; return
    fi
    mkdir -p "$OUT"
    local ds_args
    read -ra ds_args <<< "$(args_for $dsname)"
    echo "===== ${dsname}-${P}p :: Oracle $(date) ====="
    python -u run_oracle_solo.py --num_parties "$P" "${SEEDS[@]}" "${ds_args[@]}" \
                                  --results_dir "$OUT"
}

# All 9 cells in (dsname, P) order. cell args: dsname P
CELLS=(
    "c10  1"
    "c10  4"
    "c100 1"
    "c100 4"
    "tin  1"
    "tin  2"
    "tin  4"
)

echo "===== FRAG SWEEP START $(date) ====="

# ─── PHASE 1: Oracle + 4 P0 methods, swept across all cells ───
# At each cell we run Oracle (KL reference) then the 4 P0 method combos.
# After this phase: 5-row Table 1 is fully populated on all 9 cells.
echo ""; echo "##### PHASE 1 (Oracle + P0) $(date) #####"
for cell in "${CELLS[@]}"; do
    set -- $cell; dsname=$1; P=$2
    run_oracle "$dsname" "$P"
    for m in "${P0_METHODS[@]}"; do
        run_combo "${m%:*}" "${m#*:}" "$dsname" "$P"
    done
done

# ─── PHASE 2: P1 methods across all cells → 8-row Table 1 ───
echo ""; echo "##### PHASE 2 (P1) $(date) #####"
for m in "${P1_METHODS[@]}"; do
    for cell in "${CELLS[@]}"; do
        set -- $cell; dsname=$1; P=$2
        run_combo "${m%:*}" "${m#*:}" "$dsname" "$P"
    done
done

# ─── PHASE 3: P2 methods across all cells → 13-row Table 1 ───
echo ""; echo "##### PHASE 3 (P2) $(date) #####"
for m in "${P2_METHODS[@]}"; do
    for cell in "${CELLS[@]}"; do
        set -- $cell; dsname=$1; P=$2
        run_combo "${m%:*}" "${m#*:}" "$dsname" "$P"
    done
done

echo ""; echo "===== FRAG SWEEP DONE $(date) ====="
