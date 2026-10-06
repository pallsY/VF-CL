#!/bin/bash
# 4-condition MVP for the redundancy-adaptive joint method.
#
# Cell: cifar100, P=4, sum aggregation, proto_evolve backbone family.
# Timeline: 5 CIL tasks (20 cls each) + 2 UL events forced to span the
# easy/hard divide:
#   forget class 55 (otter, redundancy ~0.11 — HARD) after task 2
#   forget class 60 (plain, redundancy ~0.66 — EASY) after task 3
#
# Conditions:
#   A vanilla-light : proto_evolve         x luv             (always LIGHT)
#   B vanilla-heavy : proto_evolve         x fedosd          (always HEAVY)
#   C ul-only       : proto_evolve         x radapt_router   (route UL only)
#   D full joint    : proto_evolve_radapt  x radapt_router   (route both)
#
# Single seed 42 first (go/no-go). If D beats both A,B on the
# comm/precision Pareto, scale to 3 seeds.

set -u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate FedEMoE
cd "$(dirname "$(readlink -f "$0")")"

COMMON=(--data cifar100 --num_classes 100
        --num_tasks 5 --classes_per_task 20
        --num_parties 4 --aggregation sum --model_type resnet18
        --epochs_per_task 20 --batch_size 128 --lr 0.01 --ul_lr 1e-4
        --replay_mode prototype
        --seed 42
        --unlearn_after_tasks 2,3
        --unlearn_classes "55;60"
        --results_dir ./results)

run_one() {
    local CL=$1 UL=$2 GPU=$3 NAME=$4
    local LOG="./radapt_mvp_${NAME}.log"
    echo "[$(date)] launch ${NAME}: ${CL} x ${UL} on cuda:${GPU}"
    python -u main.py --cl_method "$CL" --ul_method "$UL" \
        "${COMMON[@]}" --device "cuda:$GPU" \
        --exp_name "radapt_mvp_${NAME}" \
        > "$LOG" 2>&1
    echo "[$(date)] done ${NAME} (exit=$?)"
}

# GPU0 lane: A vanilla-light  (~35min) -> D full joint   (~50min)  ~85min
( run_one proto_evolve         luv             0 A_vanilla_light
  run_one proto_evolve_radapt  radapt_router   0 D_full_joint    ) > /tmp/radapt_lane0.log 2>&1 &
LANE0=$!

# GPU1 lane: B vanilla-heavy  (~35min) -> C ul-only      (~35min)  ~70min
( run_one proto_evolve         fedosd          1 B_vanilla_heavy
  run_one proto_evolve         radapt_router   1 C_ul_only        ) > /tmp/radapt_lane1.log 2>&1 &
LANE1=$!

echo "lane0 pid=$LANE0  lane1 pid=$LANE1"
wait
echo "=== RADAPT MVP DONE $(date) ==="
