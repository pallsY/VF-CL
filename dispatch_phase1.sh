#!/bin/bash
# Phase-1 frag dispatch on the 8-DCU cluster: Oracle + 4 P0 methods x 7 cells
# = 35 jobs, round-robin across DCUs 0..7. Expensive tin/c100 oracles are
# queued first so they start immediately and run while cheap jobs churn.
# Idempotent (skips any cell whose aggregated.json already exists).
cd /public/home/dongshou/projects/VF-CUL
set +u   # DTK env.sh / conda init reference unset vars (LD_LIBRARY_PATH); not nounset-clean
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct
set -u

CELLS=("c10:1" "c10:4" "c100:1" "c100:4" "tin:1" "tin:2" "tin:4")
METHODS=("finetune:retrain" "der_pp:luv" "proto_evolve:luv" "proto_evolve:fedosd")

# Queue: oracles first (most expensive ds first), then the 28 method jobs.
QUEUE=()
for c in tin:4 tin:2 tin:1 c100:4 c100:1 c10:4 c10:1; do QUEUE+=("${c}:ORACLE"); done
for m in "${METHODS[@]}"; do for c in "${CELLS[@]}"; do QUEUE+=("${c}:${m}"); done; done

run_job() {
    local gpu=$1 item=$2
    local ds P a b
    IFS=":" read -r ds P a b <<< "$item"
    local -a DSA
    case "$ds" in
        c10)  DSA=(--data cifar10 --num_classes 10 --custom_tasks "0,1|2,3|4,5|6,7|8,9"
                   --unlearn_after_tasks "1,3" --unlearn_classes "0;5");;
        c100) DSA=(--data cifar100 --num_classes 100 --num_tasks 5 --classes_per_task 20
                   --unlearn_after_tasks "1,3" --unlearn_classes "20;60");;
        tin)  DSA=(--data tinyimagenet --num_classes 200 --num_tasks 10 --classes_per_task 20
                   --unlearn_after_tasks "3,7" --unlearn_classes "20;100");;
    esac
    local COMMON=(--num_parties "$P" --seeds 42,43,44 --device cuda:0
                  --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01
                  --aggregation sum --model_type resnet18)
    if [ "$a" = "ORACLE" ]; then
        local OUT="./results/frag/${ds}_${P}p/oracle"
        if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then echo "[G$gpu] SKIP oracle $ds-${P}p"; return; fi
        mkdir -p "$OUT"
        echo "[G$gpu] $(date) START oracle $ds-${P}p"
        HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
            python -u run_oracle_solo.py "${COMMON[@]}" "${DSA[@]}" --results_dir "$OUT"
        echo "[G$gpu] $(date) END oracle $ds-${P}p"
    else
        local cl=$a ul=$b
        local OUT="./results/frag/${ds}_${P}p/${cl}_x_${ul}"
        if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then echo "[G$gpu] SKIP $cl x $ul $ds-${P}p"; return; fi
        mkdir -p "$OUT"
        echo "[G$gpu] $(date) START $cl x $ul $ds-${P}p"
        HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
            python -u main.py --cl_method "$cl" --ul_method "$ul" "${COMMON[@]}" "${DSA[@]}" --results_dir "$OUT"
        echo "[G$gpu] $(date) END $cl x $ul $ds-${P}p"
    fi
}

mkdir -p logs/phase1
echo "PHASE1 dispatch: ${#QUEUE[@]} jobs over 8 DCUs $(date)"
for gpu in 0 1 2 3 4 5 6 7; do
    (
        for i in "${!QUEUE[@]}"; do
            (( i % 8 == gpu )) || continue
            run_job "$gpu" "${QUEUE[$i]}"
        done
        echo "[G$gpu] QUEUE EMPTY $(date)"
    ) > logs/phase1/gpu${gpu}.log 2>&1 &
done
wait
echo "PHASE1 ALL DONE $(date)"
