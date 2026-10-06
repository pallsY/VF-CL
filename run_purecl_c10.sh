#!/bin/bash
# Pure-CL diagnostic on CIFAR-10: NO unlearning events (unlearn_after_tasks=99 never
# matches any real task id -> timeline has only CIL events). Isolates continual
# learning from UL so we can (a) test whether new tasks still collapse at eval
# (head-bias hypothesis) and (b) rank CL methods without UL confounds.
set -u
source /home/chase/miniconda3/etc/profile.d/conda.sh
conda activate FedEMoE
cd /home/chase/ct/VF-CUL

COMMON=(--num_parties 1 --seeds 42,43,44 --device cuda:0
        --epochs_per_task 30 --batch_size 128 --lr 0.01
        --aggregation sum --model_type resnet18
        --data cifar10 --num_classes 10 --custom_tasks "0,1|2,3|4,5|6,7|8,9"
        --unlearn_after_tasks "99" --unlearn_classes "0")   # 99 => no UL events

for M in finetune er der_pp; do
    OUT="./results/_purecl2_c10/$M"
    if ls "$OUT"/**/aggregated.json >/dev/null 2>&1 || find "$OUT" -name aggregated.json 2>/dev/null | grep -q .; then
        echo "SKIP $M (done)"; continue
    fi
    mkdir -p "$OUT"
    echo "===== PURE-CL $M  $(date) ====="
    python -u main.py --cl_method "$M" --ul_method luv "${COMMON[@]}" --results_dir "$OUT"
done
echo "===== PURE-CL DONE $(date) ====="
