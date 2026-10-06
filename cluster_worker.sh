#!/bin/bash
# Cluster GPU worker. Runs the given list of (cl:ul:dsname:P) combos serially
# on the specified GPU. Idempotent on aggregated.json.
# Usage: cluster_worker.sh <gpu_id> <combo1> <combo2> ...
set -u
GPU=$1; shift

source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct
cd /public/home/dongshou/projects/VF-CUL

echo "===== WORKER GPU=$GPU start $(date) ====="
echo "queue: $@"
echo ""

for combo in "$@"; do
    IFS=":" read -r cl ul dsname P <<< "$combo"
    OUT="./results/frag/${dsname}_${P}p/${cl}_x_${ul}"
    if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then
        echo "[GPU$GPU] SKIP ${dsname}-${P}p :: ${cl} x ${ul} (already done)"
        continue
    fi
    mkdir -p "$OUT"

    case "$dsname" in
        c10)  DS_ARGS=(--data cifar10  --num_classes 10
                       --custom_tasks "0,1|2,3|4,5|6,7|8,9"
                       --unlearn_after_tasks "1,3" --unlearn_classes "0;5");;
        c100) DS_ARGS=(--data cifar100 --num_classes 100
                       --num_tasks 5 --classes_per_task 20
                       --unlearn_after_tasks "1,3" --unlearn_classes "20;60");;
        tin)  DS_ARGS=(--data tinyimagenet --num_classes 200
                       --num_tasks 10 --classes_per_task 20
                       --unlearn_after_tasks "3,7" --unlearn_classes "20;100");;
    esac

    echo "===== GPU$GPU ${dsname}-${P}p :: ${cl} x ${ul} $(date) ====="
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    python -u main.py --cl_method "$cl" --ul_method "$ul" --num_parties "$P" \
        --seeds 42,43,44 --device cuda:0 \
        --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01 \
        --aggregation sum --model_type resnet18 \
        "${DS_ARGS[@]}" --results_dir "$OUT"
done

echo ""
echo "===== WORKER GPU=$GPU done $(date) ====="
