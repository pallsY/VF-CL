#!/usr/bin/env bash
# GATING smoke: does ownership concentrate enough that |S*| < P ?  Under sum-agg
# the top weight is shared across parties -> S*=P (vacuous comm claim). Test
# CONCAT-agg at P=4/8 (per-party weight blocks differ -> ownership can
# concentrate) vs a sum-agg P=4 contrast. roar prints "union S*=[...] / N parties"
# per forget class -> grep the logs to see |S*| vs P. 15ep/1seed, o1 unlearn.
# NB: no `set -u`; plain env.sh source.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

BASE="--ul_method roar --cosine_head --seeds 42 --device cuda:0 \
  --epochs_per_task 15 --ul_epochs 5 --batch_size 128 --lr 0.01 --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5"

# tag|extra
JOBS=(
  "roar_concat_p4|--cl_method proto_evolve --feat_distill_weight 1 --aggregation concat --num_parties 4"
  "roar_concat_p8|--cl_method proto_evolve --feat_distill_weight 1 --aggregation concat --num_parties 8"
  "roar_concat_p4_ft|--cl_method finetune --aggregation concat --num_parties 4"
  "roar_sum_p4|--cl_method proto_evolve --feat_distill_weight 1 --aggregation sum --num_parties 4"
)

rm -f ALL_CONCAT_DONE.flag
i=0
for job in "${JOBS[@]}"; do
  tag="${job%%|*}"; args="${job#*|}"
  gpu=$(( i % 8 ))
  out="./results/concat_smoke/${tag}"
  echo "[GPU $gpu] $tag :: $args"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $args $BASE \
    --results_dir "$out" > "log_concat_${tag}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_CONCAT_DONE.flag
echo "ALL CONCAT SMOKE DONE $(date)"
