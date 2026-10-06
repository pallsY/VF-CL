#!/usr/bin/env bash
# CORRECTED UL benchmark re-run after the cumulative-retain fix (runner.py).
# The old results/ul_full event-2 numbers were invalid (fine-tuning methods
# re-learned the earlier-forgotten class). proto_evolve(main, cosine+featKD) x
# ALL 11 UL, c10, P=2, o1, 3 seeds, 30ep -> results/ul_full2 (new dir, keeps the
# buggy ul_full for before/after comparison). QUEUED after the concat smoke so
# it gets clean GPUs. NB: no `set -u`.
cd /public/home/dongshou/cl_fix2/run
echo "waiting for concat smoke to finish..."
while [ ! -f ALL_CONCAT_DONE.flag ]; do sleep 30; done
echo "concat done -> starting corrected UL re-run $(date)"
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--cl_method proto_evolve --cosine_head --feat_distill_weight 1 \
  --num_parties 2 --seeds 42,43,44 --device cuda:0 --epochs_per_task 30 --ul_epochs 5 \
  --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5"

ULS=(retrain gradient_ascent luv mode fucrt fedup fedosd fedau fudp radapt_router roar)

rm -f ALL_ULRERUN_DONE.flag
i=0
for ul in "${ULS[@]}"; do
  gpu=$(( i % 8 ))
  out="./results/ul_full2/${ul}"
  echo "[GPU $gpu] proto_evolve x $ul (3 seeds, FIXED)"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py --ul_method "$ul" $COMMON \
    --results_dir "$out" > "log_ulrerun_${ul}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_ULRERUN_DONE.flag
echo "ALL UL RERUN DONE $(date)"
