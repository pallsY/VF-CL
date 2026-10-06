#!/usr/bin/env bash
# Headline experiment: REAL cross-org anchor (Covertype, gate-passed entropy 0.78).
# Sweep roar_tau_own; finetune x roar, concat, cosine, P=6, 7 classes, 30ep, 3 seeds.
# Forget class 4 (S*~3) then class 6 (S*~2) -> different minimal owning sets ->
# the certified comm-vs-leakage Pareto on real data. NB no set -u.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz \
  --model_type mlp --num_parties 6 --aggregation concat --cosine_head \
  --num_classes 7 --custom_tasks 0,1|2,3|4,5|6 \
  --cl_method finetune --ul_method roar \
  --unlearn_after_tasks 2,3 --unlearn_classes 4;6 \
  --epochs_per_task 30 --seeds 42,43,44 --device cuda:0"

rm -f ALL_COVTAU_DONE.flag
i=0
for tau in 0.5 0.6 0.7 0.8 0.9; do
  gpu=$(( i % 8 ))
  out="./results/covtype_tau/tau_${tau}"
  echo "[GPU $gpu] covtype roar tau_own=$tau"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $COMMON --roar_tau_own $tau \
    --results_dir "$out" > "log_covtau_${tau}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_COVTAU_DONE.flag
echo "ALL COVTYPE TAU DONE $(date)"
