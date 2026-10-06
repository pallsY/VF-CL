#!/usr/bin/env bash
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh; source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct
COMMON="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz --model_type mlp \
  --num_parties 6 --aggregation concat --cosine_head --num_classes 7 \
  --custom_tasks 0,1|2,3|4,5|6 --cl_method finetune --ul_method roar \
  --unlearn_after_tasks 3 --unlearn_classes 6 --roar_tau_own 0.6 \
  --roar_scrub_epochs 10 --roar_scrub_lr 1e-3 --epochs_per_task 30 --seeds 42,43,44 --device cuda:0"
rm -f ALL_RAWSWEEP_DONE.flag; i=0
for rw in 0 1 5 20 50; do
  gpu=$(( i % 8 )); tag=$(echo $rw|tr '.' 'p')
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $COMMON --roar_scrub_raw_weight $rw \
    --results_dir ./results/covtype_raw/rw_${tag} > log_rawsweep_${tag}.txt 2>&1 </dev/null &
  i=$(( i+1 )); done
wait; touch ALL_RAWSWEEP_DONE.flag; echo DONE
