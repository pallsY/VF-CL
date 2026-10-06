#!/usr/bin/env bash
# GATING follow-up: homogeneous concat gave near-uniform ownership (S*=P). Does
# HETEROGENEOUS party width concentrate ownership so |S*|<P? party_widths sum=32
# (c10 image columns). Also run verify_ownership on a LINEAR head as the Prop-1
# exactness anchor (max|recon-logit|~1e-6). 15ep/1seed. NB: no `set -u`.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

ROARBASE="--ul_method roar --cl_method proto_evolve --feat_distill_weight 1 --cosine_head \
  --aggregation concat --num_parties 4 --seeds 42 --device cuda:0 \
  --epochs_per_task 15 --ul_epochs 5 --batch_size 128 --lr 0.01 --model_type resnet18 \
  --data cifar10 --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5"

rm -f ALL_HET_DONE.flag

# het roar: dominant-party splits
HIP_VISIBLE_DEVICES=4 setsid nohup python -u main.py $ROARBASE --party_widths 16,8,5,3 \
  --results_dir ./results/het_smoke/roar_het_a > log_het_roar_a.txt 2>&1 < /dev/null &
HIP_VISIBLE_DEVICES=5 setsid nohup python -u main.py $ROARBASE --party_widths 20,7,3,2 \
  --results_dir ./results/het_smoke/roar_het_b > log_het_roar_b.txt 2>&1 < /dev/null &

# Prop-1 linear-head exactness anchor (no --cosine_head -> recon should match logit)
HIP_VISIBLE_DEVICES=6 setsid nohup python -u verify_ownership.py \
  --data cifar10 --num_classes 10 --aggregation concat --num_parties 4 \
  --custom_tasks 0,1|2,3|4,5|6,7|8,9 --epochs_per_task 15 --seed 42 \
  --output_dir ./results/ownership_anchor_linear > log_ownership_anchor.txt 2>&1 < /dev/null &

wait
touch ALL_HET_DONE.flag
echo "ALL HET SMOKE DONE $(date)"
