#!/usr/bin/env bash
# Figure: ownership-concentration AXIS. Sweep the synthetic specialization knob
# rho (via --vector_npz); fixed tau_own=0.8, proto_evolve x roar, concat, cosine,
# P=8, 30ep, 3 seeds. As rho grows: measured ownership concentrates (entropy
# ratio falls), |S*| shrinks (comm drops), the Thm-1 residual bound stays valid.
# rho=0 is the negative control (= CIFAR/mfeat uniform-ownership null regime).
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

COMMON="--data synthvfl --model_type mlp --num_parties 8 --aggregation concat --cosine_head \
  --num_classes 10 --custom_tasks 0,1|2,3|4,5|6,7|8,9 \
  --cl_method finetune --ul_method roar
  --unlearn_after_tasks 1,3 --unlearn_classes 0;5 \
  --epochs_per_task 30 --seeds 42,43,44 --roar_tau_own 0.8 --device cuda:0"

rm -f ALL_SYNTHRHOFT_DONE.flag
i=0
for rho in 0p00 0p25 0p50 0p75 1p00; do
  gpu=$(( (i+3) % 8 ))
  out="./results/synth_rho_ft/rho_${rho}"
  npz="data/synthvfl/synth_rho${rho}.npz"
  echo "[GPU $gpu] roar synthvfl rho=$rho"
  HIP_VISIBLE_DEVICES=$gpu setsid nohup python -u main.py $COMMON \
    --vector_npz "$npz" --results_dir "$out" \
    > "log_synthrhoft_${rho}.txt" 2>&1 < /dev/null &
  i=$(( i + 1 ))
done
wait
touch ALL_SYNTHRHOFT_DONE.flag
echo "ALL SYNTH RHO FT DONE $(date)"
