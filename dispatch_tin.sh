#!/bin/bash
# tin "cheap" backfill on 2 idle DCUs (6,7): ROAR on tin (1p/2p/4p x 2 backbones)
# + the missing tin_2p/tin_4p Phase-1 baseline METHODS (no oracle here — oracle
# is slow, see dispatch_tin_oracle.sh). Auto-waits for the c100 ROAR run to free
# the cards, then round-robins. Idempotent on aggregated.json.
# Runs everything from VF-CUL-roar (has roar + all baseline methods); writes into
# the MAIN repo results/frag so the table stays unified.
cd /public/home/dongshou/projects/VF-CUL-roar
# wait for c100 roar to release cards 6,7
while pgrep -f "main.py --cl_method.*--ul_method roar.*cifar100" >/dev/null 2>&1; do sleep 120; done
set +u; source /opt/dtk-25.04.2/env.sh; source /public/home/dongshou/anaconda/etc/profile.d/conda.sh; conda activate ct; set -u
MAIN=/public/home/dongshou/projects/VF-CUL
TIN='--data tinyimagenet --num_classes 200 --num_tasks 10 --classes_per_task 20 --unlearn_after_tasks 3,7 --unlearn_classes 20;100'
COMMON="--seeds 42,43,44 --device cuda:0 --epochs_per_task 30 --ul_epochs 5 --batch_size 128 --lr 0.01 --aggregation sum --model_type resnet18"

# queue items: "cl:ul:P"  (ul=roar -> add roar args)
QUEUE=(
  er_ace:roar:1 proto_evolve:roar:1
  er_ace:roar:2 proto_evolve:roar:2
  er_ace:roar:4 proto_evolve:roar:4
  finetune:retrain:2 der_pp:luv:2 proto_evolve:luv:2 proto_evolve:fedosd:2
  finetune:retrain:4 der_pp:luv:4 proto_evolve:luv:4 proto_evolve:fedosd:4
)

run_job() {
  local card=$1 item=$2 cl ul P
  IFS=":" read -r cl ul P <<< "$item"
  local OUT="$MAIN/results/frag/tin_${P}p/${cl}_x_${ul}"
  if ls "$OUT"/*/aggregated.json >/dev/null 2>&1; then echo "[c$card] SKIP ${cl}x${ul} tin-${P}p"; return; fi
  mkdir -p "$OUT"
  local -a EXTRA=()
  [ "$ul" = roar ] && EXTRA=(--roar_scrub_epochs 0 --roar_recovery_epochs 5)
  echo "[c$card] START ${cl}x${ul} tin-${P}p $(date)"
  HIP_VISIBLE_DEVICES=$card CUDA_VISIBLE_DEVICES=$card python -u main.py \
    --cl_method "$cl" --ul_method "$ul" --num_parties "$P" $COMMON $TIN "${EXTRA[@]}" --results_dir "$OUT"
  echo "[c$card] END ${cl}x${ul} tin-${P}p $(date)"
}

mkdir -p logs_tin
for card in 6 7; do
  ( idx=$([ $card = 6 ] && echo 0 || echo 1)
    for i in "${!QUEUE[@]}"; do (( i % 2 == idx )) && run_job "$card" "${QUEUE[$i]}"; done
    echo "[c$card] QUEUE DONE $(date)"
  ) > logs_tin/card${card}.log 2>&1 &
done
wait
echo "DISPATCH TIN DONE $(date)"
