#!/usr/bin/env bash
# VFL-CLU benchmark MAIN GRID — CIFAR-100 column-split (long-queue dataset).
# 12 CL x 9 UL x 3 seeds; ORDERS=1 by default (Order 2 after curation).
# 10 tasks x 10 classes; UL after T2/T5/T8, 2 classes per event (6 forgotten).
# Compute-heavy: resnet18 x 10 tasks -> dispatch Order 1 first, resume-safe.
# NB no set -u.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

ORDERS="${ORDERS:-1}"
SEEDS="--seeds 42,43,44"
EP="${EP:-30}"
# space-separated CL row names to skip (deferred rows; backfill by re-running
# without CL_SKIP — resume picks up only the missing combos)
CL_SKIP="${CL_SKIP:-}"

TASKS=$(python - <<'EOF'
print('|'.join(','.join(str(c) for c in range(t*10,(t+1)*10)) for t in range(10)))
EOF
)

CL_LIST=(
  "finetune|finetune|"
  "ewc|ewc|"
  "lwf|lwf|"
  "lwf_wa|lwf_wa|"
  "lwf_fim|lwf_fim|"
  "afc|afc|"
  "gpm|gpm|"
  "adagauss|adagauss|"
  "target|target|"
  "proto_evolve|proto_evolve|"
  "proto_fedspace|proto_fedspace|"
  "ours_protoOC|proto_evolve|--own_concentrate_weight 0.5"
)
UL_LIST=(
  "retrain|retrain|"
  "ga|gradient_ascent|"
  "luv|luv|"
  "mode|mode|"
  "fucrt|fucrt|"
  "fedup|fedup|"
  "fedosd|fedosd|"
  "fedau|fedau|"
  "ours_localrt|roar|--roar_scrub_mode retrain_s --roar_tau_own 0.6 --roar_retrain_epochs 30"
)

order_args() {  # $1 = order
  if [ "$1" = 1 ]; then
    echo "--unlearn_after_tasks 2,5,8 --unlearn_classes 0,1;20,25;50,55"
  else
    echo "--unlearn_after_tasks 2,5,8 --unlearn_classes 28,29;58,59;88,89"
  fi
}

BASE="--data cifar100 --num_classes 100 --model_type resnet18 --num_parties 4 \
  --aggregation concat --cosine_head --epochs_per_task $EP --custom_tasks $TASKS"

JOBS=$(mktemp)
for od in $ORDERS; do
  for cl in "${CL_LIST[@]}"; do
    IFS='|' read -r cname cmeth cextra <<< "$cl"
    case " $CL_SKIP " in *" $cname "*) echo "skip row: $cname"; continue;; esac
    for ul in "${UL_LIST[@]}"; do
      IFS='|' read -r uname umeth uextra <<< "$ul"
      out="./results/grid_cifar/o${od}/${cname}__${uname}"
      ndone=$(find "$out" -name results.json 2>/dev/null | wc -l)
      [ "$ndone" -ge 3 ] && continue
      printf '%s\t%s\n' "$out" \
        "$BASE $(order_args $od) --cl_method $cmeth $cextra --ul_method $umeth $uextra $SEEDS --device cuda:0" \
        >> "$JOBS"
    done
  done
done
N=$(wc -l < "$JOBS")
echo "cifar grid jobs: $N"

rm -f ALL_GRIDCIFAR_DONE.flag
mkdir -p logs_grid
worker() {
  local i=0
  while IFS=$'\t' read -r out args; do
    if [ $(( i % 8 )) -eq "$1" ]; then
      mkdir -p "$out"
      echo "[GPU $1] $out"
      HIP_VISIBLE_DEVICES=$1 python -u main.py $args --results_dir "$out" \
        > "logs_grid/$(echo "$out" | tr '/' '_').log" 2>&1 < /dev/null
    fi
    i=$(( i + 1 ))
  done < "$JOBS"
}
for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
touch ALL_GRIDCIFAR_DONE.flag
echo "ALL GRID CIFAR DONE $(date)"
