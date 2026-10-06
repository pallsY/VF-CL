#!/usr/bin/env bash
# VFL-CLU benchmark MAIN GRID — tabular datasets (Covertype + NUS-WIDE).
# 12 CL x 9 UL x 2 orders x {covtype,nuswide} x 3 seeds (seeds loop inside main.py).
# 8-GPU round-robin, one worker queue per GPU. CIFAR-100 has its own dispatcher.
# Protocol: docs_benchmark_protocol.md (sanitize_cl_state defaults ON).
# NB no set -u.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

DATASETS="${DATASETS:-covtype nuswide}"
ORDERS="${ORDERS:-1 2}"
SEEDS="--seeds 42,43,44"

# CL rows: "name|cl_method|extra_args"
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
# UL cols: "name|ul_method|extra_args"
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

dataset_args() {  # $1 = dataset, $2 = order
  case "$1" in
    covtype)
      base="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz \
        --model_type mlp --num_parties 6 --aggregation concat --cosine_head \
        --num_classes 7 --custom_tasks 0,1,2|3,4|5,6 --epochs_per_task 30"
      [ "$2" = 1 ] && q="--unlearn_after_tasks 1,2 --unlearn_classes 1;6" \
                   || q="--unlearn_after_tasks 0,2 --unlearn_classes 0;4" ;;
    nuswide)
      base="--data tabvfl --vector_npz data/nuswide_vfl/nuswide_vfl.npz \
        --model_type mlp --num_parties 6 --aggregation concat --cosine_head \
        --num_classes 10 --custom_tasks 0,1,2,3|4,5,6|7,8,9 --epochs_per_task 30"
      [ "$2" = 1 ] && q="--unlearn_after_tasks 1,2 --unlearn_classes 0;5" \
                   || q="--unlearn_after_tasks 0,2 --unlearn_classes 2;8" ;;
  esac
  echo "$base $q"
}

# Build the job list (one line per job: "outdir<TAB>full args")
JOBS=$(mktemp)
for ds in $DATASETS; do
  for od in $ORDERS; do
    for cl in "${CL_LIST[@]}"; do
      IFS='|' read -r cname cmeth cextra <<< "$cl"
      for ul in "${UL_LIST[@]}"; do
        IFS='|' read -r uname umeth uextra <<< "$ul"
        out="./results/grid_tab/${ds}_o${od}/${cname}__${uname}"
        # resume: main.py multiseed nests $out/<combo>_multiseed_<ts>/<combo>/seed_<s>/
        # -> done when all 3 seeds' results.json exist
        ndone=$(find "$out" -name results.json 2>/dev/null | wc -l)
        [ "$ndone" -ge 3 ] && continue
        printf '%s\t%s\n' "$out" \
          "$(dataset_args $ds $od) --cl_method $cmeth $cextra --ul_method $umeth $uextra $SEEDS --device cuda:0" \
          >> "$JOBS"
      done
    done
  done
done
N=$(wc -l < "$JOBS")
echo "grid jobs to run: $N"

rm -f ALL_GRIDTAB_DONE.flag
mkdir -p logs_grid
worker() {  # $1 = gpu id
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
touch ALL_GRIDTAB_DONE.flag
echo "ALL GRID TAB DONE $(date)"
