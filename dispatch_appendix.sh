#!/usr/bin/env bash
# Appendix suites for the VFL-CLU benchmark:
#   A) CL-only  — each CL method on the task queue with NO unlearning events
#                 (AA/BWT table; ul_method never invoked, set to retrain)
#   B) UL-only  — each UL method on a JOINTLY-trained backbone (single task
#                 holding all classes) with one unlearning event forgetting the
#                 protocol classes (classical one-shot unlearning comparison)
# MODE=tab (covtype+nuswide, run anytime) | cifar (queue after main grid).
# Resume-safe. NB no set -u.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

MODE="${MODE:-tab}"
SEEDS="--seeds 42,43,44"

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

JOBS=$(mktemp)
add_job() {  # $1 out, $2 args
  ndone=$(find "$1" -name results.json 2>/dev/null | wc -l)
  [ "$ndone" -ge 3 ] && return
  printf '%s\t%s\n' "$1" "$2" >> "$JOBS"
}

if [ "$MODE" = tab ]; then
  for ds in covtype nuswide; do
    if [ "$ds" = covtype ]; then
      BASE="--data tabvfl --vector_npz data/covtype_vfl/covertype_vfl.npz --model_type mlp \
        --num_parties 6 --aggregation concat --cosine_head --num_classes 7"
      CLQ="--custom_tasks 0,1,2|3,4|5,6 --epochs_per_task 30 --unlearn_after_tasks 99 --unlearn_classes 99"
      ULQ="--custom_tasks 0,1,2,3,4,5,6 --epochs_per_task 60 --unlearn_after_tasks 0 --unlearn_classes 1,6"
    else
      BASE="--data tabvfl --vector_npz data/nuswide_vfl/nuswide_vfl.npz --model_type mlp \
        --num_parties 6 --aggregation concat --cosine_head --num_classes 10"
      CLQ="--custom_tasks 0,1,2,3|4,5,6|7,8,9 --epochs_per_task 30 --unlearn_after_tasks 99 --unlearn_classes 99"
      ULQ="--custom_tasks 0,1,2,3,4,5,6,7,8,9 --epochs_per_task 60 --unlearn_after_tasks 0 --unlearn_classes 0,5"
    fi
    for cl in "${CL_LIST[@]}"; do
      IFS='|' read -r cname cmeth cextra <<< "$cl"
      add_job "./results/appendix/clonly_${ds}/${cname}" \
        "$BASE $CLQ --cl_method $cmeth $cextra --ul_method retrain $SEEDS --device cuda:0"
    done
    for ul in "${UL_LIST[@]}"; do
      IFS='|' read -r uname umeth uextra <<< "$ul"
      add_job "./results/appendix/ulonly_${ds}/${uname}" \
        "$BASE $ULQ --cl_method finetune --ul_method $umeth $uextra $SEEDS --device cuda:0"
    done
  done
  FLAG=ALL_APPENDIX_TAB_DONE.flag
else
  TASKS=$(python - <<'EOF'
print('|'.join(','.join(str(c) for c in range(t*10,(t+1)*10)) for t in range(10)))
EOF
)
  ALLC=$(python -c "print(','.join(str(c) for c in range(100)))")
  BASE="--data cifar100 --num_classes 100 --model_type resnet18 --num_parties 4 \
    --aggregation concat --cosine_head"
  CLQ="--custom_tasks $TASKS --epochs_per_task 30 --unlearn_after_tasks 99 --unlearn_classes 99"
  ULQ="--custom_tasks $ALLC --epochs_per_task 60 --unlearn_after_tasks 0 --unlearn_classes 0,1"
  for cl in "${CL_LIST[@]}"; do
    IFS='|' read -r cname cmeth cextra <<< "$cl"
    add_job "./results/appendix/clonly_cifar/${cname}" \
      "$BASE $CLQ --cl_method $cmeth $cextra --ul_method retrain $SEEDS --device cuda:0"
  done
  for ul in "${UL_LIST[@]}"; do
    IFS='|' read -r uname umeth uextra <<< "$ul"
    add_job "./results/appendix/ulonly_cifar/${uname}" \
      "$BASE $ULQ --cl_method finetune --ul_method $umeth $uextra $SEEDS --device cuda:0"
  done
  FLAG=ALL_APPENDIX_CIFAR_DONE.flag
fi

N=$(wc -l < "$JOBS")
echo "appendix($MODE) jobs: $N"
rm -f "$FLAG"
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
touch "$FLAG"
echo "APPENDIX($MODE) DONE $(date)"
