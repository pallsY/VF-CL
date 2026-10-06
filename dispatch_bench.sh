#!/usr/bin/env bash
# Systematic benchmark: 3 real VFL datasets x 3 seeds.
#  Sweep A (CL axis): fix UL = ours (localized retrain of S*), vary CL method.
#  Sweep B (UL axis): fix CL = finetune, vary UL method.
#  + oracle floor per dataset. Launch in waves of 8 (one per GPU), wait between.
cd /public/home/dongshou/cl_fix2/run
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct
rm -rf results/bench; mkdir -p results/bench
SEEDS=42,43,44; EP=30
UL_OURS="--ul_method roar --roar_scrub_mode retrain_s --roar_tau_own 0.6 --roar_retrain_epochs 20"

# dataset fields separated by ';' (tasks keep their internal '|')
# name ; npz ; num_classes ; tasks ; forget_after_task ; forget_class
DSETS=(
  "mfeat;data/mfeat/mfeat_6view.npz;10;0,1|2,3|4,5|6,7|8,9;4;5"
  "cov;data/covtype_vfl/covertype_vfl.npz;7;0,1|2,3|4,5|6;3;4"
  "har;data/har_sem_vfl/har_sem_vfl.npz;6;0,1|2,3|4,5;2;2"
)
CL_METHODS="finetune ewc lwf lwf_wa er der_pp afc adagauss proto_evolve"
UL_METHODS="retrain gradient_ascent luv fucrt fedup fedosd fedau radapt_router roar"

TAGS=(); CMDS=()
add(){ TAGS+=("$1"); CMDS+=("$2"); }

for d in "${DSETS[@]}"; do
  IFS=';' read -r name npz nc tasks fat fc <<< "$d"
  base="--data tabvfl --vector_npz $npz --model_type mlp --num_parties 6 --aggregation concat --cosine_head --num_classes $nc --custom_tasks $tasks --unlearn_after_tasks $fat --unlearn_classes $fc --epochs_per_task $EP --seeds $SEEDS --device cuda:0"
  for cl in $CL_METHODS; do
    extra=""; [ "$cl" = "proto_evolve" ] && extra="--feat_distill_weight 1"
    add "A_${name}_cl-${cl}" "python -u main.py $base --cl_method $cl $extra $UL_OURS --results_dir results/bench/A_${name}_cl-${cl}"
  done
  add "A_${name}_cl-ours" "python -u main.py $base --cl_method finetune --own_concentrate_weight 1.0 $UL_OURS --results_dir results/bench/A_${name}_cl-ours"
  for ul in $UL_METHODS; do
    add "B_${name}_ul-${ul}" "python -u main.py $base --cl_method finetune --ul_method $ul --roar_scrub_mode retrain_s --roar_tau_own 0.6 --roar_retrain_epochs 20 --results_dir results/bench/B_${name}_ul-${ul}"
  done
  others=$(python3 -c "print(','.join(str(i) for i in range($nc) if i!=$fc))")
  add "oracle_${name}" "python -u attack_baseline.py --data tabvfl --vector_npz $npz --model_type mlp --num_parties 6 --aggregation concat --cosine_head --num_classes $nc --custom_tasks $others --cl_method finetune --ul_method roar --forget_class $fc --epochs_per_task $EP --seeds 42 --device cuda:0 --results_dir results/bench/oracle_${name}"
done

echo "total jobs: ${#CMDS[@]}"
rm -f ALL_BENCH_DONE.flag
for idx in "${!CMDS[@]}"; do
  gpu=$(( idx % 8 ))
  HIP_VISIBLE_DEVICES=$gpu setsid nohup ${CMDS[$idx]} > "results/bench/log_${TAGS[$idx]}.txt" 2>&1 < /dev/null &
  if (( (idx+1) % 8 == 0 )); then wait; fi
done
wait
touch ALL_BENCH_DONE.flag
echo "ALL BENCH DONE $(date)"
