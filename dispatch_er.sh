#!/usr/bin/env bash
# Sequential ER before/after sweep on one DCU (HIP_VISIBLE_DEVICES=1).
# NB: no `set -u` -- the cluster env.sh references unbound LD_LIBRARY_PATH.
cd /public/home/dongshou/cl_fix/er
source /opt/dtk-25.04.2/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct

BASE=(--cl_method er --ul_method luv --num_parties 1 --seeds 42 --device cuda:0
      --epochs_per_task 30 --batch_size 128 --lr 0.01 --aggregation sum
      --model_type resnet18 --data cifar10 --num_classes 10
      --custom_tasks "0,1|2,3|4,5|6,7|8,9" --unlearn_after_tasks "99"
      --unlearn_classes "0" --cosine_head)

run () {  # name  extra-args...
  local name="$1"; shift
  local rd="./results/${name}"
  echo "=== [$(date +%H:%M:%S)] START ${name} :: $* ==="
  HIP_VISIBLE_DEVICES=1 python -u main.py "${BASE[@]}" "$@" --results_dir "${rd}" \
      > "log_${name}.txt" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE ${name} (exit $?) ==="
  touch "done_${name}.flag"
}

# Primary result first, then control, then sensitivity.
run after_c300        --er_per_class 300
run before_c20        --er_per_class 20
run after_c500        --er_per_class 500
run after_c300_a2     --er_per_class 300 --er_alpha 2.0

touch ALL_DONE.flag
echo "=== [$(date +%H:%M:%S)] ALL RUNS COMPLETE ==="
