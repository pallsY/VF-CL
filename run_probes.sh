#!/bin/bash
# Verification probes for the CBM-direction GO/NO-GO decision.
# Runs probe_attribution.py at two fragmentation levels (P=2 and P=4) on
# CIFAR-10. Both runs are self-contained (train V-LETO from scratch + probe).
#
# Expected runtime: ~30-50 min per run on a single 4090 with the defaults
# below (5 tasks x 20 epochs ResNet18). Adjust --epochs_per_task to trade
# runtime for backbone convergence.
#
# Outputs land in ./results/probe_attribution_*/probe_report.json
set -u
# Portable across machines with different $HOME / repo paths.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate FedEMoE
cd "$(dirname "$(readlink -f "$0")")"

COMMON=(--data cifar10 --num_classes 10
        --custom_tasks "0,1|2,3|4,5|6,7|8,9"
        --aggregation sum --model_type resnet18
        --epochs_per_task 20 --batch_size 128 --lr 0.01
        --replay_mode prototype --cl_method proto_evolve
        --seed 42 --device cuda:0
        --unlearn_after_tasks 99,99 --unlearn_classes "0;0"
        --results_dir ./results)

for P in 2 4; do
    echo "===== probe P=${P} $(date) ====="
    python -u probe_attribution.py "${COMMON[@]}" \
        --num_parties "$P" \
        --exp_name "probe_attribution_p${P}"
done

echo "===== PROBES DONE $(date) ====="
echo "Look for ./results/probe_attribution_p*/probe_report.json"
