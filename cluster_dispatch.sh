#!/bin/bash
# Launches 8 GPU workers on the cluster. Distributes the Phase 2 + Phase 3
# combo queue (56 combos total) round-robin across GPUs 0..7.
#
# Phase 2 = 3 P1 methods × 7 cells = 21 combos
# Phase 3 = 5 P2 methods × 7 cells = 35 combos
# 8 GPUs × 7 combos each ≈ 7 combos per GPU.
#
# Idempotent (workers skip combos that already have aggregated.json).
set -u
cd /public/home/dongshou/projects/VF-CUL

P1_METHODS=(er_ace:luv proto_evolve:gradient_ascent proto_evolve:fedup)
P2_METHODS=(proto_evolve:fucrt proto_evolve:fedau proto_evolve:fudp er:luv proto_evolve:mode)
# Cells with their (dsname, P) — 7 total
CELLS=("c10:1" "c10:4" "c100:1" "c100:4" "tin:1" "tin:2" "tin:4")

# Build the 56-item priority queue: P1 methods first (3 × 7 = 21),
# then P2 methods (5 × 7 = 35).  Within each phase, iterate cells inner
# so each (method, cell) pair appears once.
COMBOS=()
for m in "${P1_METHODS[@]}" "${P2_METHODS[@]}"; do
    for c in "${CELLS[@]}"; do
        COMBOS+=("${m}:${c}")
    done
done
echo "Total combos to dispatch: ${#COMBOS[@]}"

mkdir -p logs/cluster_phase23

# Round-robin to 8 GPUs.
for gpu in 0 1 2 3 4 5 6 7; do
    QUEUE=()
    for i in "${!COMBOS[@]}"; do
        if (( i % 8 == gpu )); then
            QUEUE+=("${COMBOS[$i]}")
        fi
    done
    echo ""
    echo "GPU $gpu queue (${#QUEUE[@]} combos):"
    printf "  %s\n" "${QUEUE[@]}"
    nohup bash cluster_worker.sh "$gpu" "${QUEUE[@]}" \
        > logs/cluster_phase23/gpu${gpu}.log 2>&1 &
    echo "  launched PID $!"
done
echo ""
echo "All 8 GPU workers launched. Tail logs/cluster_phase23/gpu*.log for progress."
