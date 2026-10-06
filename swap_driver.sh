#!/bin/bash
# Wait for current driver's c10_4p Oracle to finish (aggregated.json appears),
# then kill old driver + python procs, launch the new 13-method driver.
cd /home/chase/ct/VF-CUL

echo "=== watching for c10_4p Oracle to finish ==="
while true; do
    if ls results/frag/c10_4p/oracle/*/aggregated.json >/dev/null 2>&1; then
        echo "$(date): c10_4p Oracle DONE. Swapping driver..."
        break
    fi
    sleep 300  # 5 min poll
done

# Kill old driver and any running python combo (may have started next combo)
pkill -9 -f run_fragmentation.sh 2>/dev/null
sleep 2
pkill -9 -f "main.py --cl_method" 2>/dev/null
pkill -9 -f "run_oracle_solo.py" 2>/dev/null
sleep 5

# Verify killed
echo "after kill:"; pgrep -af "main.py --cl_method\|run_fragmentation.sh\|run_oracle_solo" | grep -v pgrep | head

# Restart with new (already-staged) 13-method driver
source /home/chase/miniconda3/etc/profile.d/conda.sh
conda activate FedEMoE
nohup bash run_fragmentation.sh > /tmp/frag.log.swapped 2>&1 &
echo "$(date): new 13-method driver launched PID $!"
echo "log: /tmp/frag.log.swapped"
