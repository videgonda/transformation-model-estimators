#!/bin/zsh
# Regenerate the engression Monte Carlo results for all three designs under
# Z ~ N(0,1), using the tuned hyperparameters documented in the thesis
# (Table: Network, training, and extraction parameters).
#
# Usage:  zsh run_engression.sh [n_reps]     (default n_reps = 1000)
#
# The three designs are independent and run in parallel; each writes its own
# CSV/PNG and a log file eng_run_design_<d>.log. MPLBACKEND=Agg keeps the
# plt.show() calls from blocking in a headless/overnight run.

cd "$(dirname "$0")"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export PYTHONUNBUFFERED=1   # flush progress to logs so the run is monitorable

NREPS=${1:-1000}
COMMON="--n_reps ${NREPS} --num_layer 4 --no_bn --lr 5e-4 --num_epoches 2000"
# (n_samples=1000, n_eval=50, seed=42, hidden_dim=100, noise_dim=100 are script defaults;
#  extraction params 40 anchors / [0.025,0.975] / 2000 samples / window [0.05,0.95] are defaults too.)

echo "Starting engression MC: n_reps=${NREPS}, designs 1,2,3 in parallel ($(date))"
for d in 1 2 3; do
    python3 engression_monte_carlo.py --design ${d} ${=COMMON} > eng_run_design_${d}.log 2>&1 &
done
wait
echo "ALL DESIGNS COMPLETE ($(date))"
