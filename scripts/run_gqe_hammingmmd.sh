#!/bin/bash
#SBATCH --job-name=gqe_hamMMD
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_hamMMD_%j.out
#SBATCH --error=slurm_logs/gqe_hamMMD_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# Hamming-distance MMD on structured discrete targets stuck under forward KL.
# Bandwidths sigma in [0.5, 1, 2, 4] (Hamming distance ranges 0..n_qubits).
# std-coef + tiny T_min so the temperature auto-scales to the MMD range.

run() {  # $1=name $2=nq $3=ngates $4=iters $5=nsamp $6=kind $7=kw $8=gpu
  CUDA_VISIBLE_DEVICES=$8 python -u gqe_single_target.py \
    --n_qubits $2 --n_gates $3 --angle_bins_K 12 \
    --max_iters $4 --num_samples $5 --lr 3e-4 \
    --target_kind $6 --target_kw "$7" \
    --loss_metric mmd --mmd_kernel hamming --mmd_sigmas '0.5,1.0,2.0,4.0' \
    --kl_temp_std_coef 0.5 --kl_temp_min 1e-6 \
    --seed 42 --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/gqe_hamMMD_$1.pt \
    --log_dir gqe_experiments/hamMMD_$1 \
    > slurm_logs/gqe_hamMMD_$1.log 2>&1 &
}

run bas9_9q         9  30 4000 50 bas         '{}'        0
run ghz10_10q       10 30 4000 30 ghz         '{}'        1
run cardinality_5q  5  20 2000 50 cardinality '{"k":2}'   2
run cardinality_10q 10 30 4000 30 cardinality '{"k":5}'   3

wait
echo "Hamming-MMD training (4 stuck targets) done"
