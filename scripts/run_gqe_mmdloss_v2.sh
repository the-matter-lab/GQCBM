#!/bin/bash
#SBATCH --job-name=gqe_mmdloss2
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:7
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_mmdloss2_%j.out
#SBATCH --error=slurm_logs/gqe_mmdloss2_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# GQCBM trained with MMD as the optimisation metric (--loss_metric mmd),
# for the like-for-like MMD-vs-MMD comparison in Supplementary S10.4.
# Bandwidths match qcbm_pennylane: continuous [0.25,60], discrete [0.25,0.5,1].
# --kl_temp_std_coef 0.5 --kl_temp_min 1e-6 makes the temperature auto-scale to the MMD range.
# 15q omitted: kernel matrix 32768x32768 (~8.6 GB) is impractical.

run() {  # $1=name $2=nq $3=ngates $4=iters $5=nsamp $6=kind $7=kw $8=sigmas $9=gpu
  CUDA_VISIBLE_DEVICES=$9 python -u gqe_single_target.py \
    --n_qubits $2 --n_gates $3 --angle_bins_K 12 \
    --max_iters $4 --num_samples $5 --lr 3e-4 \
    --target_kind $6 --target_kw "$7" \
    --mmd_sigmas "$8" --loss_metric mmd --kl_temp_std_coef 0.5 --kl_temp_min 1e-6 \
    --seed 42 --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/gqe_mmdloss2_$1.pt \
    --log_dir gqe_experiments/mmdloss2_$1 \
    > slurm_logs/gqe_mmdloss2_$1.log 2>&1 &
}

run bell_2q     2  8  1500 50 ghz         '{}'                              '0.25,0.5,1.0' 0
run ghz5_5q     5  20 3000 50 ghz         '{}'                              '0.25,0.5,1.0' 1
run bas4_4q     4  16 2500 50 bas         '{}'                              '0.25,0.5,1.0' 2
run gauss_5q    5  20 2000 50 gaussian    '{"center":0.5,"sigma":0.20}'     '0.25,60.0'    3
run bimodal_6q  6  20 2000 50 bimodal     '{"c1":0.2857,"c2":0.7143,"sigma":0.125}' '0.25,60.0' 4
run exp_5q      5  20 2000 50 exponential '{}'                              '0.25,60.0'    5
run gauss_10q   10 30 4000 30 gaussian    '{"center":0.5,"sigma":0.10}'     '0.25,60.0'    6

wait
echo "GQCBM MMD-loss training (7 targets) done"
