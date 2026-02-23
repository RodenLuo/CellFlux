#!/bin/bash
#SBATCH --job-name=cellflux_batch
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/deng.luo/projects/CellFlux/single_cell_inference
#SBATCH --output=/home/deng.luo/projects/CellFlux/single_cell_inference/logs/batch_%j.out
#SBATCH --error=/home/deng.luo/projects/CellFlux/single_cell_inference/logs/batch_%j.err

# ── Usage ─────────────────────────────────────────────────────────
# sbatch scripts/run_batch_inference.sh
# sbatch scripts/run_batch_inference.sh --num_samples 200 --score_threshold 80
# ──────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellflux

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "Arguments: $@"
echo "---"

python batch_inference.py "$@"
