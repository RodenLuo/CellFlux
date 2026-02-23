#!/bin/bash
#SBATCH --job-name=cellflux_infer
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/deng.luo/projects/CellFlux/single_cell_inference/logs/inference_%j.out
#SBATCH --error=/home/deng.luo/projects/CellFlux/single_cell_inference/logs/inference_%j.err

# ── Usage ─────────────────────────────────────────────────────────
# cd /home/deng.luo/projects/CellFlux/single_cell_inference
#
# Single prediction:
#   sbatch scripts/run_inference.sh --drug_name taxol --cell_image example_input/control_cell.npy
#
# API server (long-running):
#   sbatch --time=04:00:00 scripts/run_inference.sh --api --port 5000
#
# All extra arguments after the script name are passed to inference.py
# ──────────────────────────────────────────────────────────────────

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellflux

SCRIPT_DIR="/home/deng.luo/projects/CellFlux/single_cell_inference"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "Arguments: $@"
echo "---"

python inference.py "$@"
