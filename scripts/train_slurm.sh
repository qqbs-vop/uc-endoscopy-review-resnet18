#!/bin/bash
#SBATCH --job-name=uc-r18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
: "${UC_RUN_ID:=resnet18_H_seed29}"
python scripts/train.py --run "$UC_RUN_ID" --device cuda:0 --resume
