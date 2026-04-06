#!/bin/bash
#SBATCH --job-name=a5000_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1

source .venv/bin/activate

python ai_slop/actually_works/baseline_fast_sweep_log.py