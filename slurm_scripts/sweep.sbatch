#!/bin/bash
#SBATCH --job-name=a5000_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:a5000:1

source .venv/bin/activate

python ai_slop/baseline_sweep.py