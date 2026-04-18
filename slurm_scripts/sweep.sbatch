#!/bin/bash
#SBATCH --job-name=a6000_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1

source .venv/bin/activate

python scripts/deepmoe.py