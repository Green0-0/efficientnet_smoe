#!/bin/bash
#SBATCH --job-name=dataset_download
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source .venv/bin/activate

python scripts/download_dataset.py