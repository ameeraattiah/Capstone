#!/bin/bash
#SBATCH --job-name=culturaX_data
#SBATCH --output=culturaX_data.out
#SBATCH --error=culturaX_data.err
#SBATCH --time=3:00:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G  

# Load required modules
module purge
module load python/3.11.0

# Store Hugging Face cache in home directory 
export HF_HOME=$HOME/.cache/huggingface
export DATASETS_CACHE=$HOME/.cache/huggingface/datasets

# Run the dataset download script
python culturaX.py

