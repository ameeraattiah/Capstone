#!/bin/bash
#SBATCH --job-name=acegpt_test
#SBATCH --output=logs/acegpt_test_%j.out
#SBATCH --error=errors/acegpt_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32 
#SBATCH --mem=160GB 
#SBATCH --gres=gpu:1  

# Set library paths before loading modules
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/include:$CPATH

module purge
module load python/3.11
module load cuda/12.2
module load sqlite/3.40.1  

# Ensure Python finds installed packages in $HOME/python_libs
export PYTHONPATH=$HOME/python_libs/lib/python3.11/site-packages:$PYTHONPATH
export PATH=$HOME/python_libs/bin:$PATH
export CUDA_HOME=/sw/rl9g/cuda/12.2/rl9_binary
export HF_HOME=$HOME/.cache/huggingface
export DS_ACCELERATOR=cuda

# Run AceGPT-7B test
python test.py
