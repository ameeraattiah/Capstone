#!/bin/bash
#SBATCH --job-name=test3
#SBATCH --output=logs/test3_%j.out
#SBATCH --error=errors/test3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32 
#SBATCH --mem=160GB 
#SBATCH --gres=gpu:1  

# Load environment
source /ibex/user/attiahas/mambaforge/etc/profile.d/conda.sh
conda activate /ibex/user/attiahas/conda-environments/capstone_env

# Set library paths before loading modules
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/include:$CPATH

module purge
# ✅ Print Debugging Information
echo "Running Python script..."
which python  # Check if Python is found
python --version  # Show Python version

module load cuda/12.2
module load sqlite/3.40.1  

echo "Starting test script execution..."

# Ensure Python finds installed packages in $HOME/python_libs
export PYTHONPATH="/ibex/user/attiahas/python_libs/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/ibex/user/attiahas/python_libs/bin:$PATH"
export CUDA_HOME=/sw/rl9g/cuda/12.2/rl9_binary
export HF_HOME="/ibex/user/attiahas/.cache/huggingface"
export DS_ACCELERATOR=cuda

# Run test 3 script
python /ibex/user/attiahas/Code/Academic/test3.py
