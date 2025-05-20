#!/bin/bash
#SBATCH --job-name=eval_batch
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=errors/eval_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=160GB
#SBATCH --gres=gpu:1
#SBATCH --array=0  # 🧠 5 tasks: job IDs 0,1,2,3,4 (total 5000 samples)

# Compute offset for this task
OFFSET=$(( SLURM_ARRAY_TASK_ID * 1000 ))
echo "🚀 Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID with OFFSET=$OFFSET"

# Load environment
source /ibex/user/attiahas/mambaforge/etc/profile.d/conda.sh
conda activate /ibex/user/attiahas/conda-environments/capstone_env

# Set library paths
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/include:$CPATH

module purge
module load cuda/12.2
module load sqlite/3.40.1

# Debugging
echo "Python binary: $(which python)"
python --version

# Environment variables
export PYTHONPATH="/ibex/user/attiahas/python_libs/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/ibex/user/attiahas/python_libs/bin:$PATH"
export CUDA_HOME=/sw/rl9g/cuda/12.2/rl9_binary
export HF_HOME="/ibex/user/attiahas/.cache/huggingface"
export DS_ACCELERATOR=cuda

# Run the Python evaluation
python /ibex/user/attiahas/Code/Academic/Evaluate.py \
  --academic "/ibex/user/attiahas/Code/data/Academic_Data.json" \
  --non_academic "/ibex/user/attiahas/Code/data/Non_Academic_Data.json" \
  --max_samples 5000 \
  --offset "$OFFSET"
