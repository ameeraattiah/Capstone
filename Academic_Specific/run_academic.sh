#!/bin/bash
#SBATCH --job-name=academic_specific
#SBATCH --output=logs/academic_specific_%A_%a.out
#SBATCH --error=errors/academic_specific_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Define paths
DATASET_DIR="/ibex/user/attiahas/Code/dataset"
DATA_FILES=($(ls $DATASET_DIR/Arabic_Data_*.json| sort))

# Print debug info
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Total dataset files found: ${#DATA_FILES[@]}"
echo "Files in dataset: ${DATA_FILES[@]}"

# Ensure SLURM_ARRAY_TASK_ID is within range
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#DATA_FILES[@]}" ]; then
    echo "No file to process for task ID $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Assign dataset file
FILE_TO_PROCESS=${DATA_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Processing file: $FILE_TO_PROCESS"

# Run script with DeepSpeed to enable model parallelism
deepspeed --num_gpus=1 Academic.py "$FILE_TO_PROCESS"

# python Academic.py "$FILE_TO_PROCESS"
