#!/bin/bash
#SBATCH --partition=batch
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --array=0-19 
#SBATCH --job-name=multi_warc_processing
#SBATCH --output=logs/output_%A_%a.log

# Set library paths before loading modules
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/include:$CPATH

# Unload conflicting modules
module purge

# Load Python and SQLite modules
module load python/3.11
module load sqlite/3.40.1

# Ensure Python finds installed packages in $HOME/python_libs
export PYTHONPATH=$HOME/python_libs/lib/python3.11/site-packages:$PYTHONPATH
export PATH=$HOME/python_libs/bin:$PATH

# Define directories
WARC_DIR="CC-MAIN-2024-42/downloaded_warc_files"
OUTPUT_DIR="Output"

# Get list of unprocessed files
PROCESSED_FILES=($(ls ${OUTPUT_DIR} | sed 's/processed_//g' | sed 's/.json//g'))
UNPROCESSED_FILES=()
for file in $(ls ${WARC_DIR}); do
    if [[ ! " ${PROCESSED_FILES[@]} " =~ " ${file} " ]]; then
        UNPROCESSED_FILES+=("${WARC_DIR}/${file}")
    fi
done

NUM_FILES=${#UNPROCESSED_FILES[@]}
FILES_PER_JOB=10  # Each job processes 10 files

# Get the batch of files for this job
START_INDEX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
END_INDEX=$((START_INDEX + FILES_PER_JOB))

# Ensure we don’t go beyond the available files
if [[ $START_INDEX -ge $NUM_FILES ]]; then
    echo "No more files to process for job ${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 0
fi

# Get the actual files for this job
FILES_TO_PROCESS=("${UNPROCESSED_FILES[@]:$START_INDEX:$FILES_PER_JOB}")

echo "Processing ${#FILES_TO_PROCESS[@]} files in job ${SLURM_ARRAY_TASK_ID}"

# Run pipeline for each file in this batch
for WARC_FILE in "${FILES_TO_PROCESS[@]}"; do
    python /ibex/user/abuhanjt/test/pipeline.py "$WARC_FILE"
done
