#!/bin/sh
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --job-name=download_warc
#SBATCH --output=logs/warc_download_%A_%a.log

# Purge any loaded modules to avoid conflicts
module purge

# Set library paths before loading modules
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/include:$CPATH

# Load Python module
module load python/3.11.0

# Ensure Python uses the correct library paths
export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/python_libs/lib/python3.11/site-packages:$PYTHONPATH
export PATH=$HOME/python_libs/bin:$PATH

# Define paths
WARC_PATHS_FILE="/ibex/user/abuhanjt/test/CC-MAIN-2024-42/warc.paths"
PROCESSED_WARC_LIST="/ibex/user/abuhanjt/test/CC-MAIN-2024-42/processed_warc_files.txt"
DOWNLOAD_DIR="/ibex/user/abuhanjt/test/CC-MAIN-2024-42/downloaded_warc_files"

# Create output directories if they don’t exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p logs

# Read all WARC files
TOTAL_FILES=$(wc -l < "$WARC_PATHS_FILE")
FILES_PER_JOB=1800 

# Assign files uniquely to each job
START_INDEX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
END_INDEX=$((START_INDEX + FILES_PER_JOB - 1))
if [[ $END_INDEX -ge $TOTAL_FILES ]]; then
    END_INDEX=$((TOTAL_FILES - 1))
fi

# Extract the correct files for this job (excluding already processed ones)
TMP_FILE="/tmp/warc_list_$SLURM_ARRAY_TASK_ID.txt"
touch "$TMP_FILE"

awk 'NR>='"$START_INDEX"' && NR<='"$END_INDEX"'' "$WARC_PATHS_FILE" > "$TMP_FILE"

# Use a global lock to prevent duplicate processing
exec 3>/ibex/user/abuhanjt/test/CC-MAIN-2024-42/processed_warc.lock
flock -x 3

# Ensure already processed files are removed
grep -Fxv -f "$PROCESSED_WARC_LIST" "$TMP_FILE" > "${TMP_FILE}_filtered"
mv "${TMP_FILE}_filtered" "$TMP_FILE"

# Release lock
exec 3>&-

# Run Python script with unique files
python3 /ibex/user/abuhanjt/test/download_warc.py $(cat "$TMP_FILE")

# Cleanup temp file
rm -f "$TMP_FILE"
