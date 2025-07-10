#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR XGBOOST HYPERPARAMETER TUNING
# Assumes this script is located in: /path/to/project_folder/scripts/
#=========================================================================================

# --- SLURM Preamble ---
#SBATCH --job-name=xgboost_tuning
#SBATCH --gpus=2
#SBATCH --time=06:00:00
#SBATCH --output=../logs/xgboost_tuning_%j.out # Save logs one level up in a 'logs' folder
#SBATCH --error=../logs/xgboost_tuning_%j.err

# --- Environment Setup ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# Activate Conda environment
# This line is robust and finds your conda installation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diss # <-- Make sure 'diss' is your environment name
echo "Conda environment 'diss' activated."

# --- Define Paths Robustly ---

PROJECT_DIR="/home/s1978431/DysfunctionalBreathingCharacterisation"
cd $PROJECT_DIR
echo "Changed to $(pwd)"

# Path to the python script to execute
PYTHON_SCRIPT="$PROJECT_DIR/src/xgBoost.py" # Assuming your python script is in a 'src' folder

# Input data directory, relative to the project root
DATA_DIR="$PROJECT_DIR/data/bishkek_csr/03_train_ready"

# Output directory for results, created relative to the project root
OUTPUT_DIR="$PROJECT_DIR/results/xgboost_run_${SLURM_JOB_ID}"

# --- Execute the Python Script ---

echo "Starting Python script: $PYTHON_SCRIPT"
echo "Data source: $DATA_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

python "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

# --- Job Completion ---

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"
