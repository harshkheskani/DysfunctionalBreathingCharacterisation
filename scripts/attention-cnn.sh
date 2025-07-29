#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR CNN HYPERPARAMETER EXPERIMENT
# Assumes this script is located in: /path/to/project_folder/scripts/
#=========================================================================================

# --- SLURM Preamble ---
#SBATCH --job-name=cnn-attention-no-acccel    # A descriptive name for your CNN job
#SBATCH --gres=gpu:1                      # CNNs typically benefit most from a single powerful GPU per process
#SBATCH --time=35:00:00                   # Request more time, CNN training can be long
#SBATCH --output=../logs/cnn_attention_%j.out # Save logs one level up in a 'logs' folder
#SBATCH --error=../logs/cnn_attention_%j.err

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
echo "Using python: $(which python)"

PROJECT_DIR="/home/s1978431/DysfunctionalBreathingCharacterisation"
cd $PROJECT_DIR
echo "Changed to $(pwd)"

# Path to the python script to execute
PYTHON_SCRIPT="$PROJECT_DIR/src/Classification/attention_cnn_cluster.py" # Assuming your python script is in a 'src' folder

# Input data directory, relative to the project root
DATA_DIR="$PROJECT_DIR/data/bishkek_csr/03_train_ready"

# Base output directory for results, created relative to the project root
# The python script will create sub-folders for each run inside this
OUTPUT_DIR="$PROJECT_DIR/results/cd${SLURM_JOB_ID}"

# --- Execute the Python Script ---

echo "Starting Python script: $PYTHON_SCRIPT"
echo "Data source: $DATA_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the Python script for the CNN experiment.
# We pass the paths as command-line arguments.
python "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --base_output_dir "$OUTPUT_DIR"

# --- Job Completion ---

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"