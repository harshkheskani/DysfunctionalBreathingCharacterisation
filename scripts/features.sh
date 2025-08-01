#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR FEATURE EXTRACTION
#=========================================================================================

# --- SLURM Preamble ---
#SBATCH --job-name=features_extract  # A descriptive name for the job
#SBATCH --cpus-per-task=12           # This is a CPU-intensive task, 12 is a good number
#SBATCH --mem=128G                    # Request a good amount of memory
#SBATCH --time=010:00:00              # Request 5 hours, adjust if needed
#SBATCH --output=../logs/features_%j.out # Save logs one level up in a 'logs' folder
#SBATCH --error=../logs/features_%j.err

# --- Environment Setup ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diss # <-- Make sure 'diss' is your environment name
echo "Conda environment 'diss' activated."
echo "Using python: $(which python)"

# --- Project and Data Paths ---
PROJECT_DIR="/home/s1978431/DysfunctionalBreathingCharacterisation"
cd $PROJECT_DIR
echo "Changed to project directory: $(pwd)"

# Path to the python script to execute
PYTHON_SCRIPT="$PROJECT_DIR/src/Classification/xgBoost2.py"

# Define the specific input directories based on your data structure
BASE_DATA_DIR="$PROJECT_DIR/data/bishkek_csr/03_train_ready"
EVENTS_DIR="$BASE_DATA_DIR/event_exports"
RESPECK_DIR="$BASE_DATA_DIR/respeck"

# Base output directory for results. A unique sub-folder will be created.
OUTPUT_DIR="$PROJECT_DIR/results/features_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR" # Create the output directory to avoid errors

# --- Execute the Python Script ---
echo "Starting Python script: $PYTHON_SCRIPT"
echo "Events source: $EVENTS_DIR"
echo "Respeck source: $RESPECK_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the Python script, passing the paths as named arguments
python "$PYTHON_SCRIPT" \
    --events_folder "$EVENTS_DIR" \
    --respeck_folder "$RESPECK_DIR" \
    --output_dir "$OUTPUT_DIR"

# --- Job Completion ---
echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"