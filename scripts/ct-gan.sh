#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR CTGAN DATA GENERATION AND MODEL TRAINING
# Assumes this script is located in: /path/to/project_folder/scripts/
#=========================================================================================

# --- SLURM Preamble ---
# Job Name: Descriptive name for your job
#SBATCH --job-name=ctgan_rf_training

# Partition: The cluster partition to run on. 'gpu' is fine, even though this
# script doesn't heavily use the GPU, it often has nodes with more memory.
# You could also use a 'cpu' or 'high-mem' partition if available.
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # CTGAN and RandomForest can use multiple cores

# Log Files: Save logs to a 'logs' folder, one level up from the script's location
#SBATCH --output=../logs/ctgan_training_%j.out
#SBATCH --error=../logs/ctgan_training_%j.err


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


# --- Execute the Python Script ---
echo "Locating project and script paths..."

# Get the absolute path to the directory where this script lives (e.g., /path/to/project/scripts)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo "Script directory: $SCRIPT_DIR"

# The project root is one directory above the script's directory
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
echo "Project directory: $PROJECT_DIR"

# Path to the python script to execute
# Make sure you have saved the previous Python script with this name in your 'src/gan/' folder
PYTHON_SCRIPT="$PROJECT_DIR/src/gan/ct-gan-rf.py"
echo "Python script to execute: $PYTHON_SCRIPT"

echo "------------------------------------------------------"
echo "Starting Python script for CTGAN training..."
echo "------------------------------------------------------"

# The Python script is now self-contained and creates its own results folder.
# We no longer need to pass directories as arguments.
python "$PYTHON_SCRIPT"


# --- Job Completion ---
echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"