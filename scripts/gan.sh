#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR TIME-SERIES GAN TRAINING
# Assumes this script is located in: /path/to/project_folder/scripts/
#=========================================================================================

# --- SLURM Preamble: Job Configuration ---
#SBATCH --job-name=tsgan_apnea_training # Job Name: Descriptive name for your job
#SBATCH --gpus=1                  # GPU Request: Our script uses one GPU
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8               # CPU cores per task: for data loading, etc.
#SBATCH --mem=64G                       # Memory request: Adjust as needed

# Log Files: Save logs to a 'logs' folder, one level up from the script's location
# The %j variable is replaced by the SLURM job ID.
#SBATCH --output=../logs/tsgan_training_%j.out
#SBATCH --error=../logs/tsgan_training_%j.err

# --- Environment Setup ---
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# Activate Conda environment
# This line is robust and finds your conda installation.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diss # <-- IMPORTANT: Make sure 'diss' is your environment name
echo "Conda environment 'diss' activated"

PROJECT_DIR=$(dirname $SLURM_SUBMIT_DIR)
echo "Project root directory set to: $PROJECT_DIR"

# Path to the python script to execute
PYTHON_SCRIPT="$PROJECT_DIR/src/gan/train_gan.py"
echo "Python script to execute: $PYTHON_SCRIPT"

# --- Define Experiment Parameters ---
# This is the best place to configure your run.
# These variables will be passed as command-line arguments to the Python script.
DATA_PATH="$PROJECT_DIR/data/bishkek_csr/"
OUTPUT_DIR="$PROJECT_DIR/outputs/gan_experiment_$(date +%Y%m%d_%H%M%S)" # Create a unique output folder for each run
EPOCHS=10000
BATCH_SIZE=128
LEARNING_RATE=0.0002
LATENT_DIM=100

# --- Execute the Python Script ---
echo "------------------------------------------------------"
echo "Starting Python script for Time-Series GAN training..."
echo "Running with the following parameters:"
echo "  - Data Path: $DATA_PATH"
echo "  - Output Dir: $OUTPUT_DIR"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Latent Dim: $LATENT_DIM"
echo "------------------------------------------------------"

# Execute the script, passing the variables as arguments
python "$PYTHON_SCRIPT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --latent_dim "$LATENT_DIM"

# --- Job Completion ---
echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"
