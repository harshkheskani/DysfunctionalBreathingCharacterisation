#!/bin/bash

#=========================================================================================
# SLURM JOB SUBMISSION SCRIPT FOR ENHANCED ATTENTION CNN WITH VISUALIZATION
# Assumes this script is located in: /path/to/project_folder/scripts/
#=========================================================================================

# --- SLURM Preamble ---
#SBATCH --job-name=enhanced-attention-cnn    # Descriptive name for enhanced CNN job
#SBATCH --gres=gpu:1                         # Single GPU for attention CNN
#SBATCH --time=48:00:00                      # Extended time for training + visualization generation
#SBATCH --mem=32G                            # More memory for visualization storage
#SBATCH --cpus-per-task=8                    # More CPUs for parallel visualization processing
#SBATCH --output=../logs/enhanced_attention_cnn_%j.out # Save logs one level up in a 'logs' folder
#SBATCH --error=../logs/enhanced_attention_cnn_%j.err

# --- Environment Setup ---
echo "======================================================"
echo "Enhanced Attention CNN Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# Activate Conda environment
# This line is robust and finds your conda installation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diss # <-- Make sure 'diss' is your environment name
echo "Conda environment 'diss' activated."
echo "Using python: $(which python)"

# Display GPU information
echo "GPU Information:"
nvidia-smi
echo "======================================================"

PROJECT_DIR="/home/s1978431/DysfunctionalBreathingCharacterisation"
cd $PROJECT_DIR
echo "Changed to $(pwd)"

# Path to the enhanced python script
PYTHON_SCRIPT="$PROJECT_DIR/src/Classification/saattention-visualization.py" # Updated script name

# Input data directory, relative to the project root
DATA_DIR="$PROJECT_DIR/data/bishkek_csr/03_train_ready"

# Base output directory for results, created relative to the project root
# The python script will create sub-folders for each run inside this
OUTPUT_DIR="$PROJECT_DIR/results/enhanced_attention_cd${SLURM_JOB_ID}"

# Create necessary directories
mkdir -p "$(dirname $OUTPUT_DIR)"
mkdir -p "$PROJECT_DIR/logs"

# --- Verify Dependencies ---
echo "Checking Python dependencies..."
python -c "
import torch
import matplotlib
import seaborn
import pandas
import numpy
import sklearn
import imblearn
print('✓ All required packages found')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# Set matplotlib backend for cluster environment (no display)
export MPLBACKEND=Agg
echo "Set matplotlib backend to Agg for headless operation"

# --- Execute the Enhanced Python Script ---

echo "======================================================"
echo "Starting Enhanced Attention CNN script: $PYTHON_SCRIPT"
echo "Data source: $DATA_DIR"
echo "Results will be saved to: $OUTPUT_DIR"
echo "Visualization enabled: True"
echo "======================================================"

# Run the Enhanced Python script for the CNN experiment with visualization.
# We pass the paths as command-line arguments and enable visualization.
python "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --base_output_dir "$OUTPUT_DIR" \
    --visualize

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "✓ Python script completed successfully"
else
    echo "✗ Python script failed with exit code $?"
    exit 1
fi

# --- Post-Processing and Results Summary ---

echo "======================================================"
echo "GENERATING RESULTS SUMMARY"
echo "======================================================"

# Create a summary of generated files
SUMMARY_FILE="$OUTPUT_DIR/job_summary.txt"
echo "Enhanced Attention CNN Job Summary" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "Hostname: $(hostname)" >> "$SUMMARY_FILE"
echo "Data Directory: $DATA_DIR" >> "$SUMMARY_FILE"
echo "Output Directory: $OUTPUT_DIR" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Count generated files
echo "Generated Files:" >> "$SUMMARY_FILE"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Total files: $(find $OUTPUT_DIR -type f | wc -l)" >> "$SUMMARY_FILE"
    echo "PNG files: $(find $OUTPUT_DIR -name "*.png" | wc -l)" >> "$SUMMARY_FILE"
    echo "Text reports: $(find $OUTPUT_DIR -name "*.txt" | wc -l)" >> "$SUMMARY_FILE"
    echo "Model checkpoints: $(find $OUTPUT_DIR -name "*.pt" | wc -l)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    echo "Directory structure:" >> "$SUMMARY_FILE"
    tree "$OUTPUT_DIR" >> "$SUMMARY_FILE" 2>/dev/null || ls -la "$OUTPUT_DIR" >> "$SUMMARY_FILE"
else
    echo "ERROR: Output directory not found!" >> "$SUMMARY_FILE"
fi

# Display summary
echo "======================================================"
echo "RESULTS SUMMARY:"
cat "$SUMMARY_FILE"
echo "======================================================"

# --- Cleanup (Optional) ---
# Remove temporary checkpoint files if desired
# find "$OUTPUT_DIR" -name "lono_checkpoint_fold_attn_*.pt" -delete

# --- Job Completion ---

echo "======================================================"
echo "Enhanced Attention CNN Job finished at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Key files to examine:"
echo "1. $OUTPUT_DIR/comprehensive_attention_report.txt"
echo "2. $OUTPUT_DIR/attention_analysis_initial/feature_importance_by_class.png"
echo "3. $OUTPUT_DIR/lono_results/confusion_matrix_lono_aggregated.png"
echo "======================================================"