#!/bin/bash

# Activate your Python virtual environment if you have one
# source /path/to/your/venv/bin/activate

echo "Starting XGBoost training run..."

# Config file is one directory up from 'src'
CONFIG_FILE="../configs/model/xgboost.yaml"

# The training script is in the 'models' subdirectory
TRAIN_SCRIPT="../src/models/train_xgboost.py"

# Run the python script from the 'src' directory
python $TRAIN_SCRIPT --config $CONFIG_FILE

echo "Training run finished."