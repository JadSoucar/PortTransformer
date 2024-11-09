#!/bin/bash

# Make sure this script has execute permissions with chmod +x run_all.sh

echo "Downloading Data..."
python data.py

echo "Starting hyperparameter tuning..."
python tune.py

echo "Starting training..."
python train.py

echo "Starting evaluation..."
python eval.py

echo "All tasks completed."
