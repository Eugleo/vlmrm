#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

models=("gpt4")

for model in "${models[@]}"; do
    python src/evaluation/multiclass_evaluator.py -d data/habitat/data.csv -t data/habitat/tasks.yaml -m "$model" -r logit,projection -a 0.0,0.25,0.50,0.75,1.0 -n 32 -e gpt4_test
done
