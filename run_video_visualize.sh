#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

python src/evaluation/multiclass_evaluator.py -d data/habitat/data_video_visualize.csv -t data/habitat/tasks.yaml -m clip -r logit -n 360 --window_size=16 --window_step=1
