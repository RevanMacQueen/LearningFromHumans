#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --array=1-3
#SBATCH --time=00:10:00
#SBATCH --mem=5G 
#SBATCH --gres=gpu:1 
#SBATCH --output=slurm/%x-%j.out

eval $(head -n $SLURM_ARRAY_TASK_ID small_experiment.txt | tail -n 1)