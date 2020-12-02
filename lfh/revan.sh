#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --array=1-54
#SBATCH --time=14:00:00
#SBATCH --mem=12G 
#SBATCH --gres=gpu:1 
#SBATCH --output=slurm/%x-%j.out

eval $(head -n $SLURM_ARRAY_TASK_ID experiments_revan.txt | tail -n 1)