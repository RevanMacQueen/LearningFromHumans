#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --array=1-54
#SBATCH --time=02-12:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1 
#SBATCH --output=slurm/%x-%j.out
#SBATCH --mail-user=revan@ualberta.ca
#SBATCH --mail-type=ALL

eval $(head -n $SLURM_ARRAY_TASK_ID experiments_david.txt | tail -n 1)