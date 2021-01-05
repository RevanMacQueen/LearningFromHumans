#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --array=1-108
#SBATCH --time=02-12:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=revan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/%x-%j.out

eval $(head -n $SLURM_ARRAY_TASK_ID experiments_revan.txt | tail -n 1)