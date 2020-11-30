#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --array=1-4
#SBATCH --time=00:00:05
#SBATCH --mem=2G 
#SBATCH --gres=gpu:1 

head -n $SLURM_ARRAY_TASK_ID gpu_experiments.txt | tail -n 1