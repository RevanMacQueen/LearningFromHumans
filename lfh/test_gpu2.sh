#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --time=00:00:05
#SBATCH --mem=2G 
#SBATCH --gres=gpu:1 
#SBATCH --output=slurm/%x-%j.out

python gpu_test.py 