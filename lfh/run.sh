#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5G 
#SBATCH --gres=gpu:1

parallel --joblog log.log -j $SLURM_CPUS_PER_TASK < ./experiments.txt