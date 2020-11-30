#!/bin/bash
#SBATCH --account=def-jrwright
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:00:01
#SBATCH --mem-per-cpu=1G 
#SBATCH --gres=gpu:1

parallel --joblog log.log -j $SLURM_CPUS_PER_TASK < ./gpu_experiments.txt