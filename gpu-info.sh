#!/bin/bash
#SBATCH --job-name=check-gpu
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=gpu-info.out

nvidia-smi