#!/bin/bash

#SBATCH --job-name=corrector_gpu_test        # Job name
#SBATCH --partition=gpu                      # Queue name 
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=v.shevchenko@skoltech.ru # Where to send mail

#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Number of tasks per node
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --gpus-per-task=1                   # Number of GPUs per task
#SBATCH --mem=350G                          # Memory per node   

#SBATCH --time=24:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/trinity/home/v.shevchenko/logs/parallel_%j.log   
#SBATCH --error=/trinity/home/v.shevchenko/logs/parallel_error_%j.log

srun singularity exec --nv biascorrectiondl.sif /opt/conda/envs/bias_correction/bin/python cmip6_era5_dl.py 