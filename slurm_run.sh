#!/bin/bash

#SBATCH --job-name=corrector_gpu_unet        # Job name
#SBATCH --partition=ais-gpu                  # Queue name 
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=v.shevchenko@skoltech.ru # Where to send mail

#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gpus=2                            # Number of GPUs per task
#SBATCH --mem=300G                          # Memory per node   

#SBATCH --time=144:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/trinity/home/aleksandr.lukashevich/logs_corrector/parallel_%j.log   
#SBATCH --error=/trinity/home/aleksandr.lukashevich/logs_corrector/parallel_error_%j.log

srun singularity exec --nv biascorrectiondl.sif /opt/conda/envs/bias_correction/bin/python cmip6_era5_dl.py 
