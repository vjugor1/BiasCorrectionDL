#!/bin/bash

# Script to run the BiasCorrectionDL Docker container

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker could not be found. Please install Docker."
    exit 1
fi

# Configuration options (could be passed as arguments or modified here)
MEMORY_LIMIT="256000m"
CPUS="32"
GPUS="device=0,1,2,3,4,5"
CONTAINER_NAME="$(id -un)-corrector_1"
PROJECT_DIR="/home/$(id -un)/BiasCorrectionDL"
DATA_DIR="/mnt/ssd/bias_correction"
IMAGE_NAME="s.lukashevich/bias:3.0"
WANDB_DIR="/app/out"

# Run the Docker container
docker run -it --name "${CONTAINER_NAME}" \
    -v "${PROJECT_DIR}:/app/" \
    -v "${DATA_DIR}:/app/data" \
    -v /mnt/public-datasets/d.tanyushkina/downscaling/raw:/app/data/raw \
    -v /mnt/public-datasets/d.tanyushkina/downscaling/experiments:/app/data/experiments \
    -m "${MEMORY_LIMIT}" --cpus="${CPUS}" --gpus '"'"${GPUS}"'"' \
    --ipc=host \
    -w="/app" \
    -e "WANDB_API_KEY=${WANDB_API_KEY}" \
    -e "WANDB_DATA_DIR=${WANDB_DIR}" \
    -e "WANDB_DIR=${WANDB_DIR}" \
    -e "WANDB_CACHE_DIR=${WANDB_DIR}" \
    "${IMAGE_NAME}"

# Check if the Docker command succeeded
if [ $? -ne 0 ]; then
    echo "Failed to start the Docker container."
    exit 1
fi

echo "Docker container ${CONTAINER_NAME} started successfully."
