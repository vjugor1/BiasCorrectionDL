#!/bin/bash
docker run -it \
-v $(pwd):/app/wind \
-v /mnt/datalake/platform-gisaai/esg/corrector:/app/wind/data \
-m 256000m --cpus=16 --gpus '"device=0"' \
--ipc=host \
--user="$(id -u):$(id -g)" \
-w="/app/wind" \
-e "WANDB_API_KEY=$WANDB_API_KEY" \
-e "WANDB_DATA_DIR=/app/wind/out" \
-e "WANDB_DIR=/app/wind/out" \
-e "WANDB_CACHE_DIR=/app/wind/out" \
--name s.lukashevich-corrector \
s.lukashevich:corrector
