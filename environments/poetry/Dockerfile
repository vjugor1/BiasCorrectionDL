# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.2-base-ubuntu22.04 as builder

# Set non-interactive shell to avoid prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update \
    && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim

# Create a non-root user and set permissions
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

COPY poetry.lock pyproject.toml ./

RUN python -m pip install --no-cache-dir poetry==1.8.2 \
    && poetry config virtualenvs.in-project true \
    && poetry lock --no-update \
    && poetry install --no-interaction --no-ansi

# Create and set the working directory
RUN mkdir /app &&\
    adduser --disabled-password --uid $DOCKER_USER_ID --gecos "" --shell /bin/bash user && \
    chown -R user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user

WORKDIR /app
ENV WANDB_API_KEY="SECRET"

# Define default command use the new environment:
RUN echo "poetry shell" >> ~/.bashrc