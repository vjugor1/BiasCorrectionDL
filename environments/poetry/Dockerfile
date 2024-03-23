# Use NVIDIA CUDA devel image for GPU support
FROM nvidia/cuda:12.2.2-base-ubuntu22.04 

# Set non-interactive shell to avoid prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    ca-certificates \
    build-essential \ 
    wget \
    sudo \
    git \
    unzip \
    bzip2 \
    libx11-6 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

# Create a non-root user and set permissions
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID
RUN groupadd -g $DOCKER_GROUP_ID usergroup && \
    useradd -m -u $DOCKER_USER_ID -g usergroup -s /bin/bash user

# Install poetry using pip and configure it to create virtual environments within the project directory
COPY poetry.lock pyproject.toml ./
RUN python -m pip install --no-cache-dir poetry==1.8.2 \
    && poetry config virtualenvs.in-project true \
    && poetry lock --no-update \
    && poetry install --no-interaction --no-ansi

# Create and set the working directory
# RUN mkdir /app && chown -R user:usergroup /app
WORKDIR /app

# Switch to the non-root user
# USER user

ENV WANDB_API_KEY="SECRET"

# Define default command use the new environment:
RUN echo "poetry shell" >> ~/.bashrc