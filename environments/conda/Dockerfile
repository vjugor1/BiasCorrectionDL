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

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN conda --version

# Create the environment:
COPY environment.yml .
COPY . .
# create conda environment
RUN conda init bash \
    && . ~/.bashrc \
    && conda env create -f environment.yml

WORKDIR /app

# Switch to the non-root user
# USER user

ENV WANDB_API_KEY="SECRET"

# Make RUN commands use the new environment:
RUN echo "conda activate bias_correction" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
