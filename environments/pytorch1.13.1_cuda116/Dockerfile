FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y tzdata

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install gcc 7
RUN apt update -qq \
    && apt install -yq software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt update -qq \
    && apt install -yq g++-7 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Configure alias
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-7 \
    --slave /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-7 \
    --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 \
    --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7

RUN mkdir /app
WORKDIR /app

RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
RUN useradd -u 1031 -ms /bin/bash username

USER user


ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ENV WANDB_API_KEY="SECRET"

CMD ["python3"]
