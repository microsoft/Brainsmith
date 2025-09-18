# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM ubuntu:22.04
LABEL maintainer="Thomas Keller <thomaskeller@microsoft.com>, Mahdi Ghandi <Mahdi.Ghandi@microsoft.com>"

# ARG ENTRYPOINT - removed: no longer used for backward compatibility

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev-i386 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    nano \
    zsh \
    rsync \
    git \
    openssh-client \
    sshpass \
    wget \
    sudo \
    unzip \
    zip \
    locales \
    lsb-core \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-setuptools-scm \
    python3-venv \
    pybind11-dev \
    libfmt-dev \
    libboost-dev \
    libjansson-dev \
    libgetdata-dev \
    libtinfo5 \
    g++-10 \
    python3-poetry
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN locale-gen "en_US.UTF-8"

# Configure Poetry for container use
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false

# Install dependencies using Poetry
COPY pyproject.toml poetry.lock* /tmp/
WORKDIR /tmp
RUN poetry install --no-root --only main && \
    rm -rf /tmp/pyproject.toml /tmp/poetry.lock
WORKDIR /

# Xilinx toolflow specific imports
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy all entrypoint scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/entrypoint_fast.sh /usr/local/bin/entrypoint_fast.sh
COPY docker/setup_env.sh /usr/local/bin/setup_env.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/entrypoint_fast.sh /usr/local/bin/setup_env.sh

# Set default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
