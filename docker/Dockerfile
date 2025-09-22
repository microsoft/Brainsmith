# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM ubuntu:22.04
LABEL maintainer="Thomas Keller <thomaskeller@microsoft.com>, Mahdi Ghandi <Mahdi.Ghandi@microsoft.com>"

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
    curl
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN locale-gen "en_US.UTF-8"

# Install Poetry and toml
ENV POETRY_VERSION=1.8.5
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_VIRTUALENVS_PATH=/tmp/poetry_venvs
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry --version && \
    pip3 install toml

# Skip pre-installing dependencies - they'll be installed at runtime
# This avoids permission issues when running as non-root user
WORKDIR /

# Xilinx toolflow specific imports
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy all entrypoint scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/entrypoint-poetry.sh /usr/local/bin/entrypoint-poetry.sh
COPY docker/setup-env.sh /usr/local/bin/setup-env.sh
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/entrypoint-exec.sh /usr/local/bin/entrypoint-exec.sh
COPY docker/setup-env.sh /usr/local/bin/setup-env.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/entrypoint-exec.sh /usr/local/bin/setup-env.sh

# Set default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]