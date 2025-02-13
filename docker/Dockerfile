# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM ubuntu:jammy-20230126
LABEL maintainer="Thomas Keller <thomaskeller@microsoft.com>, Mahdi Ghandi <Mahdi.Ghandi@microsoft.com>"

WORKDIR /workspace

# some Vitis deps require a timezone to be specified, which hangs in Docker
# use workaround from https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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
    libtinfo5
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN locale-gen "en_US.UTF-8"

# install Verilator from source to get the right version
RUN apt-get install -y git perl make autoconf g++-10 flex bison ccache libgoogle-perftools-dev numactl perl-doc libfl2 libfl-dev zlib1g zlib1g-dev
RUN git clone https://github.com/verilator/verilator
RUN cd verilator && \
    git checkout v4.224 && \
    autoconf && \
    ./configure && \
    make -j4 && \
    make install

# versioned Python package requirements for BrainSmith
# these are given in requirements.txt
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt


# install PyTorch
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121

# extra Python package dependencies (for testing and interaction)
RUN pip install pygments==2.14.0
RUN pip install ipykernel==6.21.2
RUN pip install markupsafe==2.0.1
RUN pip install matplotlib==3.7.0  --ignore-installed
RUN pip install pytest-dependency==0.5.1
RUN pip install pytest-xdist[setproctitle]==3.2.0
RUN pip install pytest-parallel==0.1.1
RUN pip install "netron>=5.0.0"
RUN pip install pandas==1.5.3
RUN pip install scikit-learn==1.2.1
RUN pip install tqdm==4.64.1
RUN pip install -e git+https://github.com/fbcotter/dataset_loading.git@0.0.4#egg=dataset_loading
# these versions of pytest and associated plugins allow for stable collection of
# test reports and code coverage reports in HTML
RUN pip install pytest==6.2.5
RUN pip install pytest-metadata==1.7.0
RUN pip install pytest-html==3.0.0
RUN pip install pytest-html-merger==0.0.8
RUN pip install pytest-cov==4.1.0

# extra dependencies from other deps
# installed in Docker image to make entrypoint script go faster
# finn-experimental
RUN pip install deap==1.3.1
RUN pip install mip==1.13.0
RUN pip install networkx==2.8
# brevitas
RUN pip install future-annotations==1.0.0
RUN pip install dependencies==2.0.1
RUN pip install tokenize-rt==4.2.1
# pyverilator
RUN pip install tclwrapper==0.0.1

# assure that we have the right setuptools version
RUN pip install setuptools==68.2.2

# extra environment variables for BrainSmith
ENV VIVADO_IP_CACHE "/tmp/vivado_ip_cache"

COPY docker/brainsmith_entrypoint.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/brainsmith_entrypoint.sh
ENTRYPOINT ["brainsmith_entrypoint.sh"]
CMD ["bash"]
