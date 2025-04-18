#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Load util functions and variables for terminal output
source docker/terminal-utils.sh

# Parse Docker variables
BSMITH_DIR="$(readlink -f -- "${BASH_SOURCE[0]%/*}")"
DOCKER_GID=$(id -g)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_PASSWD="brainsmith"
DOCKER_INST_NAME="brainsmith_dev_0"
# DOCKER_INST_NAME="brainsmith_dev_${DOCKER_UNAME}"
DOCKER_INST_NAME="${DOCKER_INST_NAME,,}"

# Docker variables overwritten by environment variables if available
: ${BSMITH_HW_COMPILER="finn"}
: ${BSMITH_DOCKER_TAG="microsoft/brainsmith:$(git describe --always --tags --dirty)"}
: ${LOCALHOST_URL="localhost"}
: ${NETRON_PORT=8080}
: ${NUM_DEFAULT_WORKERS=4}
: ${NVIDIA_VISIBLE_DEVICES=""}
# Directories
: ${BSMITH_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}
: ${BSMITH_SSH_KEY_DIR="$BSMITH_DIR/ssh_keys"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
# Xilinx specific variables
: ${OHMYXILINX="${BSMITH_DIR}/deps/oh-my-xilinx"}
: ${VIVADO_HLS_LOCAL=$VIVADO_PATH}
: ${VIVADO_IP_CACHE=$BSMITH_BUILD_DIR/vivado_ip_cache}
# Enable/disable Docker build options
: ${DOCKER_BUILDKIT="1"}
: ${BSMITH_DOCKER_PREBUILT="0"}
: ${BSMITH_DOCKER_NO_CACHE="0"}
: ${BSMITH_SKIP_DEP_REPOS="0"}
# Enable/disable Docker run options
: ${BSMITH_DOCKER_RUN_AS_ROOT="0"}
: ${BSMITH_DOCKER_GPU="$(docker info | grep nvidia | wc -m)"}
# Additional Docker options
: ${BSMITH_DOCKER_BUILD_FLAGS=""}
: ${BSMITH_DOCKER_FLAGS=""}

# Determine run command based on CLI arguments
if [ -z "$1" ]; then
  gecho "Running Brainsmith docker container"
  DOCKER_CMD="bash"
  DOCKER_INTERACTIVE="-it"
elif [ "$1" = "build_df_core" ] || [ "$1" = "build_dataflow" ]; then
  # TODO: Add support for one-off builds
  DOCKER_CMD="bash"
  DOCKER_INTERACTIVE="-it"
elif [ "$1" = "pytest" ]; then
  JOB_DIR=$(readlink -f "$2")
  gecho "Running Brainsmith pytest suite"
  DOCKER_CMD="cd tests && pytest ./ -v"
else
  gecho "Running Brainsmith docker container with passed arguments"
  DOCKER_CMD="$@"
  DOCKER_INTERACTIVE=""
fi

# Enable GPU support if available
if [ "$BSMITH_DOCKER_GPU" != 0 ]; then
  gecho "nvidia-docker detected, enabling GPUs"
  if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ]; then
    BSMITH_DOCKER_FLAGS+=" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
  else
    BSMITH_DOCKER_FLAGS+=" --gpus all"
  fi
fi

# Determine paths based on the HW Compiler backend
if [ "$BSMITH_HW_COMPILER" = "finn" ]; then
  DEPS_PATH="$BSMITH_DIR/docker/fetch-repos.sh"
  ENTRYPOINT_PATH="docker/entrypoint.sh"
fi

# Create directories if they do not exist
mkdir -p $BSMITH_BUILD_DIR
# TAFK: Temp commented out
# mkdir -p $BSMITH_SSH_KEY_DIR

# Build Docker image in Brainsmith root directory
if [ "$BSMITH_DOCKER_PREBUILT" = "0" ]; then
  OLD_PWD=$(pwd)
  cd $BSMITH_DIR
  [ "$BSMITH_DOCKER_NO_CACHE" = "1" ] && BSMITH_DOCKER_BUILD_FLAGS+="--no-cache "
  docker build -f docker/Dockerfile --build-arg BACKEND=$BSMITH_HW_COMPILER --build-arg ENTRYPOINT=$ENTRYPOINT_PATH --tag=$BSMITH_DOCKER_TAG $BSMITH_DOCKER_BUILD_FLAGS .
  cd $OLD_PWD
fi

# Compose Docker execution flags and commands
DOCKER_BASE="docker run -t --rm $DOCKER_INTERACTIVE --tty --init --hostname $DOCKER_INST_NAME "
DOCKER_EXEC="-e SHELL=/bin/bash "
DOCKER_EXEC+="-w $BSMITH_DIR "
DOCKER_EXEC+="-v $BSMITH_DIR:$BSMITH_DIR "
DOCKER_EXEC+="-v $BSMITH_BUILD_DIR:$BSMITH_BUILD_DIR "
DOCKER_EXEC+="-e BSMITH_BUILD_DIR="$BSMITH_BUILD_DIR" "
DOCKER_EXEC+="-e BSMITH_DIR="$BSMITH_DIR" "
DOCKER_EXEC+="-e LOCALHOST_URL=$LOCALHOST_URL "
DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
if [ "$BSMITH_DOCKER_RUN_AS_ROOT" = "0" ];then
  DOCKER_EXEC+="-v /etc/group:/etc/group:ro "
  DOCKER_EXEC+="-v /etc/passwd:/etc/passwd:ro "
  DOCKER_EXEC+="-v /etc/shadow:/etc/shadow:ro "
  DOCKER_EXEC+="-v /etc/sudoers.d:/etc/sudoers.d:ro "
  DOCKER_EXEC+="-v $BSMITH_SSH_KEY_DIR:$HOME/.ssh "
  DOCKER_EXEC+="--user $DOCKER_UID:$DOCKER_GID "
else
  DOCKER_EXEC+="-v $BSMITH_SSH_KEY_DIR:/root/.ssh "
fi

# Pull dependencies specific to the selected HW Compiler
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
  source $DEPS_PATH
  # Add flags to Docker run command
  DOCKER_EXEC+="-e VIVADO_IP_CACHE=$BSMITH_BUILD_DIR/vivado_ip_cache "
  DOCKER_EXEC+="-e OHMYXILINX=${BSMITH_DIR}/deps/oh-my-xilinx "
  # Workaround for FlexLM issue, see:
  # https://community.flexera.com/t5/InstallAnywhere-Forum/Issues-when-running-Xilinx-tools-or-Other-vendor-tools-in-docker/m-p/245820#M10647
  DOCKER_EXEC+="-e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 "
  # Workaround for running multiple Vivado instances simultaneously, see:
  # https://adaptivesupport.amd.com/s/article/63253?language=en_US
  DOCKER_EXEC+="-e XILINX_LOCAL_USER_DATA=no "
  # Xilinx specific commands
  if [ ! -z "$BSMITH_XILINX_PATH" ];then
      VIVADO_PATH="$BSMITH_XILINX_PATH/Vivado/$BSMITH_XILINX_VERSION"
      VITIS_PATH="$BSMITH_XILINX_PATH/Vitis/$BSMITH_XILINX_VERSION"
      HLS_PATH="$BSMITH_XILINX_PATH/Vitis_HLS/$BSMITH_XILINX_VERSION"
      DOCKER_EXEC+="-v $BSMITH_XILINX_PATH:$BSMITH_XILINX_PATH "
      if [ -d "$VIVADO_PATH" ];then
        DOCKER_EXEC+="-e "XILINX_VIVADO=$VIVADO_PATH" "
        DOCKER_EXEC+="-e VIVADO_PATH=$VIVADO_PATH "
      fi
      if [ -d "$HLS_PATH" ];then
        DOCKER_EXEC+="-e HLS_PATH=$HLS_PATH "
      fi
      if [ -d "$VITIS_PATH" ];then
        DOCKER_EXEC+="-e VITIS_PATH=$VITIS_PATH "
      fi
      if [ -d "$PLATFORM_REPO_PATHS" ];then
        DOCKER_EXEC+="-v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS "
        DOCKER_EXEC+="-e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS "
      fi
  fi
fi

# Compose and execute Docker command
DOCKER_EXEC+=" $BSMITH_DOCKER_FLAGS"
CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $BSMITH_DOCKER_TAG $DOCKER_CMD"
$CMD_TO_RUN
