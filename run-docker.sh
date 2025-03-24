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
    gecho "Running BrainSmith docker container"
    DOCKER_CMD="bash"
    DOCKER_INTERACTIVE="-it"
elif [ "$1" = "build_df_core" ] || [ "$1" = "build_dataflow" ]; then
    JOB_DIR=$(readlink -f "$2")
    gecho "Running $1 for folder $JOB_DIR"
    BSMITH_DOCKER_FLAGS+="-v $JOB_DIR:$JOB_DIR "
    DOCKER_CMD="$1 $JOB_DIR"
    DOCKER_INTERACTIVE="-it"
else
    gecho "Running BrainSmith docker container with passed arguments"
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
# TMP commented out
# mkdir -p $BSMITH_SSH_KEY_DIR

# Build Docker image in BrainSmith root directory
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
fi

# Compose and execute Docker command
DOCKER_EXEC+=" $BSMITH_DOCKER_FLAGS"
CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $BSMITH_DOCKER_TAG $DOCKER_CMD"
$CMD_TO_RUN
