#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# green echo
gecho () {
  echo -e "${GREEN}$1${NC}"
}

# red echo
recho () {
  echo -e "${RED}$1${NC}"
}

if [ -z "$BSMITH_XILINX_PATH" ];then
  recho "Please set the BSMITH_XILINX_PATH environment variable to the path to your Xilinx tools installation directory (e.g. /opt/Xilinx)."
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$BSMITH_XILINX_VERSION" ];then
  recho "Please set the BSMITH_XILINX_VERSION to the version of the Xilinx tools to use (e.g. 2022.2)"
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$PLATFORM_REPO_PATHS" ];then
  recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
  recho "This is required to be able to use Alveo PCIe cards."
fi

DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_PASSWD="brainsmith"
DOCKER_INST_NAME="brainsmith_dev_${DOCKER_UNAME}"
# ensure Docker inst. name is all lowercase
DOCKER_INST_NAME=$(echo "$DOCKER_INST_NAME" | tr '[:upper:]' '[:lower:]')
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# the settings below will be taken from environment variables if available,
# otherwise the defaults below will be used
: ${NETRON_PORT=8080}
: ${LOCALHOST_URL="localhost"}
: ${NUM_DEFAULT_WORKERS=4}
: ${BSMITH_SSH_KEY_DIR="$SCRIPTPATH/ssh_keys"}
: ${ALVEO_USERNAME="alveo_user"}
: ${ALVEO_PASSWORD=""}
: ${ALVEO_BOARD="U250"}
: ${ALVEO_TARGET_DIR="/tmp"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
: ${BSMITH_HOST_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}
: ${BSMITH_DOCKER_TAG="microsoft/brainsmith:$(git describe --always --tags --dirty)"}
: ${BSMITH_DOCKER_PREBUILT="0"}
: ${BSMITH_DOCKER_RUN_AS_ROOT="0"}
: ${BSMITH_DOCKER_GPU="$(docker info | grep nvidia | wc -m)"}
: ${BSMITH_DOCKER_EXTRA=""}
: ${BSMITH_DOCKER_BUILD_EXTRA=""}
: ${BSMITH_SKIP_DEP_REPOS="0"}
: ${BSMITH_SKIP_BOARD_FILES="0"}
: ${OHMYXILINX="${SCRIPTPATH}/deps/oh-my-xilinx"}
: ${NVIDIA_VISIBLE_DEVICES=""}
: ${DOCKER_BUILDKIT="1"}
: ${BSMITH_SINGULARITY=""}
: ${BSMITH_DOCKER_NO_CACHE="0"}

DOCKER_INTERACTIVE=""

# Catch BSMITH_DOCKER_EXTRA options being passed in without a trailing space
BSMITH_DOCKER_EXTRA+=" "

if [ "$1" = "example" ]; then
  gecho "This is an example of how to structure a docker flag"
  DOCKER_CMD=""
  DOCKER_INTERACTIVE=""
elif [ -z "$1" ]; then
   gecho "Running container only"
   DOCKER_CMD="bash"
   DOCKER_INTERACTIVE="-it"
else
  gecho "Running container with passed arguments"
  DOCKER_CMD="$@"
fi

if [ "$BSMITH_DOCKER_GPU" != 0 ] && [ -z "$BSMITH_SINGULARITY" ];then
  gecho "nvidia-docker detected, enabling GPUs"
  if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ];then
    BSMITH_DOCKER_EXTRA+="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES "
  else
    BSMITH_DOCKER_EXTRA+="--gpus all "
  fi
fi

VIVADO_HLS_LOCAL=$VIVADO_PATH
VIVADO_IP_CACHE=$BSMITH_HOST_BUILD_DIR/vivado_ip_cache

# ensure build dir exists locally
mkdir -p $BSMITH_HOST_BUILD_DIR
mkdir -p $BSMITH_SSH_KEY_DIR

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $BSMITH_DOCKER_TAG"
gecho "Mounting $BSMITH_HOST_BUILD_DIR into $BSMITH_HOST_BUILD_DIR"
gecho "Mounting $BSMITH_XILINX_PATH into $BSMITH_XILINX_PATH"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"
gecho "Vivado IP cache dir is at $VIVADO_IP_CACHE"

# Ensure git-based deps are checked out at correct commit
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
  ./fetch-repos.sh
fi

if [ "$BSMITH_DOCKER_NO_CACHE" = "1" ]; then
  BSMITH_DOCKER_BUILD_EXTRA+="--no-cache "
fi

# Build the BrainSmith Docker image
if [ "$BSMITH_DOCKER_PREBUILT" = "0" ] && [ -z "$BSMITH_SINGULARITY" ]; then
  # Need to ensure this is done within the brainsmith/ root folder:
  OLD_PWD=$(pwd)
  cd $SCRIPTPATH
  docker build -f docker/Dockerfile --tag=$BSMITH_DOCKER_TAG $BSMITH_DOCKER_BUILD_EXTRA .
  cd $OLD_PWD
fi

# Launch container with current directory mounted
# important to pass the --init flag here for correct Vivado operation, see:
# https://stackoverflow.com/questions/55733058/vivado-synthesis-hangs-in-docker-container-spawned-by-jenkins
DOCKER_BASE="docker run -t --rm $DOCKER_INTERACTIVE --tty --init --hostname $DOCKER_INST_NAME "
DOCKER_EXEC="-e SHELL=/bin/bash "
DOCKER_EXEC+="-w $SCRIPTPATH "
DOCKER_EXEC+="-v $SCRIPTPATH:$SCRIPTPATH "
DOCKER_EXEC+="-v $BSMITH_HOST_BUILD_DIR:$BSMITH_HOST_BUILD_DIR "
DOCKER_EXEC+="-e BSMITH_BUILD_DIR=$BSMITH_HOST_BUILD_DIR "
DOCKER_EXEC+="-e BSMITH_BUILD_DIR="$SCRIPTPATH" "
DOCKER_EXEC+="-e LOCALHOST_URL=$LOCALHOST_URL "
DOCKER_EXEC+="-e VIVADO_IP_CACHE=$VIVADO_IP_CACHE "
DOCKER_EXEC+="-e OHMYXILINX=$OHMYXILINX "
DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
# Workaround for FlexLM issue, see:
# https://community.flexera.com/t5/InstallAnywhere-Forum/Issues-when-running-Xilinx-tools-or-Other-vendor-tools-in-docker/m-p/245820#M10647
DOCKER_EXEC+="-e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 "
# Workaround for running multiple Vivado instances simultaneously, see:
# https://adaptivesupport.amd.com/s/article/63253?language=en_US
DOCKER_EXEC+="-e XILINX_LOCAL_USER_DATA=no "
if [ "$BSMITH_DOCKER_RUN_AS_ROOT" = "0" ] && [ -z "$BSMITH_SINGULARITY" ];then
  DOCKER_EXEC+="-v /etc/group:/etc/group:ro "
  DOCKER_EXEC+="-v /etc/passwd:/etc/passwd:ro "
  DOCKER_EXEC+="-v /etc/shadow:/etc/shadow:ro "
  DOCKER_EXEC+="-v /etc/sudoers.d:/etc/sudoers.d:ro "
  DOCKER_EXEC+="-v $BSMITH_SSH_KEY_DIR:$HOME/.ssh "
  DOCKER_EXEC+="--user $DOCKER_UID:$DOCKER_GID "
else
  DOCKER_EXEC+="-v $BSMITH_SSH_KEY_DIR:/root/.ssh "
fi
if [ ! -z "$IMAGENET_VAL_PATH" ];then
  DOCKER_EXEC+="-v $IMAGENET_VAL_PATH:$IMAGENET_VAL_PATH "
  DOCKER_EXEC+="-e IMAGENET_VAL_PATH=$IMAGENET_VAL_PATH "
fi
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
    DOCKER_EXEC+="-e ALVEO_IP=$ALVEO_IP "
    DOCKER_EXEC+="-e ALVEO_USERNAME=$ALVEO_USERNAME "
    DOCKER_EXEC+="-e ALVEO_PASSWORD=$ALVEO_PASSWORD "
    DOCKER_EXEC+="-e ALVEO_BOARD=$ALVEO_BOARD "
    DOCKER_EXEC+="-e ALVEO_TARGET_DIR=$ALVEO_TARGET_DIR "
  fi
fi

DOCKER_EXEC+="$BSMITH_DOCKER_EXTRA "

if [ -z "$BSMITH_SINGULARITY" ];then
  CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $BSMITH_DOCKER_TAG $DOCKER_CMD"
else
  SINGULARITY_BASE="singularity exec"
  # Replace command options for Singularity
  SINGULARITY_EXEC="${DOCKER_EXEC//"-e "/"--env "}"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//"-v "/"-B "}"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//"-w "/"--pwd "}"
  CMD_TO_RUN="$SINGULARITY_BASE $SINGULARITY_EXEC $BSMITH_SINGULARITY /usr/local/bin/brainsmith_entrypoint.sh $DOCKER_CMD"
  gecho "BSMITH_SINGULARITY is set, launching Singularity container instead of Docker"
fi

$CMD_TO_RUN
