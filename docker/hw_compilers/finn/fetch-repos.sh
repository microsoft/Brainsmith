#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Dependency Git URLs, hashes/branches, and directory names
FINN_URL="https://github.com/Xilinx/finn.git"
QONNX_URL="https://github.com/fastmachinelearning/qonnx.git"
FINN_EXP_URL="https://github.com/Xilinx/finn-experimental.git"
BREVITAS_URL="https://github.com/Xilinx/brevitas.git"
PYVERILATOR_URL="https://github.com/maltanar/pyverilator.git"
CNPY_URL="https://github.com/maltanar/cnpy.git"
HLSLIB_URL="https://github.com/Xilinx/finn-hlslib.git"
OMX_URL="https://github.com/maltanar/oh-my-xilinx.git"
AVNET_BDF_URL="https://github.com/Avnet/bdf.git"
XIL_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
RFSOC4x2_BDF_URL="https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git"
KV260_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
PYXSI_URL="https://github.com/maltanar/pyxsi.git"

FINN_COMMIT="custom/transformer"
QONNX_COMMIT="2281a777d84aa5cbd7469085c2e534fb4a03ccf9"
FINN_EXP_COMMIT="0724be21111a21f0d81a072fccc1c446e053f851"
BREVITAS_COMMIT="d4834bd2a0fad3c1fbc0ff7e1346d5dcb3797ea4"
PYVERILATOR_COMMIT="ce0a08c20cb8c1d1e84181d6f392390f846adbd1"
CNPY_COMMIT="8c82362372ce600bbd1cf11d64661ab69d38d7de"
HLSLIB_COMMIT="7783acaac835e702da25aa6b7103254b3cbcdf83"
OMX_COMMIT="0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a"
AVNET_BDF_COMMIT="2d49cfc25766f07792c0b314489f21fe916b639b"
XIL_BDF_COMMIT="8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"
RFSOC4x2_BDF_COMMIT="13fb6f6c02c7dfd7e4b336b18b959ad5115db696"
KV260_BDF_COMMIT="98e0d3efc901f0b974006bc4370c2a7ad8856c79"
EXP_BOARD_FILES_MD5="226ca927a16ea4ce579f1332675e9e9a"
PYXSI_COMMIT="4f4ec10a3631c4c44b5bc0ede698d41c924d2b86"

FINN_DIR="finn"
QONNX_DIR="qonnx"
FINN_EXP_DIR="finn-experimental"
BREVITAS_DIR="brevitas"
PYVERILATOR_DIR="pyverilator"
CNPY_DIR="cnpy"
HLSLIB_DIR="finn-hlslib"
OMX_DIR="oh-my-xilinx"
AVNET_BDF_DIR="avnet-bdf"
XIL_BDF_DIR="xil-bdf"
RFSOC4x2_BDF_DIR="rfsoc4x2-bdf"
KV260_SOM_BDF_DIR="kv260-som-bdf"
PYXSI_DIR="pyxsi"

# Validate environment variables for licensed Xilinx tools
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

# Set environment variables if not already set
: ${OHMYXILINX="${BSMITH_DIR}/deps/oh-my-xilinx"}
: ${VIVADO_HLS_LOCAL=$VIVADO_PATH}
: ${VIVADO_IP_CACHE=$BSMITH_BUILD_DIR/vivado_ip_cache}

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
fi

# Define functions
fetch_repo() {
    # URL for git repo to be cloned
    REPO_URL=$1
    # commit hash for repo
    REPO_COMMIT=$2
    # directory to clone to under deps
    REPO_DIR=$3
    # absolute path for the repo local copy
    CLONE_TO=$BSMITH_DIR/deps/$REPO_DIR

    # clone repo if dir not found
    if [ ! -d "$CLONE_TO" ]; then
        git clone $REPO_URL $CLONE_TO
    fi
    # verify and try to pull repo if not at correct commit
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
    if [ $CURRENT_COMMIT != $REPO_COMMIT ]; then
        git -C $CLONE_TO pull
        # checkout the expected commit
        git -C $CLONE_TO checkout $REPO_COMMIT
    fi
    # verify one last time
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
    if [ $CURRENT_COMMIT == $REPO_COMMIT ]; then
        echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
    else
        echo "Could not check out $REPO_DIR. Check your internet connection and try again."
    fi
}

fetch_board_files() {
    echo "Downloading and extracting board files..."
    mkdir -p "$BSMITH_DIR/deps/board_files"
    OLD_PWD=$(pwd)
    cd "$BSMITH_DIR/deps/board_files"
    wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    wget -q https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    cp -r $BSMITH_DIR/deps/$AVNET_BDF_DIR/* $BSMITH_DIR/deps/board_files/
    cp -r $BSMITH_DIR/deps/$XIL_BDF_DIR/boards/Xilinx/rfsoc2x2 $BSMITH_DIR/deps/board_files/;
    cp -r $BSMITH_DIR/deps/$RFSOC4x2_BDF_DIR/board_files/rfsoc4x2 $BSMITH_DIR/deps/board_files/;
    cp -r $BSMITH_DIR/deps/$KV260_SOM_BDF_DIR/boards/Xilinx/kv260_som $BSMITH_DIR/deps/board_files/;
    cd $OLD_PWD
}

fetch_repo $FINN_URL $FINN_COMMIT $FINN_DIR
fetch_repo $QONNX_URL $QONNX_COMMIT $QONNX_DIR
fetch_repo $FINN_EXP_URL $FINN_EXP_COMMIT $FINN_EXP_DIR
fetch_repo $BREVITAS_URL $BREVITAS_COMMIT $BREVITAS_DIR
fetch_repo $PYVERILATOR_URL $PYVERILATOR_COMMIT $PYVERILATOR_DIR
fetch_repo $CNPY_URL $CNPY_COMMIT $CNPY_DIR
fetch_repo $HLSLIB_URL $HLSLIB_COMMIT $HLSLIB_DIR
fetch_repo $OMX_URL $OMX_COMMIT $OMX_DIR
fetch_repo $AVNET_BDF_URL $AVNET_BDF_COMMIT $AVNET_BDF_DIR
fetch_repo $XIL_BDF_URL $XIL_BDF_COMMIT $XIL_BDF_DIR
fetch_repo $RFSOC4x2_BDF_URL $RFSOC4x2_BDF_COMMIT $RFSOC4x2_BDF_DIR
fetch_repo $KV260_BDF_URL $KV260_BDF_COMMIT $KV260_SOM_BDF_DIR
fetch_repo $PYXSI_URL $PYXSI_COMMIT $PYXSI_DIR

# Can skip downloading of board files entirely if desired
if [ "$FINN_SKIP_BOARD_FILES" = "1" ]; then
    echo "Skipping download and verification of board files"
else
    # download extra board files and extract if needed
    if [ ! -d "$BSMITH_DIR/deps/board_files" ]; then
        fetch_board_files
    else
        cd $BSMITH_DIR
        BOARD_FILES_MD5=$(find deps/board_files/ -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -d' ' -f 1)
        if [ "$BOARD_FILES_MD5" = "$EXP_BOARD_FILES_MD5" ]; then
            echo "Verified board files folder content md5: $BOARD_FILES_MD5"
        else
            echo "Board files folder md5: expected $BOARD_FILES_MD5 found $EXP_BOARD_FILES_MD5"
            echo "Board files folder content mismatch, removing and re-downloading"
            rm -rf deps/board_files/
            fetch_board_files
        fi
    fi
fi

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $BSMITH_DOCKER_TAG"
gecho "Mounting $BSMITH_BUILD_DIR into $BSMITH_BUILD_DIR"
gecho "Mounting $BSMITH_XILINX_PATH into $BSMITH_XILINX_PATH"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"
gecho "Vivado IP cache dir is at $VIVADO_IP_CACHE"