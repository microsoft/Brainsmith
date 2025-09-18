#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Git dependency fetcher for Brainsmith

# ==============================================================================
# REPOSITORY CONFIGURATION - Edit URLs and commits here
# ==============================================================================

# Core dependencies (always fetched)
QONNX_URL="https://github.com/fastmachinelearning/qonnx.git"
QONNX_COMMIT="9153395712b5617d38b058900c873c6fc522b343"

FINN_URL="https://github.com/Xilinx/finn.git"
FINN_COMMIT="bd9baeb7ddad0f613689f3be81df28067f8c1d9b"

FINN_EXPERIMENTAL_URL="https://github.com/Xilinx/finn-experimental.git"
FINN_EXPERIMENTAL_COMMIT="0724be21111a21f0d81a072fccc1c446e053f851"

BREVITAS_URL="https://github.com/Xilinx/brevitas.git"
BREVITAS_COMMIT="95edaa0bdc8e639e39b1164466278c59df4877be"

DATASET_LOADING_URL="https://github.com/fbcotter/dataset_loading.git"
DATASET_LOADING_COMMIT="0.0.4"

CNPY_URL="https://github.com/maltanar/cnpy.git"
CNPY_COMMIT="8c82362372ce600bbd1cf11d64661ab69d38d7de"

FINN_HLSLIB_URL="https://github.com/Xilinx/finn-hlslib.git"
FINN_HLSLIB_COMMIT="5c5ad631e3602a8dd5bd3399a016477a407d6ee7"

OH_MY_XILINX_URL="https://github.com/maltanar/oh-my-xilinx.git"
OH_MY_XILINX_COMMIT="0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a"

# Board dependencies (fetched if BSMITH_FETCH_BOARDS=true)
AVNET_BDF_URL="https://github.com/Avnet/bdf.git"
AVNET_BDF_COMMIT="2d49cfc25766f07792c0b314489f21fe916b639b"

XIL_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
XIL_BDF_COMMIT="8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"

RFSOC4X2_BDF_URL="https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git"
RFSOC4X2_BDF_COMMIT="13fb6f6c02c7dfd7e4b336b18b959ad5115db696"

KV260_SOM_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
KV260_SOM_BDF_COMMIT="98e0d3efc901f0b974006bc4370c2a7ad8856c79"

# Experimental dependencies (fetched if BSMITH_FETCH_EXPERIMENTAL=true)
ONNXSCRIPT_URL="https://github.com/jsmonson/onnxscript.git"
ONNXSCRIPT_COMMIT="62c7110aba46554432ce8e82ba2d8a086bd6227c"

# ==============================================================================

set -e

# Color functions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

gecho() { echo -e "${GREEN}$1${NC}"; }
recho() { echo -e "${RED}$1${NC}"; }
yecho() { echo -e "${YELLOW}$1${NC}"; }

# Find the Brainsmith root directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export BSMITH_DIR="${BSMITH_DIR:-$(readlink -f "$SCRIPT_DIR/..")}"

# Load environment configuration
if [ -f "$BSMITH_DIR/docker/env-config.sh" ]; then
    source "$BSMITH_DIR/docker/env-config.sh"
fi

# Dependencies directory
DEPS_DIR="${DEPS_DIR:-$BSMITH_DIR/deps}"

# Function to fetch a git repository
fetch_repo() {
    local name=$1
    local url=$2
    local commit=$3
    local clone_dir="${DEPS_DIR}/${name}"
    
    echo "Fetching $name from $url..."
    
    # Clone if directory doesn't exist
    if [ ! -d "$clone_dir" ]; then
        echo "Cloning $name..."
        if git clone "$url" "$clone_dir"; then
            gecho "Successfully cloned $name"
        else
            recho "Failed to clone $name"
            return 1
        fi
    fi
    
    # Checkout the specified commit
    cd "$clone_dir"
    current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    if [ "$current_commit" != "$commit" ]; then
        echo "Updating $name to commit $commit..."
        git fetch --quiet || echo "Fetch failed, continuing..."
        
        if git checkout "$commit"; then
            gecho "Successfully checked out $name at $commit"
        else
            recho "Failed to checkout $name at $commit"
            return 1
        fi
    else
        gecho "$name already at correct commit $commit"
    fi
    
    cd "$BSMITH_DIR"
}

# Function to download board files
fetch_board_files() {
    echo "Downloading and extracting additional board files..."
    mkdir -p "$DEPS_DIR/board_files"
    cd "$DEPS_DIR/board_files"
    
    # Download Pynq board files
    if [ ! -f pynq-z1.zip ]; then
        wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
        unzip -q pynq-z1.zip
    fi
    
    if [ ! -f pynq-z2.zip ]; then
        wget -q https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
        unzip -q pynq-z2.zip
    fi
    
    # Copy board files from other repos
    if [ -d "$DEPS_DIR/avnet-bdf" ]; then
        cp -r "$DEPS_DIR/avnet-bdf"/* "$DEPS_DIR/board_files/" 2>/dev/null || true
    fi
    
    if [ -d "$DEPS_DIR/xil-bdf/boards/Xilinx/rfsoc2x2" ]; then
        cp -r "$DEPS_DIR/xil-bdf/boards/Xilinx/rfsoc2x2" "$DEPS_DIR/board_files/"
    fi
    
    if [ -d "$DEPS_DIR/rfsoc4x2-bdf/board_files/rfsoc4x2" ]; then
        cp -r "$DEPS_DIR/rfsoc4x2-bdf/board_files/rfsoc4x2" "$DEPS_DIR/board_files/"
    fi
    
    if [ -d "$DEPS_DIR/kv260-som-bdf/boards/Xilinx/kv260_som" ]; then
        cp -r "$DEPS_DIR/kv260-som-bdf/boards/Xilinx/kv260_som" "$DEPS_DIR/board_files/"
    fi
    
    cd "$BSMITH_DIR"
    gecho "Board files ready"
}

# Main execution
main() {
    # Create deps directory
    mkdir -p "$DEPS_DIR"
    
    gecho "Fetching core dependencies..."
    
    # Core dependencies - always fetched
    fetch_repo "qonnx" "$QONNX_URL" "$QONNX_COMMIT"
    fetch_repo "finn" "$FINN_URL" "$FINN_COMMIT"
    fetch_repo "finn-experimental" "$FINN_EXPERIMENTAL_URL" "$FINN_EXPERIMENTAL_COMMIT"
    fetch_repo "brevitas" "$BREVITAS_URL" "$BREVITAS_COMMIT"
    fetch_repo "dataset_loading" "$DATASET_LOADING_URL" "$DATASET_LOADING_COMMIT"
    fetch_repo "cnpy" "$CNPY_URL" "$CNPY_COMMIT"
    fetch_repo "finn-hlslib" "$FINN_HLSLIB_URL" "$FINN_HLSLIB_COMMIT"
    fetch_repo "oh-my-xilinx" "$OH_MY_XILINX_URL" "$OH_MY_XILINX_COMMIT"
    
    # Fetch board dependencies if requested
    if [ "${BSMITH_FETCH_BOARDS:-true}" = "true" ] && [ "${FINN_SKIP_BOARD_FILES}" != "1" ]; then
        gecho "Fetching board dependencies..."
        fetch_repo "avnet-bdf" "$AVNET_BDF_URL" "$AVNET_BDF_COMMIT"
        fetch_repo "xil-bdf" "$XIL_BDF_URL" "$XIL_BDF_COMMIT"
        fetch_repo "rfsoc4x2-bdf" "$RFSOC4X2_BDF_URL" "$RFSOC4X2_BDF_COMMIT"
        fetch_repo "kv260-som-bdf" "$KV260_SOM_BDF_URL" "$KV260_SOM_BDF_COMMIT"
        fetch_board_files
    else
        yecho "Skipping board dependencies (BSMITH_FETCH_BOARDS=false or FINN_SKIP_BOARD_FILES=1)"
    fi
    
    # Fetch experimental dependencies if requested
    if [ "${BSMITH_FETCH_EXPERIMENTAL:-false}" = "true" ]; then
        gecho "Fetching experimental dependencies..."
        fetch_repo "onnxscript" "$ONNXSCRIPT_URL" "$ONNXSCRIPT_COMMIT"
    else
        yecho "Skipping experimental dependencies (set BSMITH_FETCH_EXPERIMENTAL=true to enable)"
    fi
    
    # Show summary
    echo
    gecho "Dependency fetching complete!"
    gecho "Build directory: ${BSMITH_BUILD_DIR}"
    
    # Show Xilinx status
    if [ -n "$XILINX_VIVADO" ]; then
        gecho "Xilinx Vivado: $XILINX_VIVADO"
    else
        yecho "Xilinx tools not configured (set XILINX_VIVADO for FPGA support)"
    fi
}

# Run main function
main "$@"