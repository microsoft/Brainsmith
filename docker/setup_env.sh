#!/bin/bash
# Brainsmith Environment Setup Script
# Handles environment setup that was previously in entrypoint.sh

export HOME=/tmp/home_dir
export SHELL=/bin/bash
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
# colorful terminal output
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
export PATH=$PATH:$OHMYXILINX

# Set up key FINN environment variables
export FINN_BUILD_DIR=$BSMITH_BUILD_DIR
export FINN_DEPS_DIR="${BSMITH_DIR}/deps"
export FINN_ROOT="${FINN_DEPS_DIR}/finn"

# Define colors for terminal output
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Colorful terminal output functions
yecho () {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

gecho () {
  echo -e "${GREEN}$1${NC}"
}

recho () {
  echo -e "${RED}$1${NC}"
}

if [ -f "$VITIS_PATH/settings64.sh" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  source $VITIS_PATH/settings64.sh
  gecho "Found Vitis at $VITIS_PATH"
else
  yecho "Unable to find $VITIS_PATH/settings64.sh"
  yecho "Functionality dependent on Vitis will not be available."
  yecho "If you need Vitis, ensure VITIS_PATH is set correctly and mounted into the Docker container."
  if [ -f "$VIVADO_PATH/settings64.sh" ];then
    # source Vivado env.vars
    export XILINX_VIVADO=$VIVADO_PATH
    source $VIVADO_PATH/settings64.sh
    gecho "Found Vivado at $VIVADO_PATH"
  else
    yecho "Unable to find $VIVADO_PATH/settings64.sh"
    yecho "Functionality dependent on Vivado will not be available."
    yecho "If you need Vivado, ensure VIVADO_PATH is set correctly and mounted into the Docker container."
  fi
fi

if [ -z "${XILINX_VIVADO}" ]; then
  yecho "pyxsi is unavailable since Vivado was not found"
else
  if [ -f "${BSMITH_DIR}/deps/pyxsi/pyxsi.so" ]; then
    gecho "Found pyxsi at ${BSMITH_DIR}/deps/pyxsi/pyxsi.so"
  else
    if [ -d "${BSMITH_DIR}/deps/pyxsi" ]; then
      gecho "Building pyxsi at ${BSMITH_DIR}/deps/pyxsi"
      OLDPWD=$(pwd)
      cd ${BSMITH_DIR}/deps/pyxsi || {
        recho "Failed to enter pyxsi directory"
        exit 1
      }
      make || {
        recho "Failed to build pyxsi"
        exit 1
      }
      cd $OLDPWD
    else
      recho "pyxsi directory not found at ${BSMITH_DIR}/deps/pyxsi"
      recho "This suggests dependencies were not fetched properly"
      exit 1
    fi
  fi
  
  export PYTHONPATH=$PYTHONPATH:${BSMITH_DIR}/deps/pyxsi:${BSMITH_DIR}/deps/pyxsi/py
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o
fi

if [ -f "$HLS_PATH/settings64.sh" ];then
  # source Vitis HLS env.vars
  source $HLS_PATH/settings64.sh
  gecho "Found Vitis HLS at $HLS_PATH"
else
  yecho "Unable to find $HLS_PATH/settings64.sh"
  yecho "Functionality dependent on Vitis HLS will not be available."
  yecho "Please note that FINN needs at least version 2020.2 for Vitis HLS support. Our recommendation is to use version 2022.2"
  yecho "If you need Vitis HLS, ensure HLS_PATH is set correctly and mounted into the Docker container."
fi

if [ -d "$BSMITH_DIR/.Xilinx" ]; then
  mkdir -p "$HOME/.Xilinx"
  if [ -f "$BSMITH_DIR/.Xilinx/HLS_init.tcl" ]; then
    cp "$BSMITH_DIR/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
    gecho "Found HLS_init.tcl and copied to $HOME/.Xilinx/HLS_init.tcl"
  else
    yecho "Unable to find $BSMITH_DIR/.Xilinx/HLS_init.tcl"
  fi

  if [ -f "$BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl" ]; then
    mkdir -p "$HOME/.Xilinx/Vivado/"
    cp "$BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
    gecho "Found Vivado_init.tcl and copied to $HOME/.Xilinx/Vivado/Vivado_init.tcl"
  else
    yecho "Unable to find $BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl"
  fi
else
  echo "If you need to enable a beta device, ensure .Xilinx/HLS_init.tcl and/or .Xilinx/Vivado/Vivado_init.tcl are set correctly and mounted"
  echo "See https://docs.xilinx.com/r/en-US/ug835-vivado-tcl-commands/Tcl-Initialization-Scripts"
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VITIS_PATH/lnx64/tools/fpo_v7_1"
export PATH=$PATH:$HOME/.local/bin
