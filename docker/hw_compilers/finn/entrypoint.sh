#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

export HOME=/tmp/home_dir
export SHELL=/bin/bash
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
# colorful terminal output
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '
export PATH=$PATH:$OHMYXILINX

source docker/terminal-utils.sh

# qonnx (using workaround for https://github.com/pypa/pip/issues/7953)
# to be fixed in future Ubuntu versions (https://bugs.launchpad.net/ubuntu/+source/setuptools/+bug/1994016)
mv ${BSMITH_DIR}/deps/qonnx/pyproject.toml ${BSMITH_DIR}/deps/qonnx/pyproject.tmp
pip install --user -e ${BSMITH_DIR}/deps/qonnx
mv ${BSMITH_DIR}/deps/qonnx/pyproject.tmp ${BSMITH_DIR}/deps/qonnx/pyproject.toml
# finn-experimental
pip install --user -e ${BSMITH_DIR}/deps/finn-experimental
# brevitas
pip install --user -e ${BSMITH_DIR}/deps/brevitas
# finn
pip install --user -e ${BSMITH_DIR}/deps/finn

if [ -f "${BSMITH_DIR}/setup.py" ];then
  # run pip install for BrainSmith
  pip install --user -e ${BSMITH_DIR}
else
  recho "Unable to find BrainSmith source code in ${BSMITH_DIR}"
  recho "Ensure you have passed -v <path-to-finn-repo>:<path-to-finn-repo> to the docker run command"
  exit -1
fi

if [ -f "$VITIS_PATH/settings64.sh" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
#   export XILINX_XRT=/opt/xilinx/xrt
  source $VITIS_PATH/settings64.sh
  gecho "Found Vitis at $VITIS_PATH"
#   if [ -f "$XILINX_XRT/setup.sh" ];then
#     # source XRT
#     source $XILINX_XRT/setup.sh
#     gecho "Found XRT at $XILINX_XRT"
#   else
#     recho "XRT not found on $XILINX_XRT, did you skip the download or did the installation fail?"
#     exit -1
#   fi
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
  mkdir "$HOME/.Xilinx"
  if [ -f "$BSMITH_DIR/.Xilinx/HLS_init.tcl" ]; then
    cp "$BSMITH_DIR/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
    gecho "Found HLS_init.tcl and copied to $HOME/.Xilinx/HLS_init.tcl"
  else
    yecho "Unable to find $BSMITH_DIR/.Xilinx/HLS_init.tcl"
  fi

  if [ -f "$BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl" ]; then
    mkdir "$HOME/.Xilinx/Vivado/"
    cp "$BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
    gecho "Found Vivado_init.tcl and copied to $HOME/.Xilinx/Vivado/Vivado_init.tcl"
  else
    yecho "Unable to find $BSMITH_DIR/.Xilinx/Vivado/Vivado_init.tcl"
  fi
else
  echo "If you need to enable a beta device, ensure .Xilinx/HLS_init.tcl and/or .Xilinx/Vivado/Vivado_init.tcl are set correctly and mounted"
  echo "See https://docs.xilinx.com/r/en-US/ug835-vivado-tcl-commands/Tcl-Initialization-Scripts"
fi

export PATH=$PATH:$HOME/.local/bin
# execute the provided command(s) as root
exec "$@"