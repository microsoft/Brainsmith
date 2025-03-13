#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export HOME=/tmp/home_dir
export SHELL=/bin/bash
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LANGUAGE="en_US:en"
# colorful terminal output
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '

RED='\033[0;31m'
NC='\033[0m' # No Color

recho () {
  echo -e "${RED}ERROR: $1${NC}"
}

# qonnx (using workaround for https://github.com/pypa/pip/issues/7953)
# to be fixed in future Ubuntu versions (https://bugs.launchpad.net/ubuntu/+source/setuptools/+bug/1994016)
mv ${BSMITH_ROOT}/deps/qonnx/pyproject.toml ${BSMITH_ROOT}/deps/qonnx/pyproject.tmp
pip install --user -e ${BSMITH_ROOT}/deps/qonnx
mv ${BSMITH_ROOT}/deps/qonnx/pyproject.tmp ${BSMITH_ROOT}/deps/qonnx/pyproject.toml
# finn-experimental
pip install --user -e ${BSMITH_ROOT}/deps/finn-experimental
# brevitas
pip install --user -e ${BSMITH_ROOT}/deps/brevitas
# finn
pip install --user -e ${BSMITH_ROOT}/deps/finn

if [ -f "${BSMITH_ROOT}/setup.py" ];then
  # run pip install for BrainSmith
  pip install --user -e ${BSMITH_ROOT}
else
  recho "Unable to find BrainSmith source code in ${BSMITH_ROOT}"
  recho "Ensure you have passed -v <path-to-brainsmith-repo>:<path-to-brainsmith-repo> to the docker run command"
  exit -1
fi


export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VITIS_PATH/lnx64/tools/fpo_v7_1"

export PATH=$PATH:$HOME/.local/bin

# execute the provided command(s) as root
exec "$@"
