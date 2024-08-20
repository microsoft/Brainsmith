#!/bin/bash
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is subject to the Xilinx Design License Agreement located
# in the LICENSE.md file in the root directory of this repository.
#
# This file contains confidential and proprietary information of Xilinx, Inc.
# and is protected under U.S. and international copyright and other
# intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in
# any application requiring failsafe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

##
# @brief	Download the correct version of the FINN compiler for BERT
##

# URL for git repo to be cloned
REPO_URL=https://github.com/Xilinx/finn
# commit hash for repo
REPO_COMMIT=d8f6457f5c84628f601b1f30a85a510cb71c99b8
# directory (under the same folder as this script) to clone to
REPO_DIR=finn


# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
# absolute path for the repo local copy
CLONE_TO=$SCRIPTPATH/$REPO_DIR

# clone repo if dir not found
if [ ! -d "$CLONE_TO" ]; then
  git clone $REPO_URL $CLONE_TO
fi
git -C $CLONE_TO pull
# checkout the expected commit
git -C $CLONE_TO checkout $REPO_COMMIT
# verify
CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
if [ $CURRENT_COMMIT == $REPO_COMMIT ]; then
  echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
else
  echo "Could not check out $REPO_DIR. Check your internet connection and try again."
fi
