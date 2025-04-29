#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
