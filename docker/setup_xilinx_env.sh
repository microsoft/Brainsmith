#!/bin/bash
# setup_xilinx_env.sh - Set up environment for Xilinx tools
# Source this file when doing FPGA synthesis work

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Xilinx tools are available
if [ -z "$XILINX_VIVADO" ]; then
    echo -e "${RED}Error: XILINX_VIVADO not set${NC}"
    echo "Please set XILINX_VIVADO to your Vivado installation directory"
    echo "Example: export XILINX_VIVADO=/opt/Xilinx/Vivado/2024.2"
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$XILINX_VIVADO" ]; then
    echo -e "${RED}Error: XILINX_VIVADO directory not found: $XILINX_VIVADO${NC}"
    return 1 2>/dev/null || exit 1
fi

# Source Vivado settings
if [ -f "$XILINX_VIVADO/settings64.sh" ]; then
    echo "Sourcing Vivado settings..."
    source "$XILINX_VIVADO/settings64.sh"
    echo -e "${GREEN}✓ Vivado environment configured${NC}"
else
    echo -e "${RED}Error: settings64.sh not found in $XILINX_VIVADO${NC}"
    return 1 2>/dev/null || exit 1
fi

# Set up FINN-specific environment variables
export FINN_XILINX_PATH="$XILINX_VIVADO"
export FINN_XILINX_VERSION=$(basename "$XILINX_VIVADO")

# Set up library paths for Xilinx runtime
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$XILINX_VIVADO/lib/lnx64.o"

# Set up Python path for finnxsi if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/deps/finnxsi" ]; then
    export PYTHONPATH="$SCRIPT_DIR/deps/finnxsi:$PYTHONPATH"
    echo -e "${GREEN}✓ finnxsi added to PYTHONPATH${NC}"
fi

# Check for optional Vitis/HLS
if [ -n "$XILINX_VITIS" ] && [ -f "$XILINX_VITIS/settings64.sh" ]; then
    source "$XILINX_VITIS/settings64.sh"
    echo -e "${GREEN}✓ Vitis environment configured${NC}"
fi

if [ -n "$XILINX_HLS" ] && [ -f "$XILINX_HLS/settings64.sh" ]; then
    source "$XILINX_HLS/settings64.sh"
    echo -e "${GREEN}✓ Vitis HLS environment configured${NC}"
fi

echo
echo "Xilinx environment ready for $(basename "$XILINX_VIVADO")"
echo "You can now run FPGA synthesis workflows"