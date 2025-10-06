#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# validate_env.sh - Check if system meets Brainsmith requirements
# Validates local system against Docker environment dependencies

# Color codes
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Checking Brainsmith local installation requirements..."
echo "(Validating against Docker environment: Ubuntu 22.04)"

ERRORS=0
WARNINGS=0
FIRST_MESSAGE=true

# Helper to print messages with proper spacing
print_message() {
    if [ "$FIRST_MESSAGE" = "true" ]; then
        echo  # Add single blank line before first message
        FIRST_MESSAGE=false
    fi
    echo -e "$@"
}

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]] || [[ "${VERSION_ID%%.*}" -lt 22 ]]; then
        print_message "${RED}[ERROR]${NC} Requires Ubuntu 22.04 or newer (found: $ID $VERSION_ID)"
        ((ERRORS++))
    fi
fi

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    print_message "${RED}[ERROR]${NC} Python3 not found"
    ((ERRORS++))
else
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
    
    # Check against pyproject.toml requirement: python = ">=3.10"
    if [[ $PY_MAJOR -ne 3 ]] || [[ $PY_MINOR -lt 10 ]]; then
        print_message "${RED}[ERROR]${NC} Python $PY_VERSION found (requires >= 3.10)"
        ((ERRORS++))
    elif [[ $PY_MINOR -gt 12 ]]; then
        print_message "${YELLOW}[WARN]${NC} Python $PY_VERSION found (tested with 3.10-3.12)"
        ((WARNINGS++))
    fi
fi

# Check truly essential packages
MISSING_ESSENTIAL=""
for pkg in build-essential git python3-pip python3-venv; do
    if ! dpkg -l 2>/dev/null | grep -q "^ii  $pkg "; then
        MISSING_ESSENTIAL="$MISSING_ESSENTIAL $pkg"
        ((ERRORS++))
    fi
done

if [ -n "$MISSING_ESSENTIAL" ]; then
    print_message "${RED}[ERROR]${NC} Missing essential packages:$MISSING_ESSENTIAL"
fi

# Check if curl is available (only needed for Poetry installation)
if ! command -v curl >/dev/null 2>&1 && ! command -v poetry >/dev/null 2>&1; then
    print_message "${YELLOW}[WARN]${NC} curl not found (needed to install Poetry)"
    ((WARNINGS++))
fi

# Check for unzip (might be needed for board files)
if ! command -v unzip >/dev/null 2>&1; then
    print_message "${YELLOW}[INFO]${NC} unzip not found (may be needed for board file downloads)"
fi

# Check C++ compiler (only for cppsim, not rtlsim)
if command -v g++ >/dev/null 2>&1; then
    # Test if g++ actually works
    if echo 'int main(){}' | g++ -x c++ -o /dev/null - 2>/dev/null; then
        : # g++ works, no message needed
    else
        print_message "${YELLOW}[INFO]${NC} g++ found but not working properly"
        echo "       Only affects C++ simulation (cppsim)"
    fi
else
    print_message "${YELLOW}[INFO]${NC} C++ compiler (g++) not found"
    echo "       Only needed for C++ simulation (cppsim)"
    echo "       RTL simulation (rtlsim) works without it"
fi

# Check if Xilinx tools actually work
if [ -n "$XILINX_VIVADO" ] && [ -d "$XILINX_VIVADO" ]; then
    if [ -x "$XILINX_VIVADO/bin/vivado" ]; then
        if ! "$XILINX_VIVADO/bin/vivado" -version >/dev/null 2>&1; then
            print_message "${YELLOW}[WARN]${NC} Vivado found but fails to run"
            ((WARNINGS++))
        fi
    else
        print_message "${YELLOW}[WARN]${NC} Vivado directory found but vivado executable missing"
        ((WARNINGS++))
    fi
fi

if [ -n "$XILINX_VITIS" ] && [ -d "$XILINX_VITIS" ]; then
    if [ -x "$XILINX_VITIS/bin/vitis" ]; then
        if ! "$XILINX_VITIS/bin/vitis" -version >/dev/null 2>&1; then
            print_message "${YELLOW}[WARN]${NC} Vitis found but fails to run"
            ((WARNINGS++))
        fi
    else
        print_message "${YELLOW}[WARN]${NC} Vitis directory found but vitis executable missing"
        ((WARNINGS++))
    fi
fi

# Check Xilinx setup
if [ -n "$BSMITH_XILINX_PATH" ] || [ -n "$XILINX_VIVADO" ]; then
    if [ -n "$BSMITH_XILINX_PATH" ] && [ ! -d "$BSMITH_XILINX_PATH" ]; then
        print_message "${YELLOW}[WARN]${NC} BSMITH_XILINX_PATH set but directory not found: $BSMITH_XILINX_PATH"
        ((WARNINGS++))
    elif [ -n "$XILINX_VIVADO" ] && [ ! -d "$XILINX_VIVADO" ]; then
        print_message "${YELLOW}[WARN]${NC} XILINX_VIVADO set but directory not found: $XILINX_VIVADO"
        ((WARNINGS++))
    fi
    
    # libtinfo5 check is already done above
fi

# Check Poetry
if ! command -v poetry >/dev/null 2>&1; then
    print_message "${RED}[ERROR]${NC} Poetry not found (required for dependency management)"
    echo "       Install with: curl -sSL https://install.python-poetry.org | sh"
    ((ERRORS++))
fi

# Summary
echo
if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ System ready for local Brainsmith installation${NC}"
        echo "  All essential requirements met"
    else
        echo -e "${GREEN}✓ Basic requirements met${NC} ($WARNINGS warnings)"
        echo "  Consider installing recommended packages for full functionality"
    fi
else
    echo -e "${RED}✗ Cannot proceed with local installation${NC} ($ERRORS errors, $WARNINGS warnings)"
    echo
    echo "To install missing packages:"
    if [ -n "$MISSING_ESSENTIAL" ]; then
        echo "  sudo apt-get update && sudo apt-get install$MISSING_ESSENTIAL"
    fi
    echo
    echo "Alternatively, use Docker workflow: ./ctl-docker.sh start"
fi

exit $ERRORS