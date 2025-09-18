#!/bin/bash
# validate_env.sh - Check if system meets Brainsmith requirements

# Color codes
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Checking Brainsmith requirements..."
echo

ERRORS=0
WARNINGS=0

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]] || [[ "${VERSION_ID%%.*}" -lt 22 ]]; then
        echo -e "${RED}[ERROR]${NC} Requires Ubuntu 22.04 or newer (found: $ID $VERSION_ID)"
        ((ERRORS++))
    fi
fi

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Python3 not found"
    ((ERRORS++))
else
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PY_VERSION" != "3.10" ]] && [[ "$PY_VERSION" != "3.11" ]]; then
        echo -e "${YELLOW}[WARN]${NC} Python $PY_VERSION found (recommended: 3.10 or 3.11)"
        ((WARNINGS++))
    fi
fi

# Check essential packages
MISSING_ESSENTIAL=""
for pkg in build-essential git wget zip unzip python3-pip python3-venv; do
    if ! dpkg -l 2>/dev/null | grep -q "^ii  $pkg "; then
        MISSING_ESSENTIAL="$MISSING_ESSENTIAL $pkg"
        ((ERRORS++))
    fi
done

if [ -n "$MISSING_ESSENTIAL" ]; then
    echo -e "${RED}[ERROR]${NC} Missing packages:$MISSING_ESSENTIAL"
fi

# Check optional packages
if ! dpkg -l 2>/dev/null | grep -q "^ii  pybind11-dev "; then
    echo -e "${YELLOW}[WARN]${NC} pybind11-dev not found (optional: enables finnxsi)"
    ((WARNINGS++))
fi

# Check GUI libraries for matplotlib
GUI_MISSING=""
for lib in libglib2.0-0 libsm6 libxext6; do
    if ! dpkg -l 2>/dev/null | grep -q "^ii  $lib "; then
        GUI_MISSING="$GUI_MISSING $lib"
    fi
done
if [ -n "$GUI_MISSING" ]; then
    echo -e "${YELLOW}[WARN]${NC} GUI libraries missing (optional: matplotlib displays)"
    ((WARNINGS++))
fi

# Check Xilinx if user has set XILINX_VIVADO
if [ -n "$XILINX_VIVADO" ]; then
    if [ ! -d "$XILINX_VIVADO" ]; then
        echo -e "${YELLOW}[WARN]${NC} XILINX_VIVADO set but directory not found: $XILINX_VIVADO"
        ((WARNINGS++))
    elif ! dpkg -l 2>/dev/null | grep -q "^ii  libtinfo5 "; then
        echo -e "${YELLOW}[WARN]${NC} libtinfo5 missing (needed for Vivado compatibility)"
        ((WARNINGS++))
    fi
fi

# Summary
echo
if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ Ready to proceed${NC}"
    else
        echo -e "${GREEN}✓ Basic requirements met${NC} ($WARNINGS warnings)"
    fi
else
    echo -e "${RED}✗ Cannot proceed${NC} ($ERRORS errors, $WARNINGS warnings)"
    echo
    echo "To fix:"
    echo "  sudo apt-get update && sudo apt-get install$MISSING_ESSENTIAL"
fi

exit $ERRORS