#!/bin/bash
############################################################################
# End-to-end test runner for Hardware Kernel Generator
#
# Usage:
#   ./run_e2e_test.sh              # Run verification against golden
#   ./run_e2e_test.sh --golden     # Generate golden reference
############################################################################

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAINSMITH_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Hardware Kernel Generator End-to-End Test ===${NC}"
echo "Test directory: $SCRIPT_DIR"
echo "Brainsmith root: $BRAINSMITH_DIR"
echo

# Change to brainsmith directory
cd "$BRAINSMITH_DIR"

# Check if generating golden
if [[ "$1" == "--golden" ]] || [[ "$1" == "-g" ]]; then
    echo -e "${YELLOW}Generating golden reference...${NC}"
    python "$SCRIPT_DIR/test_e2e_generation.py" --generate-golden
else
    echo -e "${GREEN}Running verification against golden reference...${NC}"
    python "$SCRIPT_DIR/test_e2e_generation.py"
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Test completed successfully!${NC}"
else
    echo -e "\n${RED}❌ Test failed!${NC}"
    exit 1
fi