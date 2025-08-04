#!/bin/bash
############################################################################
# Run all RTL Parser demo files
# 
# This script parses each demo file and shows the extracted metadata.
# Run from the brainsmith-2 root directory.
############################################################################

DEMO_DIR="tests/tools/kernel_integrator/rtl_parser/demo_rtl"
PARSER_SCRIPT="tests/tools/kernel_integrator/rtl_parser/rtl_parser_demo.py"

echo "==========================================="
echo "Running RTL Parser on all demo files"
echo "==========================================="
echo

# Check if we're in the right directory
if [ ! -f "smithy" ]; then
    echo "Error: Please run this script from the brainsmith-2 root directory"
    exit 1
fi

# Run parser on each demo file
for demo in $DEMO_DIR/*.sv; do
    if [ -f "$demo" ]; then
        filename=$(basename "$demo")
        echo "-------------------------------------------"
        echo "Processing: $filename"
        echo "-------------------------------------------"
        
        # Run the parser
        ./smithy exec "python $PARSER_SCRIPT $demo" 2>&1
        
        echo
        echo
    fi
done

echo "==========================================="
echo "All demos processed!"
echo "==========================================="