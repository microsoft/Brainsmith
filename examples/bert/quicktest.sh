#!/bin/bash

set -e

# Set longer timeout for RTL simulation (BERT models can take longer)
export LIVENESS_THRESHOLD=10000000

echo "Running BERT Modern Demo Quick Test"
echo "==================================="

# Change to demo directory
cd "$(dirname "$0")"

# Clean up any existing quicktest build directory
if [ -d "${BSMITH_BUILD_DIR}/quicktest" ]; then
    echo "Removing existing quicktest build directory..."
    rm -rf "${BSMITH_BUILD_DIR}/quicktest"
fi

# Generate folding config
echo "Generating folding configuration..."
python gen_folding_config.py \
    --simd 4 \
    --pe 4 \
    --num_layers 1 \
    -t 1 \
    -o ./configs/quicktest_folding.json

# Run BERT demo
echo "Running BERT demo with 1 layer..."
python bert_demo.py \
    -o quicktest \
    -n 4 \
    -l 1 \
    -z 64 \
    -i 256 \
    -b 4 \
    -q 32 \
    --blueprint bert_quicktest.yaml

echo "Quick test completed!"