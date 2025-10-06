#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

# Set longer timeout for RTL simulation (BERT models can take longer)
export LIVENESS_THRESHOLD=10000000

echo "Running BERT Modern Demo Quick Test"
echo "==================================="

# Change to demo directory
cd "$(dirname "$0")"

# Generate folding config
echo "Generating folding configuration..."
python gen_folding_config.py \
    --simd 1 \
    --pe 1 \
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
    --blueprint bert_quicktest.yaml \
    --force


echo "Quick test completed!"