#!/bin/bash
# Quick test script - matches functionality of old quicktest.sh

set -e

# Set longer timeout for RTL simulation (BERT models can take longer)
export LIVENESS_THRESHOLD=10000000

echo "Running BERT Modern Demo Quick Test"
echo "==================================="

# Change to demo directory
cd "$(dirname "$0")/.."

# Generate folding config
echo "Generating folding configuration..."
python gen_folding_config.py \
    --simd 12 \
    --pe 8 \
    --num_layers 1 \
    -t 1 \
    -o ./configs/l1_simd12_pe8.json

# Run BERT demo
echo "Running BERT demo with 1 layer..."
python bert_demo.py \
    -o quicktest \
    -n 12 \
    -l 1 \
    -z 384 \
    -i 1536 \
    -p ./configs/l1_simd12_pe8.json

echo "Quick test completed!"