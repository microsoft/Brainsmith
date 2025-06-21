#!/bin/bash
# Quick test script for BERT accelerator using ultra-small configuration
set -e

echo "Running BERT demo with auto-calculated folding..."
echo "This avoids the FIFO shape mismatch issue with pre-computed folding configs"

# Run without pre-computed folding config - let FINN auto-calculate
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./quicktest_output \
    --target-fps 100 \
    --clock-period 5.0 \
    --board V80

echo "Test completed successfully!"
