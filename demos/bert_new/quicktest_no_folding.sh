#!/bin/bash
# Quick test script without pre-computed folding config
set -e

echo "Running BERT demo without folding config..."
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./quicktest_output_no_folding \
    --target-fps 1 \
    --clock-period 5.0 \
    --board V80

echo "Test completed!"