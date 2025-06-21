#!/bin/bash
# Test with extremely conservative settings to avoid FIFO shape mismatches
set -e

echo "Running BERT demo with very conservative auto-folding..."
echo "Using target_fps=1 to minimize parallelization"

# Run with very low target_fps to avoid aggressive parallelization
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./quicktest_conservative_output \
    --target-fps 1 \
    --clock-period 10.0 \
    --board V80

echo "Test completed!"