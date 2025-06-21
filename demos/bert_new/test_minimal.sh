#!/bin/bash
# Test with minimal folding to see if it works
set -e

echo "Testing with minimal folding configuration..."
python end2end_bert.py -p ./minimal_folding.json \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./test_minimal_output \
    --target-fps 100 \
    --clock-period 5.0 \
    --board V80