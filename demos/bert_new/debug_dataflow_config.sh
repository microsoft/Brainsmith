#!/bin/bash
# Debug script to print DataflowBuildConfig parameters
set -e

echo "Running debug test to print DataflowBuildConfig parameters..."
echo "This will fail intentionally after printing the config."
echo ""

# Run with minimal config to trigger debug output
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./debug_output \
    --target-fps 1 \
    --clock-period 5.0 \
    --board V80 2>&1 | tee debug_dataflow_config.log

echo "Check debug_dataflow_config.log for the DataflowBuildConfig parameters"