#!/bin/bash
# Quick test script for BERT accelerator using pre-computed folding config
# This avoids the 12-hour RTL simulation for FIFO sizing
set -e

echo "🚀 Running BERT demo with pre-computed folding configuration..."
echo "📋 This skips RTL simulation for FIFO sizing (saves ~12 hours)"

# Generate HLS-compatible folding configuration
echo "🔧 Generating HLS-compatible folding configuration..."
python gen_compatible_folding.py -n 1 -o ./compatible_folding_1L.json

# Run with pre-computed folding config
echo "🏃 Running BERT accelerator generation..."
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./quicktest_output_folded \
    --param ./compatible_folding_1L.json \
    --clock-period 5.0 \
    --board V80

echo "✅ Test completed successfully with pre-computed folding!"
echo "📁 Output in: ./quicktest_output_folded"