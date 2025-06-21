#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Quick test with complete folding config including FIFO depths
# This should skip the expensive FIFO sizing step!

set -e

echo "🚀 BERT demo test with complete folding config (includes FIFO depths)"
echo "📋 Using folding configuration extracted from successful bert_direct run"
echo "⚡ This should skip the expensive FIFO sizing step!"

# Configuration
OUTPUT_DIR="quicktest_complete_folding_validation"
FOLDING_CONFIG="./l1_simd12_pe8_with_fifos.json"

# Clean up previous run if it exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  Removing previous output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# Check folding config exists
if [ ! -f "$FOLDING_CONFIG" ]; then
    echo "❌ Complete folding config not found: $FOLDING_CONFIG"
    echo "💡 Run: python extract_fifo_depths_to_folding.py to create it"
    exit 1
fi

echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Folding config: $FOLDING_CONFIG (with FIFO depths)"
echo ""

# Show FIFO config summary
echo "📊 Folding config summary:"
python3 -c "
import json
with open('$FOLDING_CONFIG', 'r') as f:
    config = json.load(f)
    total = len(config) - 1
    with_fifo = sum(1 for v in config.values() if isinstance(v, dict) and ('inFIFODepths' in v or 'outFIFODepth' in v))
    print(f'  Total layers: {total}')
    print(f'  Layers with FIFO config: {with_fifo}')
"
echo ""

# Run with complete folding config
python end2end_bert.py \
    --num-heads 12 \
    --num-layers 1 \
    --hidden-size 384 \
    --intermediate-size 1536 \
    --output-dir "./$OUTPUT_DIR" \
    --clock-period 5.0 \
    --board V80 \
    -p "$FOLDING_CONFIG"

echo ""
echo "✅ Test completed!"
echo "📁 Results in: $OUTPUT_DIR"

# Check if FIFO sizing was skipped
BUILD_LOG="$OUTPUT_DIR/finn_output/build_dataflow.log"
if [ -f "$BUILD_LOG" ]; then
    echo ""
    echo "🔍 Checking if FIFO sizing was skipped..."
    if grep -q "step_set_fifo_depths" "$BUILD_LOG"; then
        echo "⚠️  FIFO sizing step was still executed"
        echo "📋 Checking for FIFO reuse..."
        
        # Look for evidence of FIFO depth reuse
        if grep -q "Reusing folding config FIFO depths" "$BUILD_LOG" || \
           grep -q "FIFO depths already specified" "$BUILD_LOG" || \
           grep -q "Skipping FIFO depth calculation" "$BUILD_LOG"; then
            echo "✅ FIFO depths were reused from config!"
        else
            echo "🤔 FIFO sizing may have recalculated depths"
        fi
        
        # Show timing for FIFO step
        echo ""
        echo "⏱️  FIFO sizing step timing:"
        grep -A2 -B2 "step_set_fifo_depths" "$BUILD_LOG" | grep -E "(Running step:|elapsed time)" | tail -5
    else
        echo "✅ FIFO sizing step was completely skipped!"
    fi
    
    echo ""
    echo "📊 Build summary:"
    echo "  Log file: $BUILD_LOG"
    echo "  Total lines: $(wc -l < "$BUILD_LOG")"
    grep -c "Running step:" "$BUILD_LOG" | xargs echo "  Total steps executed:"
fi

echo ""
echo "📁 Output structure for validation:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (Output directory not yet created)"
if [ -d "$OUTPUT_DIR/finn_output/intermediate_models" ]; then
    echo ""
    echo "📄 Intermediate models:"
    ls -la "$OUTPUT_DIR/finn_output/intermediate_models/" | grep -E "(step_set_fifo|step_measure)" | tail -5
fi