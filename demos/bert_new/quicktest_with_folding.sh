#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Quick test with pre-computed folding config (matching bert_direct)

set -e

echo "🚀 BERT demo test with pre-computed folding config"
echo "📋 Using known-working folding configuration from bert_direct"

# Configuration
OUTPUT_DIR="quicktest_folding_$(date +%Y%m%d_%H%M%S)"
FOLDING_CONFIG="./l1_simd12_pe8.json"

# Check folding config exists
if [ ! -f "$FOLDING_CONFIG" ]; then
    echo "❌ Folding config not found: $FOLDING_CONFIG"
    exit 1
fi

echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Folding config: $FOLDING_CONFIG"
echo ""

# Run with folding config instead of target_fps
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