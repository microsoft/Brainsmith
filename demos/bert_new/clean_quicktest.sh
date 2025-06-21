#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Clean quick test for BERT demo - ensures no state pollution

set -e

echo "🧹 Clean BERT demo test - removing any previous state"

# Clean output directory
OUTPUT_DIR="clean_quicktest_output_$(date +%Y%m%d_%H%M%S)"
echo "📁 Using fresh output directory: $OUTPUT_DIR"

# Remove any existing directory with similar name
rm -rf "./${OUTPUT_DIR}"
rm -rf "/home/tafk/builds/brainsmith/${OUTPUT_DIR}"

echo ""
echo "🚀 Running BERT demo with auto-calculated folding..."
echo "This ensures a completely clean run with no state pollution"

python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir "./${OUTPUT_DIR}" \
    --target-fps 100 \
    --clock-period 5.0 \
    --board V80

echo ""
echo "✅ Clean test completed!"
echo "📁 Results in: ${OUTPUT_DIR}"