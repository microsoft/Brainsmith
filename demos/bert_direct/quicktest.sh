#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# BERT Direct Demo Quick Test
# Tests direct DataflowBuildConfig execution bypassing 6-entrypoint system
#
# Note: This demo uses pre-generated reference IO tensors to avoid a 6-minute
# computation delay during the generate_reference_io step. The tensors were
# generated from a BERT model with head/tail removed and are stored in this directory.

set -e

echo "🚀 BERT Direct Demo - Quick Test"
echo "⚡ Testing BrainSmith transforms directly with FINN (bypassing 6-entrypoint)"
echo "📝 Using cached reference IO tensors for faster execution"

# Test parameters matching working old demo
OUTPUT_DIR="direct_test_$(date +%Y%m%d_%H%M%S)"
HIDDEN_SIZE=384
NUM_LAYERS=1
NUM_HEADS=12
INTERMEDIATE_SIZE=1536
SEQUENCE_LENGTH=128
BITWIDTH=8

# Use conservative folding config for fast testing
FOLDING_CONFIG="./l1_simd12_pe8.json"

echo "📋 Test Configuration:"
echo "  Model: ${NUM_LAYERS}L x ${HIDDEN_SIZE}D x ${NUM_HEADS}H"
echo "  Sequence: ${SEQUENCE_LENGTH}"
echo "  Bitwidth: ${BITWIDTH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Folding: ${FOLDING_CONFIG}"

# Check folding config exists
if [ ! -f "$FOLDING_CONFIG" ]; then
    echo "❌ Folding config not found: $FOLDING_CONFIG"
    echo "💡 Run: cp ../bert_new/l1_simd12_pe8.json ."
    exit 1
fi

echo ""
echo "🏗️  Starting direct BERT demo execution..."

# Run direct demo (no DCP generation for speed)
# Use unbuffered output for real-time progress monitoring
PYTHONUNBUFFERED=1 python -u end2end_bert_direct.py \
    -o "$OUTPUT_DIR" \
    -z "$HIDDEN_SIZE" \
    -l "$NUM_LAYERS" \
    -n "$NUM_HEADS" \
    -i "$INTERMEDIATE_SIZE" \
    -q "$SEQUENCE_LENGTH" \
    -b "$BITWIDTH" \
    -p "$FOLDING_CONFIG" \
    -d false \
    -x

echo ""
echo "✅ Direct demo execution completed!"
echo "📁 Results in: ${BSMITH_BUILD_DIR}/${OUTPUT_DIR}"
echo ""
echo "🔍 Key artifacts to check:"
echo "  - ${BSMITH_BUILD_DIR}/${OUTPUT_DIR}/intermediate_models/ (step outputs)"
echo "  - ${BSMITH_BUILD_DIR}/${OUTPUT_DIR}/output.onnx (final model)"
echo "  - ${BSMITH_BUILD_DIR}/${OUTPUT_DIR}/stitched_ip/ (RTL outputs)"
echo ""
echo "💡 Compare with old demo results to validate transforms"