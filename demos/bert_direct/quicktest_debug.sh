#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# BERT Direct Demo Quick Test with Debug Output

set -e

echo "🚀 BERT Direct Demo - Debug Test"
echo "⚡ Testing model generation with verbose output"

# Run with explicit Python debugging
python -u end2end_bert_direct.py \
    -o "debug_test" \
    -z 384 \
    -l 1 \
    -n 12 \
    -i 1536 \
    -q 128 \
    -b 8 \
    -p "./l1_simd12_pe8.json" \
    -d false \
    -x \
    -s "generate_reference_io_step" 2>&1 | tee debug_output.log

echo "✅ Debug test completed. Check debug_output.log for details."