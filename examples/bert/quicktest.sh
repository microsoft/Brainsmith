#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

# Set longer timeout for RTL simulation (BERT models can take longer)
export LIVENESS_THRESHOLD=10000000

# Optional diagnostics collection (disabled by default, enabled in CI/CD)
COLLECT_DIAGNOSTICS=${COLLECT_DIAGNOSTICS:-0}

# Function to collect diagnostics for debugging
collect_diagnostics() {
    if [ "$COLLECT_DIAGNOSTICS" != "1" ]; then
        return 0
    fi

    local exit_code=$1
    echo "=== Collecting diagnostics (exit code: ${exit_code}) ==="

    # Create diagnostics directory
    DIAG_DIR="/tmp/brainsmith_diagnostics"
    mkdir -p ${DIAG_DIR}

    # Environment check
    echo "FINN_ROOT=${FINN_ROOT:-NOT SET}" > ${DIAG_DIR}/env_check.txt
    echo "FINN_BUILD_DIR=${FINN_BUILD_DIR:-NOT SET}" >> ${DIAG_DIR}/env_check.txt

    # Check xsi.so availability
    if [ -n "$FINN_ROOT" ] && [ -f "$FINN_ROOT/finn_xsi/xsi.so" ]; then
        echo "xsi.so: FOUND" >> ${DIAG_DIR}/env_check.txt
    else
        echo "xsi.so: NOT FOUND" >> ${DIAG_DIR}/env_check.txt
    fi

    # Copy build logs if present
    find /tmp -name "build_dataflow.log" -o -name "*rtlsim*.log" 2>/dev/null | head -5 | xargs -I {} cp {} ${DIAG_DIR}/ 2>&1 || true

    echo "✓ Diagnostics collected in ${DIAG_DIR}"
}

# Set trap to collect diagnostics on exit (only if enabled)
if [ "$COLLECT_DIAGNOSTICS" = "1" ]; then
    trap 'collect_diagnostics $?' EXIT
    echo "Diagnostics collection ENABLED"
fi

echo "Running BERT Modern Demo Quick Test"
echo "==================================="

# Change to demo directory
cd "$(dirname "$0")"

# Generate folding config
echo "Generating folding configuration..."
python gen_folding_config.py \
    --simd 1 \
    --pe 1 \
    --num_layers 1 \
    -t 1 \
    -o ./configs/quicktest_folding.json

# Run BERT demo
echo "Running BERT demo with 1 layer..."
python bert_demo.py \
    -o quicktest \
    -n 4 \
    -l 1 \
    -z 64 \
    -i 256 \
    -b 4 \
    -q 32 \
    --blueprint bert_quicktest.yaml \
    --force


echo "Quick test completed!"
