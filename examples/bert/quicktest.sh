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

    # Environment snapshot
    env | sort > ${DIAG_DIR}/env_vars.txt

    # FINN structure and xsi.so info
    if [ -n "$FINN_ROOT" ] && [ -d "$FINN_ROOT" ]; then
        ls -laR $FINN_ROOT > ${DIAG_DIR}/finn_root_listing.txt 2>&1
        find $FINN_ROOT/finn_xsi -type f -exec file {} \; > ${DIAG_DIR}/finn_xsi_files.txt 2>&1 || true

        if [ -f "$FINN_ROOT/finn_xsi/xsi.so" ]; then
            file $FINN_ROOT/finn_xsi/xsi.so > ${DIAG_DIR}/xsi_file_info.txt 2>&1
            ldd $FINN_ROOT/finn_xsi/xsi.so > ${DIAG_DIR}/xsi_dependencies.txt 2>&1 || true
            ls -l $FINN_ROOT/finn_xsi/xsi.so > ${DIAG_DIR}/xsi_stat.txt 2>&1
        else
            echo "xsi.so not found at $FINN_ROOT/finn_xsi/xsi.so" > ${DIAG_DIR}/xsi_missing.txt
        fi
    else
        echo "FINN_ROOT not set or directory does not exist" > ${DIAG_DIR}/finn_root_missing.txt
    fi

    # Build logs
    find /tmp -name "build_dataflow.log" -exec cp {} ${DIAG_DIR}/ \; 2>&1 || true
    find /tmp -name "*rtlsim*.log" -exec cp {} ${DIAG_DIR}/ \; 2>&1 || true

    # Python info
    python --version > ${DIAG_DIR}/python_version.txt 2>&1
    pip list > ${DIAG_DIR}/pip_list.txt 2>&1 || true

    echo "✓ Diagnostics collected in ${DIAG_DIR}"
    ls -la ${DIAG_DIR} || true
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