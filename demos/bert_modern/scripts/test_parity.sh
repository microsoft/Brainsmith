#!/bin/bash
# Test script to verify parity between old and new BERT demos

set -e

echo "BERT Demo Parity Test"
echo "===================="

# Configuration
LAYERS=1
HIDDEN=384
HEADS=12
INTERMEDIATE=1536
SIMD=12
PE=8

# Paths
OLD_DEMO_DIR="../../bert_old"
NEW_DEMO_DIR=".."
BUILD_DIR="${BSMITH_BUILD_DIR:-./build}"

# Function to run old demo
run_old_demo() {
    echo "Running OLD demo..."
    cd "$OLD_DEMO_DIR"
    
    # Generate folding config
    python gen_initial_folding.py \
        --simd $SIMD \
        --pe $PE \
        --num_layers $LAYERS \
        -t 1 \
        -o ./configs/test_config.json
    
    # Run demo
    python end2end_bert.py \
        -o parity_test_old \
        -n $HEADS \
        -l $LAYERS \
        -z $HIDDEN \
        -i $INTERMEDIATE \
        -p ./configs/test_config.json \
        --stop-step step_hw_codegen
    
    cd -
}

# Function to run new demo
run_new_demo() {
    echo "Running NEW demo..."
    cd "$NEW_DEMO_DIR"
    
    # Generate folding config
    python gen_folding_config.py \
        --simd $SIMD \
        --pe $PE \
        --num_layers $LAYERS \
        -t 1 \
        -o ./configs/test_config.json
    
    # Run demo
    python bert_demo.py \
        -o parity_test_new \
        -n $HEADS \
        -l $LAYERS \
        -z $HIDDEN \
        -i $INTERMEDIATE \
        -p ./configs/test_config.json \
        --stop-step step_hw_codegen
    
    cd -
}

# Function to compare outputs
compare_outputs() {
    echo "Comparing outputs..."
    
    OLD_BUILD="$BUILD_DIR/parity_test_old"
    NEW_BUILD="$BUILD_DIR/parity_test_new"
    
    # Check if builds exist
    if [ ! -d "$OLD_BUILD" ]; then
        echo "ERROR: Old build not found at $OLD_BUILD"
        exit 1
    fi
    
    if [ ! -d "$NEW_BUILD" ]; then
        echo "ERROR: New build not found at $NEW_BUILD"
        exit 1
    fi
    
    # Compare final models
    echo "Comparing final models..."
    python3 << EOF
import onnx
import numpy as np

old_model = onnx.load("$OLD_BUILD/output.onnx")
new_model = onnx.load("$NEW_BUILD/output.onnx")

# Compare graph structure
old_nodes = [(n.name, n.op_type) for n in old_model.graph.node]
new_nodes = [(n.name, n.op_type) for n in new_model.graph.node]

if old_nodes == new_nodes:
    print("✓ Graph structure matches")
else:
    print("✗ Graph structure differs")
    print(f"  Old: {len(old_nodes)} nodes")
    print(f"  New: {len(new_nodes)} nodes")
    
    # Find differences
    old_set = set(old_nodes)
    new_set = set(new_nodes)
    only_old = old_set - new_set
    only_new = new_set - old_set
    
    if only_old:
        print(f"  Only in old: {list(only_old)[:5]}...")
    if only_new:
        print(f"  Only in new: {list(only_new)[:5]}...")

# Compare input/output shapes
old_in = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) 
          for i in old_model.graph.input]
new_in = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) 
          for i in new_model.graph.input]

if old_in == new_in:
    print("✓ Input shapes match")
else:
    print("✗ Input shapes differ")

EOF
    
    # Compare build artifacts
    echo ""
    echo "Comparing build artifacts..."
    
    # List of key files to check
    KEY_FILES=(
        "intermediate_models/step_create_dataflow_partition.onnx"
        "intermediate_models/step_specialize_layers.onnx"
        "intermediate_models/step_hw_codegen.onnx"
    )
    
    for file in "${KEY_FILES[@]}"; do
        if [ -f "$OLD_BUILD/$file" ] && [ -f "$NEW_BUILD/$file" ]; then
            echo "✓ Both have $file"
        else
            echo "✗ Missing $file in one build"
        fi
    done
    
    echo ""
    echo "Parity test completed!"
}

# Main execution
echo "Starting parity test between old and new BERT demos"
echo ""

# Check if running in container
if [ -z "$SMITHY" ]; then
    echo "WARNING: Not running in smithy container"
    echo "Run with: smithy ./scripts/test_parity.sh"
fi

# Run demos
#run_old_demo
#run_new_demo

# Compare results
#compare_outputs

echo ""
echo "Note: This script is for testing only."
echo "Uncomment the function calls to actually run the test."