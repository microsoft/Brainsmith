#!/usr/bin/env python3
"""Compare outputs of infer_hardware vs infer_kernels."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import onnx
from brainsmith.core.plugins import get_transform, get_transforms_by_metadata
from brainsmith.core import apply_transforms

def compare_inference_approaches():
    print("=== Comparing infer_hardware vs infer_kernels ===\n")
    
    # Load a simple test model
    test_model_path = "/home/tafk/builds/brainsmith/test_infer_hardware/root/intermediate_models/bert_streamlining_step.onnx"
    
    if not Path(test_model_path).exists():
        print(f"Test model not found: {test_model_path}")
        return
    
    # Load model
    from qonnx.core.modelwrapper import ModelWrapper
    model = ModelWrapper(test_model_path)
    
    # Create a mock config object
    class MockConfig:
        def __init__(self):
            self.output_dir = "/tmp/test_compare"
            self.kernel_selections = [
                ('LayerNorm', 'LayerNorm'),
                ('DuplicateStreams', 'DuplicateStreams'),
                ('ElementwiseBinaryOperation', 'ElementwiseBinaryOperation'),
                ('Shuffle', 'Shuffle'),
                ('HWSoftmax', 'HWSoftmax'),
                ('Thresholding', 'Thresholding'),
                ('MVAU', 'MVAU')
            ]
    
    cfg = MockConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Test 1: Run infer_hardware approach
    print("1. Testing infer_hardware approach...")
    model1 = ModelWrapper(test_model_path)
    
    # Apply transforms in order (from core_steps.py)
    model1 = apply_transforms(model1, [
        'InferLayerNorm',
        'InferDuplicateStreamsLayer'
    ])
    
    # Count nodes with custom domains
    custom_nodes1 = []
    for node in model1.graph.node:
        if node.domain and node.domain != "":
            custom_nodes1.append(f"{node.op_type} (domain={node.domain})")
    
    print(f"   Custom nodes after first 2 transforms: {len(custom_nodes1)}")
    
    model1 = apply_transforms(model1, [
        'InferElementwiseBinaryOperation',
        'InferShuffle',
        'InferHWSoftmax',
        'InferThresholdingLayer',
        'InferQuantizedMatrixVectorActivation',
        'EnsureCustomOpsetImports'
    ])
    
    # Count final nodes
    custom_nodes1_final = []
    for node in model1.graph.node:
        if node.domain and node.domain != "":
            custom_nodes1_final.append(f"{node.op_type} (domain={node.domain})")
    
    print(f"   Final custom nodes: {len(custom_nodes1_final)}")
    print(f"   Unique op types: {set(n.split(' ')[0] for n in custom_nodes1_final)}")
    
    # Test 2: Run infer_kernels approach
    print("\n2. Testing infer_kernels approach...")
    model2 = ModelWrapper(test_model_path)
    
    # Apply inference for each kernel
    for kernel_name, backend in cfg.kernel_selections:
        # Find transforms by metadata
        transforms = get_transforms_by_metadata(kernel=kernel_name)
        if transforms:
            transform_name = transforms[0]
            print(f"   {kernel_name} -> {transform_name}")
            Transform = get_transform(transform_name)
            model2 = model2.transform(Transform())
        else:
            print(f"   {kernel_name} -> NO TRANSFORM FOUND")
    
    # Apply EnsureCustomOpsetImports
    model2 = apply_transforms(model2, ['EnsureCustomOpsetImports'])
    
    # Count final nodes
    custom_nodes2_final = []
    for node in model2.graph.node:
        if node.domain and node.domain != "":
            custom_nodes2_final.append(f"{node.op_type} (domain={node.domain})")
    
    print(f"\n   Final custom nodes: {len(custom_nodes2_final)}")
    print(f"   Unique op types: {set(n.split(' ')[0] for n in custom_nodes2_final)}")
    
    # Compare
    print("\n3. Comparison:")
    print(f"   infer_hardware: {len(custom_nodes1_final)} custom nodes")
    print(f"   infer_kernels:  {len(custom_nodes2_final)} custom nodes")
    
    if len(custom_nodes1_final) != len(custom_nodes2_final):
        print("   ❌ Different number of custom nodes!")
        
        # Find differences
        ops1 = sorted([n.split(' ')[0] for n in custom_nodes1_final])
        ops2 = sorted([n.split(' ')[0] for n in custom_nodes2_final])
        
        from collections import Counter
        count1 = Counter(ops1)
        count2 = Counter(ops2)
        
        print("\n   Node count differences:")
        all_ops = set(count1.keys()) | set(count2.keys())
        for op in sorted(all_ops):
            c1 = count1.get(op, 0)
            c2 = count2.get(op, 0)
            if c1 != c2:
                print(f"     {op}: infer_hardware={c1}, infer_kernels={c2}")
    else:
        print("   ✓ Same number of custom nodes")

if __name__ == "__main__":
    compare_inference_approaches()