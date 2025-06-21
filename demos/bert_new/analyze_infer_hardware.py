#!/usr/bin/env python3
"""Analyze what happens during infer_hardware_step that introduces zero dimensions."""

import onnx
import qonnx.custom_op.registry as registry

def analyze_node_shapes(model_path, node_names):
    """Get detailed info about specific nodes."""
    model = onnx.load(model_path)
    
    for node in model.graph.node:
        if node.name in node_names or any(name in node.name for name in node_names):
            print(f"\nNode: {node.name}")
            print(f"Type: {node.op_type}")
            print(f"Inputs: {list(node.input)}")
            print(f"Outputs: {list(node.output)}")
            
            # Get attributes
            attrs = {}
            for attr in node.attribute:
                if attr.type == 1:  # FLOAT
                    attrs[attr.name] = attr.f
                elif attr.type == 2:  # INT
                    attrs[attr.name] = attr.i
                elif attr.type == 3:  # STRING
                    attrs[attr.name] = attr.s.decode()
                elif attr.type == 7:  # INTS
                    attrs[attr.name] = list(attr.ints)
            
            print(f"Attributes: {attrs}")
            
            try:
                inst = registry.getCustomOp(node)
                if hasattr(inst, 'get_folded_output_shape'):
                    print(f"Folded output shape: {inst.get_folded_output_shape()}")
                if hasattr(inst, 'get_normal_output_shape'):
                    print(f"Normal output shape: {inst.get_normal_output_shape()}")
                if hasattr(inst, 'get_nodeattr'):
                    # Try to get folding parameters
                    for attr in ['PE', 'SIMD', 'MW', 'MH', 'numInputVectors']:
                        try:
                            val = inst.get_nodeattr(attr)
                            if val is not None:
                                print(f"{attr}: {val}")
                        except:
                            pass
            except:
                pass

def compare_models(before_path, after_path):
    """Compare specific nodes before and after a transformation."""
    print("=" * 80)
    print("BEFORE infer_hardware_step (streamlining_step.onnx):")
    print("=" * 80)
    
    # Look at nodes that will become Thresholding/DuplicateStreams
    before_nodes = ["MultiThreshold_0", "MultiThreshold_1", "MultiThreshold_2", 
                    "MultiThreshold_9", "global_in", "LayerNorm"]
    analyze_node_shapes(before_path, before_nodes)
    
    print("\n" + "=" * 80)
    print("AFTER infer_hardware_step:")
    print("=" * 80)
    
    after_nodes = ["Thresholding", "DuplicateStreams", "LayerNorm", "ElementwiseAdd"]
    analyze_node_shapes(after_path, after_nodes)

if __name__ == "__main__":
    compare_models(
        "finn_output/intermediate_models/streamlining_step.onnx",
        "finn_output/intermediate_models/infer_hardware_step.onnx"
    )