#!/usr/bin/env python3
"""Debug the remaining zero-dimension nodes."""

import onnx
import qonnx.custom_op.registry as registry

def analyze_zero_nodes():
    model_path = "finn_output/intermediate_models/infer_hardware_step.onnx"
    model = onnx.load(model_path)
    
    print("Analyzing zero-dimension nodes in the model...")
    print("=" * 60)
    
    # Find all nodes and their connections
    zero_nodes = ["DuplicateStreams_hls_1", "Thresholding_rtl_9", 
                  "ElementwiseAdd_hls_1", "LayerNorm_hls_1"]
    
    for node_name in zero_nodes:
        for node in model.graph.node:
            if node.name == node_name:
                print(f"\nNode: {node.name}")
                print(f"Type: {node.op_type}")
                print(f"Inputs: {list(node.input)}")
                print(f"Outputs: {list(node.output)}")
                
                # Get attributes
                try:
                    inst = registry.getCustomOp(node)
                    if hasattr(inst, 'get_nodeattr'):
                        print(f"numInputVectors: {inst.get_nodeattr('numInputVectors')}")
                except:
                    pass
                
                # Find producer of inputs
                for inp in node.input:
                    for prod_node in model.graph.node:
                        if inp in prod_node.output:
                            print(f"  Input '{inp}' produced by: {prod_node.name} ({prod_node.op_type})")
    
    # Check model inputs
    print("\n\nModel inputs:")
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            else:
                shape.append('?')
        print(f"  {inp.name}: {shape}")

if __name__ == "__main__":
    analyze_zero_nodes()