#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper
import os

model_path = "/home/tafk/builds/brainsmith/work/intermediate_models/simp.onnx"
if os.path.exists(model_path):
    model = ModelWrapper(model_path)
    
    # Find the Mul node
    for node in model.graph.node:
        if node.op_type == "Mul" and "scaled_dot_product_attention" in node.name:
            print(f"Found Mul node: {node.name}")
            print(f"Inputs: {list(node.input)}")
            print(f"Outputs: {list(node.output)}")
            
            # Check datatypes
            for inp in node.input:
                try:
                    dt = model.get_tensor_datatype(inp)
                    print(f"  Input '{inp}' datatype: {dt}")
                except:
                    print(f"  Input '{inp}' datatype: Unknown")
                    
            for out in node.output:
                try:
                    dt = model.get_tensor_datatype(out)
                    print(f"  Output '{out}' datatype: {dt}")
                except:
                    print(f"  Output '{out}' datatype: Unknown")
                    
            # Check if it has successors
            successors = model.find_direct_successors(node)
            print(f"  Has {len(successors)} successors")
            if successors:
                print(f"    First successor: {successors[0].op_type}")