#!/usr/bin/env python3
"""Trace dataflow to understand zero dimensions."""

import onnx

def trace_dataflow():
    # Check models at different stages
    models_to_check = [
        ("After remove_tail", "finn_output/intermediate_models/remove_tail_step.onnx"),
        ("After qonnx_to_finn", "finn_output/intermediate_models/qonnx_to_finn_step.onnx"),
        ("After streamlining", "finn_output/intermediate_models/streamlining_step.onnx"),
        ("After infer_hardware", "finn_output/intermediate_models/infer_hardware_step.onnx"),
    ]
    
    for stage, model_path in models_to_check:
        try:
            model = onnx.load(model_path)
            print(f"\n{stage}:")
            print("-" * 40)
            
            # Count inputs and outputs
            print(f"Inputs: {len(model.graph.input)}")
            for inp in model.graph.input:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        shape.append(f"'{dim.dim_param}'")
                    else:
                        shape.append('?')
                print(f"  {inp.name}: {shape}")
            
            print(f"Outputs: {len(model.graph.output)}")
            for out in model.graph.output:
                print(f"  {out.name}")
                
            # Look for specific problematic nodes
            problematic = ["DuplicateStreams", "LayerNorm", "ElementwiseAdd"]
            print("Relevant nodes:")
            for node in model.graph.node:
                if any(p in node.op_type for p in problematic):
                    print(f"  {node.name} ({node.op_type})")
                    
        except Exception as e:
            print(f"\n{stage}: Error - {e}")

if __name__ == "__main__":
    trace_dataflow()