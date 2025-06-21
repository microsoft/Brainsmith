#!/usr/bin/env python3
"""Check the global input shape in the model."""

import onnx

def check_input_shape(model_path):
    """Check the shape of global_in."""
    model = onnx.load(model_path)
    
    print(f"\nChecking: {model_path}")
    print("-" * 60)
    
    # Check graph inputs
    for inp in model.graph.input:
        if inp.name == "global_in":
            print(f"Input: {inp.name}")
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            print(f"Shape: {shape}")
            print(f"Data type: {inp.type.tensor_type.elem_type}")

# Check models before and after head/tail removal
models = [
    "finn_output/intermediate_models/cleanup_step.onnx",
    "finn_output/intermediate_models/remove_head_step.onnx",
    "finn_output/intermediate_models/remove_tail_step.onnx",
    "finn_output/intermediate_models/qonnx_to_finn_step.onnx",
    "finn_output/intermediate_models/streamlining_step.onnx",
    "finn_output/intermediate_models/infer_hardware_step.onnx"
]

for model in models:
    check_input_shape(model)