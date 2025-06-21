#!/usr/bin/env python3
import sys
sys.path.append('/home/tafk/dev/brainsmith-1/deps/finn/src')
sys.path.append('/home/tafk/dev/brainsmith-1/deps/qonnx/src')
import onnx
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.insert_fifo import _suitable_node
from qonnx.custom_op.registry import getCustomOp

import os
model_path = 'finn_output/intermediate_models/step_measure_rtlsim_performance.onnx'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)
model = ModelWrapper(model_path)

# Find problematic connections
print("Checking for folded shape mismatches...")
mismatches = []

for node in model.graph.node:
    if _suitable_node(node):
        n0 = getCustomOp(node)
        for idx_out, output_name in enumerate(node.output):
            consumers = model.find_direct_successors(node)
            if consumers is not None and len(consumers) == 1:
                consumer = consumers[0]
                if _suitable_node(consumer):
                    n1 = getCustomOp(consumer)
                    # Get folded shapes
                    try:
                        fld_shape = n0.get_folded_output_shape(ind=idx_out)
                        for idx, inp in enumerate(consumer.input):
                            if inp == output_name:
                                fld_shape_2 = n1.get_folded_input_shape(ind=idx)
                                if tuple(fld_shape) != tuple(fld_shape_2):
                                    print(f'\nShape mismatch found:')
                                    print(f'  Producer: {node.name} ({node.op_type})')
                                    print(f'    Output shape: {fld_shape}')
                                    print(f'  Consumer: {consumer.name} ({consumer.op_type})')
                                    print(f'    Input shape: {fld_shape_2}')
                                    print(f'  Connection: {output_name}')
                                    
                                    # Check if stream widths match but sizes don't
                                    if fld_shape[-1] == fld_shape_2[-1]:
                                        print(f'  Stream widths match: {fld_shape[-1]}')
                                        import numpy as np
                                        size1 = np.prod(fld_shape)
                                        size2 = np.prod(fld_shape_2)
                                        print(f'  Size mismatch: {size1} vs {size2}')
                    except Exception as e:
                        print(f"Error checking {node.name} -> {consumer.name}: {e}")

print("\nDone checking.")