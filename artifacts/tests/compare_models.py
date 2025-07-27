#!/usr/bin/env python3
import onnx

# Check initial model
model1 = onnx.load('/home/tafk/builds/brainsmith/work/debug_models/00_initial_brevitas.onnx')
print("Initial Brevitas model:")
print(f"  Inputs: {[inp.name for inp in model1.graph.input]}")
print(f"  Outputs: {[out.name for out in model1.graph.output]}")

# Check after cleanup
model2 = onnx.load('/home/tafk/builds/brainsmith/work/debug_models/02_after_qonnx_cleanup.onnx')
print("\nAfter QONNX cleanup:")
print(f"  Inputs: {[inp.name for inp in model2.graph.input]}")
print(f"  Outputs: {[out.name for out in model2.graph.output]}")

# Check model saved to /tmp
model3 = onnx.load('/tmp/bert_test_run/root/input.onnx')
print("\nModel in /tmp/bert_test_run/root/input.onnx:")
print(f"  Inputs: {[inp.name for inp in model3.graph.input]}")
print(f"  Outputs: {[out.name for out in model3.graph.output]}")