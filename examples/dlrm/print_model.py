import onnx
from onnxscript import ir

proto = onnx.load("dlrm_s_pytorch.onnx")
ir_model = ir.serde.deserialize_model(proto)

# Export to ONNX Script Python format
python_code = ir_model.display()
print(python_code)
