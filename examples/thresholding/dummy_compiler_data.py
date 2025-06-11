# Dummy compiler data file for testing

onnx_patterns = []

def cost_function(*args, **kwargs):
    return 1.0

# Required compiler_data variable
compiler_data = {
    "target": "fpga",
    "optimization": "area",
    "onnx_patterns": onnx_patterns,
    "cost_function": cost_function,
    "enable_resource_estimation": False,
    "enable_verification": False
}
