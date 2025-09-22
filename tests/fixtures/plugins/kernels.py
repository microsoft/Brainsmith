"""Test kernels, backends, and kernel inference for unit testing."""

import onnx
from typing import Dict, Any

# Import decorators from brainsmith
from brainsmith.core.plugins import kernel, backend, kernel_inference


# Test Kernel Plugins

@kernel(name="TestKernel")
class TestKernelPlugin:
    """A simple test kernel."""
    
    def __init__(self):
        self.name = "TestKernel"
        self.supported_operations = ["TestOp"]
    
    def verify_configuration(self, config: Dict[str, Any]) -> bool:
        """Verify kernel configuration."""
        return True


@kernel(name="TestKernelWithBackends")
class TestKernelWithBackendsPlugin:
    """Kernel with multiple backend implementations."""
    
    def __init__(self):
        self.name = "TestKernelWithBackends"


# Test Backend Plugins

@backend(name="TestKernel_hls", kernel="TestKernel", language="hls")
class TestKernelHLS:
    """HLS backend for TestKernel."""
    
    def generate(self, config: Dict[str, Any]) -> str:
        """Generate HLS code."""
        return "// HLS implementation"


@backend(name="TestKernel_rtl", kernel="TestKernel", language="rtl") 
class TestKernelRTL:
    """RTL backend for TestKernel."""
    
    def generate(self, config: Dict[str, Any]) -> str:
        """Generate RTL code."""
        return "// RTL implementation"


@backend(name="TestKernelWithBackends_hls", kernel="TestKernelWithBackends", language="hls")
class TestKernelWithBackendsHLS:
    """HLS backend for TestKernelWithBackends."""
    
    def generate(self, config: Dict[str, Any]) -> str:
        """Generate HLS code."""
        return "// HLS implementation for TestKernelWithBackends"


# Test Kernel Inference

@kernel_inference(kernel="TestKernel")
class InferTestKernel:
    """Inference transform for TestKernel."""
    
    def apply(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply inference to convert ops to TestKernel."""
        # Add marker that inference was applied
        if model.graph.node:
            node = model.graph.node[0]
            attr = node.attribute.add()
            attr.name = "kernel_inferred"
            attr.type = onnx.AttributeProto.STRING
            attr.s = b"TestKernel"
        return model