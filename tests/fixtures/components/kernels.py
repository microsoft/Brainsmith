"""Real test kernel components."""

from brainsmith.registry import kernel


@kernel(name="TestKernel")
class TestKernel:
    """Basic test kernel for integration tests."""

    op_type = "TestOp"
    domain = "test.custom"

    def __init__(self, onnx_node):
        self.onnx_node = onnx_node

    def execute(self, context):
        """Dummy execution for testing."""
        return context


@kernel(name="AnotherTestKernel")
class AnotherTestKernel:
    """Another test kernel for multi-component tests."""

    op_type = "AnotherOp"
    domain = "test.custom"

    def __init__(self, onnx_node):
        self.onnx_node = onnx_node


@kernel(name="TestKernelWithInfer")
class TestKernelWithInfer:
    """Test kernel without infer transform (removed broken reference to test.infer)."""

    op_type = "TestInferOp"
    domain = "test.custom"

    def __init__(self, onnx_node):
        self.onnx_node = onnx_node
