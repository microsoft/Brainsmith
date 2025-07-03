"""
Comprehensive Test Plugin Library for Real Component Testing

This module provides a complete set of real plugins for testing the Phase 1
Design Space Constructor without mocks. The plugins are designed to be:
- Minimal but functional
- Deterministic in behavior
- Covering all plugin types and scenarios
- Easy to understand and maintain
"""

from brainsmith.core.plugins.decorators import transform, kernel, backend


# =============================================================================
# TEST KERNELS - 10 kernels covering different operation categories
# =============================================================================

@kernel(name="TestMatMul", op_type="matmul")
class TestMatMul:
    """Test kernel for matrix multiplication operations."""
    def compile(self, node):
        return {"type": "matmul", "node": node, "compiled": True}


@kernel(name="TestLayerNorm", op_type="normalization")
class TestLayerNorm:
    """Test kernel for layer normalization operations."""
    def compile(self, node):
        return {"type": "layernorm", "node": node, "compiled": True}


@kernel(name="TestSoftmax", op_type="activation")
class TestSoftmax:
    """Test kernel for softmax activation operations."""
    def compile(self, node):
        return {"type": "softmax", "node": node, "compiled": True}


@kernel(name="TestAttention", op_type="attention")
class TestAttention:
    """Test kernel for attention mechanisms."""
    def compile(self, node):
        return {"type": "attention", "node": node, "compiled": True}


@kernel(name="TestConv2D", op_type="convolution")
class TestConv2D:
    """Test kernel for 2D convolution operations."""
    def compile(self, node):
        return {"type": "conv2d", "node": node, "compiled": True}


@kernel(name="TestPooling", op_type="pooling")
class TestPooling:
    """Test kernel for pooling operations."""
    def compile(self, node):
        return {"type": "pooling", "node": node, "compiled": True}


@kernel(name="TestReshape", op_type="shape")
class TestReshape:
    """Test kernel for reshape operations."""
    def compile(self, node):
        return {"type": "reshape", "node": node, "compiled": True}


@kernel(name="TestActivation", op_type="activation")
class TestActivation:
    """Test kernel for general activation functions."""
    def compile(self, node):
        return {"type": "activation", "node": node, "compiled": True}


@kernel(name="TestBatchNorm", op_type="normalization")
class TestBatchNorm:
    """Test kernel for batch normalization."""
    def compile(self, node):
        return {"type": "batchnorm", "node": node, "compiled": True}


@kernel(name="TestCustomOp", op_type="custom")
class TestCustomOp:
    """Test kernel for custom operations."""
    def compile(self, node):
        return {"type": "custom", "node": node, "compiled": True}


# =============================================================================
# TEST BACKENDS - 30 backends (3 per kernel: HLS, RTL, DSP variants)
# =============================================================================

# MatMul backends
@backend(name="TestMatMulHLS", kernel="TestMatMul", language="hls", default=True)
class TestMatMulHLS:
    """HLS backend for MatMul operations."""
    def generate(self, kernel_instance):
        return "// TestMatMul HLS implementation"


@backend(name="TestMatMulRTL", kernel="TestMatMul", language="rtl")
class TestMatMulRTL:
    """RTL backend for MatMul operations."""
    def generate(self, kernel_instance):
        return "// TestMatMul RTL implementation"


@backend(name="TestMatMulDSP", kernel="TestMatMul", language="dsp")
class TestMatMulDSP:
    """DSP backend for MatMul operations."""
    def generate(self, kernel_instance):
        return "// TestMatMul DSP implementation"


# LayerNorm backends
@backend(name="TestLayerNormHLS", kernel="TestLayerNorm", language="hls", default=True)
class TestLayerNormHLS:
    """HLS backend for LayerNorm operations."""
    def generate(self, kernel_instance):
        return "// TestLayerNorm HLS implementation"


@backend(name="TestLayerNormRTL", kernel="TestLayerNorm", language="rtl")
class TestLayerNormRTL:
    """RTL backend for LayerNorm operations."""
    def generate(self, kernel_instance):
        return "// TestLayerNorm RTL implementation"


@backend(name="TestLayerNormDSP", kernel="TestLayerNorm", language="dsp")
class TestLayerNormDSP:
    """DSP backend for LayerNorm operations."""
    def generate(self, kernel_instance):
        return "// TestLayerNorm DSP implementation"


# Softmax backends
@backend(name="TestSoftmaxHLS", kernel="TestSoftmax", language="hls", default=True)
class TestSoftmaxHLS:
    """HLS backend for Softmax operations."""
    def generate(self, kernel_instance):
        return "// TestSoftmax HLS implementation"


@backend(name="TestSoftmaxRTL", kernel="TestSoftmax", language="rtl")
class TestSoftmaxRTL:
    """RTL backend for Softmax operations."""
    def generate(self, kernel_instance):
        return "// TestSoftmax RTL implementation"


@backend(name="TestSoftmaxDSP", kernel="TestSoftmax", language="dsp")
class TestSoftmaxDSP:
    """DSP backend for Softmax operations."""
    def generate(self, kernel_instance):
        return "// TestSoftmax DSP implementation"


# Attention backends
@backend(name="TestAttentionHLS", kernel="TestAttention", language="hls", default=True)
class TestAttentionHLS:
    """HLS backend for Attention operations."""
    def generate(self, kernel_instance):
        return "// TestAttention HLS implementation"


@backend(name="TestAttentionRTL", kernel="TestAttention", language="rtl")
class TestAttentionRTL:
    """RTL backend for Attention operations."""
    def generate(self, kernel_instance):
        return "// TestAttention RTL implementation"


@backend(name="TestAttentionCUDA", kernel="TestAttention", language="cuda")
class TestAttentionCUDA:
    """CUDA backend for Attention operations."""
    def generate(self, kernel_instance):
        return "// TestAttention CUDA implementation"


# Conv2D backends
@backend(name="TestConv2DHLS", kernel="TestConv2D", language="hls", default=True)
class TestConv2DHLS:
    """HLS backend for Conv2D operations."""
    def generate(self, kernel_instance):
        return "// TestConv2D HLS implementation"


@backend(name="TestConv2DRTL", kernel="TestConv2D", language="rtl")
class TestConv2DRTL:
    """RTL backend for Conv2D operations."""
    def generate(self, kernel_instance):
        return "// TestConv2D RTL implementation"


@backend(name="TestConv2DDSP", kernel="TestConv2D", language="dsp")
class TestConv2DDSP:
    """DSP backend for Conv2D operations."""
    def generate(self, kernel_instance):
        return "// TestConv2D DSP implementation"


# Pooling backends
@backend(name="TestPoolingHLS", kernel="TestPooling", language="hls", default=True)
class TestPoolingHLS:
    """HLS backend for Pooling operations."""
    def generate(self, kernel_instance):
        return "// TestPooling HLS implementation"


@backend(name="TestPoolingRTL", kernel="TestPooling", language="rtl")
class TestPoolingRTL:
    """RTL backend for Pooling operations."""
    def generate(self, kernel_instance):
        return "// TestPooling RTL implementation"


@backend(name="TestPoolingDSP", kernel="TestPooling", language="dsp")
class TestPoolingDSP:
    """DSP backend for Pooling operations."""
    def generate(self, kernel_instance):
        return "// TestPooling DSP implementation"


# Reshape backends
@backend(name="TestReshapeHLS", kernel="TestReshape", language="hls", default=True)
class TestReshapeHLS:
    """HLS backend for Reshape operations."""
    def generate(self, kernel_instance):
        return "// TestReshape HLS implementation"


@backend(name="TestReshapeRTL", kernel="TestReshape", language="rtl")
class TestReshapeRTL:
    """RTL backend for Reshape operations."""
    def generate(self, kernel_instance):
        return "// TestReshape RTL implementation"


@backend(name="TestReshapeDSP", kernel="TestReshape", language="dsp")
class TestReshapeDSP:
    """DSP backend for Reshape operations."""
    def generate(self, kernel_instance):
        return "// TestReshape DSP implementation"


# Activation backends
@backend(name="TestActivationHLS", kernel="TestActivation", language="hls", default=True)
class TestActivationHLS:
    """HLS backend for Activation operations."""
    def generate(self, kernel_instance):
        return "// TestActivation HLS implementation"


@backend(name="TestActivationRTL", kernel="TestActivation", language="rtl")
class TestActivationRTL:
    """RTL backend for Activation operations."""
    def generate(self, kernel_instance):
        return "// TestActivation RTL implementation"


@backend(name="TestActivationDSP", kernel="TestActivation", language="dsp")
class TestActivationDSP:
    """DSP backend for Activation operations."""
    def generate(self, kernel_instance):
        return "// TestActivation DSP implementation"


# BatchNorm backends
@backend(name="TestBatchNormHLS", kernel="TestBatchNorm", language="hls", default=True)
class TestBatchNormHLS:
    """HLS backend for BatchNorm operations."""
    def generate(self, kernel_instance):
        return "// TestBatchNorm HLS implementation"


@backend(name="TestBatchNormRTL", kernel="TestBatchNorm", language="rtl")
class TestBatchNormRTL:
    """RTL backend for BatchNorm operations."""
    def generate(self, kernel_instance):
        return "// TestBatchNorm RTL implementation"


@backend(name="TestBatchNormDSP", kernel="TestBatchNorm", language="dsp")
class TestBatchNormDSP:
    """DSP backend for BatchNorm operations."""
    def generate(self, kernel_instance):
        return "// TestBatchNorm DSP implementation"


# CustomOp backends
@backend(name="TestCustomOpHLS", kernel="TestCustomOp", language="hls", default=True)
class TestCustomOpHLS:
    """HLS backend for CustomOp operations."""
    def generate(self, kernel_instance):
        return "// TestCustomOp HLS implementation"


@backend(name="TestCustomOpRTL", kernel="TestCustomOp", language="rtl")
class TestCustomOpRTL:
    """RTL backend for CustomOp operations."""
    def generate(self, kernel_instance):
        return "// TestCustomOp RTL implementation"


@backend(name="TestCustomOpDSP", kernel="TestCustomOp", language="dsp")
class TestCustomOpDSP:
    """DSP backend for CustomOp operations."""
    def generate(self, kernel_instance):
        return "// TestCustomOp DSP implementation"


# =============================================================================
# TEST TRANSFORMS - 20 transforms across all 6 stages
# =============================================================================

# Pre-processing stage (3 transforms)
@transform(name="TestQuantizeModel", stage="pre_proc", framework="brainsmith")
class TestQuantizeModel:
    """Test transform for model quantization."""
    def apply(self, model):
        return model, True  # Changed model


@transform(name="TestValidateModel", stage="pre_proc", framework="brainsmith")
class TestValidateModel:
    """Test transform for model validation."""
    def apply(self, model):
        return model, False  # No change


@transform(name="TestOptionalPreproc", stage="pre_proc", framework="brainsmith")
class TestOptionalPreproc:
    """Optional test transform for preprocessing."""
    def apply(self, model):
        return model, False


# Cleanup stage (4 transforms)
@transform(name="TestRemoveIdentityOps", stage="cleanup", framework="brainsmith")
class TestRemoveIdentityOps:
    """Test transform for removing identity operations."""
    def apply(self, model):
        return model, True


@transform(name="TestRemoveUnusedTensors", stage="cleanup", framework="brainsmith")
class TestRemoveUnusedTensors:
    """Test transform for removing unused tensors."""
    def apply(self, model):
        return model, True


@transform(name="TestGiveReadableTensorNames", stage="cleanup", framework="brainsmith")
class TestGiveReadableTensorNames:
    """Test transform for tensor naming."""
    def apply(self, model):
        return model, True


@transform(name="TestOptionalCleanup", stage="cleanup", framework="brainsmith")
class TestOptionalCleanup:
    """Optional test transform for cleanup."""
    def apply(self, model):
        return model, False


# Topology optimization stage (4 transforms)
@transform(name="TestFoldConstants", stage="topology_opt", framework="brainsmith")
class TestFoldConstants:
    """Test transform for constant folding."""
    def apply(self, model):
        return model, True


@transform(name="TestConvertQONNXtoFINN", stage="topology_opt", framework="qonnx")
class TestConvertQONNXtoFINN:
    """Test transform for QONNX to FINN conversion."""
    def apply(self, model):
        return model, True


@transform(name="TestInferDataLayouts", stage="topology_opt", framework="finn")
class TestInferDataLayouts:
    """Test transform for data layout inference."""
    def apply(self, model):
        return model, True


@transform(name="TestOptionalTopology", stage="topology_opt", framework="brainsmith")
class TestOptionalTopology:
    """Optional test transform for topology optimization."""
    def apply(self, model):
        return model, False


# Kernel optimization stage (3 transforms)
@transform(name="TestOptimizeKernelConfig", stage="kernel_opt", framework="brainsmith")
class TestOptimizeKernelConfig:
    """Test transform for kernel configuration optimization."""
    def apply(self, model):
        return model, True


@transform(name="TestTuneKernelParams", stage="kernel_opt", framework="brainsmith")
class TestTuneKernelParams:
    """Test transform for kernel parameter tuning."""
    def apply(self, model):
        return model, True


@transform(name="TestOptionalKernel", stage="kernel_opt", framework="brainsmith")
class TestOptionalKernel:
    """Optional test transform for kernel optimization."""
    def apply(self, model):
        return model, False


# Dataflow optimization stage (3 transforms)
@transform(name="TestOptimizeDataflow", stage="dataflow_opt", framework="brainsmith")
class TestOptimizeDataflow:
    """Test transform for dataflow optimization."""
    def apply(self, model):
        return model, True


@transform(name="TestBalancePipeline", stage="dataflow_opt", framework="brainsmith")
class TestBalancePipeline:
    """Test transform for pipeline balancing."""
    def apply(self, model):
        return model, True


@transform(name="TestOptionalDataflow", stage="dataflow_opt", framework="brainsmith")
class TestOptionalDataflow:
    """Optional test transform for dataflow optimization."""
    def apply(self, model):
        return model, False


# Post-processing stage (3 transforms)
@transform(name="TestFinalizeModel", stage="post_proc", framework="brainsmith")
class TestFinalizeModel:
    """Test transform for model finalization."""
    def apply(self, model):
        return model, True


@transform(name="TestValidateOutput", stage="post_proc", framework="brainsmith")
class TestValidateOutput:
    """Test transform for output validation."""
    def apply(self, model):
        return model, False


@transform(name="TestOptionalPostproc", stage="post_proc", framework="brainsmith")
class TestOptionalPostproc:
    """Optional test transform for post-processing."""
    def apply(self, model):
        return model, False


# =============================================================================
# ERROR SCENARIO PLUGINS - For testing failure cases
# =============================================================================

@kernel(name="ErrorKernel", op_type="error")
class ErrorKernel:
    """Kernel that raises an error during compilation."""
    def compile(self, node):
        raise RuntimeError("ErrorKernel always fails")


@kernel(name="TestNoBackendsKernel", op_type="test_no_backends")
class TestNoBackendsKernel:
    """Kernel with no backends - for testing error scenarios."""
    def compile(self, node):
        return {"type": "no_backends", "node": node}


@backend(name="ErrorBackend", kernel="TestMatMul", language="error")
class ErrorBackend:
    """Backend that raises an error during generation."""
    def generate(self, kernel_instance):
        raise RuntimeError("ErrorBackend always fails")


@transform(name="ErrorTransform", stage="cleanup", framework="brainsmith")
class ErrorTransform:
    """Transform that raises an error during application."""
    def apply(self, model):
        raise RuntimeError("ErrorTransform always fails")


# =============================================================================
# PLUGIN CATEGORIES FOR ORGANIZED TESTING
# =============================================================================

# Collections for easy access in tests
TEST_KERNELS = [
    "TestMatMul", "TestLayerNorm", "TestSoftmax", "TestAttention", "TestConv2D",
    "TestPooling", "TestReshape", "TestActivation", "TestBatchNorm", "TestCustomOp"
]

TEST_BACKENDS = [
    # MatMul
    "TestMatMulHLS", "TestMatMulRTL", "TestMatMulDSP",
    # LayerNorm
    "TestLayerNormHLS", "TestLayerNormRTL", "TestLayerNormDSP",
    # Softmax
    "TestSoftmaxHLS", "TestSoftmaxRTL", "TestSoftmaxDSP",
    # Attention
    "TestAttentionHLS", "TestAttentionRTL", "TestAttentionCUDA",
    # Conv2D
    "TestConv2DHLS", "TestConv2DRTL", "TestConv2DDSP",
    # Pooling
    "TestPoolingHLS", "TestPoolingRTL", "TestPoolingDSP",
    # Reshape
    "TestReshapeHLS", "TestReshapeRTL", "TestReshapeDSP",
    # Activation
    "TestActivationHLS", "TestActivationRTL", "TestActivationDSP",
    # BatchNorm
    "TestBatchNormHLS", "TestBatchNormRTL", "TestBatchNormDSP",
    # CustomOp
    "TestCustomOpHLS", "TestCustomOpRTL", "TestCustomOpDSP"
]

TEST_TRANSFORMS = [
    # Pre-processing
    "TestQuantizeModel", "TestValidateModel", "TestOptionalPreproc",
    # Cleanup
    "TestRemoveIdentityOps", "TestRemoveUnusedTensors", "TestGiveReadableTensorNames", "TestOptionalCleanup",
    # Topology optimization
    "TestFoldConstants", "TestConvertQONNXtoFINN", "TestInferDataLayouts", "TestOptionalTopology",
    # Kernel optimization
    "TestOptimizeKernelConfig", "TestTuneKernelParams", "TestOptionalKernel",
    # Dataflow optimization
    "TestOptimizeDataflow", "TestBalancePipeline", "TestOptionalDataflow",
    # Post-processing
    "TestFinalizeModel", "TestValidateOutput", "TestOptionalPostproc"
]

ERROR_PLUGINS = ["ErrorKernel", "TestNoBackendsKernel", "ErrorBackend", "ErrorTransform"]

# Backend mappings for testing
KERNEL_BACKEND_MAP = {
    "TestMatMul": ["TestMatMulHLS", "TestMatMulRTL", "TestMatMulDSP"],
    "TestLayerNorm": ["TestLayerNormHLS", "TestLayerNormRTL", "TestLayerNormDSP"],
    "TestSoftmax": ["TestSoftmaxHLS", "TestSoftmaxRTL", "TestSoftmaxDSP"],
    "TestAttention": ["TestAttentionHLS", "TestAttentionRTL", "TestAttentionCUDA"],
    "TestConv2D": ["TestConv2DHLS", "TestConv2DRTL", "TestConv2DDSP"],
    "TestPooling": ["TestPoolingHLS", "TestPoolingRTL", "TestPoolingDSP"],
    "TestReshape": ["TestReshapeHLS", "TestReshapeRTL", "TestReshapeDSP"],
    "TestActivation": ["TestActivationHLS", "TestActivationRTL", "TestActivationDSP"],
    "TestBatchNorm": ["TestBatchNormHLS", "TestBatchNormRTL", "TestBatchNormDSP"],
    "TestCustomOp": ["TestCustomOpHLS", "TestCustomOpRTL", "TestCustomOpDSP"],
}

STAGE_TRANSFORM_MAP = {
    "pre_proc": ["TestQuantizeModel", "TestValidateModel", "TestOptionalPreproc"],
    "cleanup": ["TestRemoveIdentityOps", "TestRemoveUnusedTensors", "TestGiveReadableTensorNames", "TestOptionalCleanup"],
    "topology_opt": ["TestFoldConstants", "TestConvertQONNXtoFINN", "TestInferDataLayouts", "TestOptionalTopology"],
    "kernel_opt": ["TestOptimizeKernelConfig", "TestTuneKernelParams", "TestOptionalKernel"],
    "dataflow_opt": ["TestOptimizeDataflow", "TestBalancePipeline", "TestOptionalDataflow"],
    "post_proc": ["TestFinalizeModel", "TestValidateOutput", "TestOptionalPostproc"],
}


def get_test_plugins_info():
    """Get information about all test plugins."""
    return {
        "kernels": len(TEST_KERNELS),
        "backends": len(TEST_BACKENDS),
        "transforms": len(TEST_TRANSFORMS),
        "error_plugins": len(ERROR_PLUGINS),
        "total": len(TEST_KERNELS) + len(TEST_BACKENDS) + len(TEST_TRANSFORMS) + len(ERROR_PLUGINS)
    }