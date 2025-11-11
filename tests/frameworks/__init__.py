"""Test frameworks for kernel testing.

This package provides Arete-compliant test frameworks with compositional
configuration, eliminating bloat and complexity theater.

Architecture:
- Composition over inheritance
- Config as data (ModelStructure, DesignParameters, PlatformConfig, ValidationConfig)
- Property accessors for flat access to nested structure
- No delegation bloat (config methods called directly)
- Support utilities via composition (not inheritance)

Frameworks:
- KernelTest: Test one kernel implementation vs golden reference
- KernelParityTest: Test FINN vs Brainsmith parity + both vs golden

Configuration:
- KernelTestConfig: Compositional configuration with sub-configs
- ModelStructure: What we're testing (operation, shapes, dtypes)
- DesignParameters: How we configure it (PE, SIMD, backend variants)
- PlatformConfig: Where we run it (FPGA part)
- ValidationConfig: How we validate it (tolerances)

Support Utilities (tests.support):
- PipelineRunner: Unified ONNX → Hardware pipeline execution
- GoldenValidator: Output validation against golden reference
- Executors: Backend execution (Python, cppsim, rtlsim)

Example (KernelTest):
    from tests.frameworks.kernel_test import KernelTest
    from tests.frameworks.test_config import (
        KernelTestConfig,
        ModelStructure,
        DesignParameters,
        PlatformConfig,
        ValidationConfig,
    )
    from qonnx.core.datatype import DataType

    # Reusable sub-configs
    ZYNQ_7020 = PlatformConfig(fpgapart="xc7z020clg400-1")
    STANDARD_VALIDATION = ValidationConfig(
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    )

    class TestMyKernel(KernelTest):
        @pytest.fixture(params=[
            KernelTestConfig(
                test_id="add_int8_baseline",
                model=ModelStructure(
                    operation="Add",
                    input_shapes={"input": (1, 64), "param": (1, 64)},
                    input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
                ),
                design=DesignParameters(),  # No parallelization
                platform=ZYNQ_7020,
                validation=STANDARD_VALIDATION,
            ),
        ])
        def kernel_test_config(self, request):
            return request.param

        def make_test_model(self, kernel_test_config):
            # Create ONNX model from config
            # Framework uses property accessors: kernel_test_config.input_shapes
            return model, ["input", "param"]

        def get_kernel_inference_transform(self):
            from brainsmith.primitives.transforms.infer_kernels import InferKernels
            from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
            return lambda: InferKernels([ElementwiseBinaryOp])

Example (KernelParityTest):
    from tests.frameworks.kernel_parity_test import KernelParityTest

    class TestAddStreamsParity(KernelParityTest):
        # Same make_test_model and kernel_test_config as above

        def infer_kernel_reference(self, model, target_node):
            # Reference (FINN) implementation
            from finn.transformation.fpgadataflow.convert_to_hw_layers import InferElementwiseBinaryOperation
            model = model.transform(InferElementwiseBinaryOperation())
            nodes = model.get_nodes_by_op_type("ElementwiseAdd")
            from qonnx.custom_op.registry import getCustomOp
            return getCustomOp(nodes[0]), model

        def get_backend_variants_reference(self):
            # Reference backend variants
            from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import ElementwiseAdd_hls
            return [ElementwiseAdd_hls]

        def get_kernel_inference_transform(self):
            # Primary (Brainsmith) implementation
            from brainsmith.primitives.transforms.infer_kernels import InferKernels
            from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
            return lambda: InferKernels([ElementwiseBinaryOp])
"""

from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.kernel_test import KernelTest
from tests.frameworks.kernel_test_base import KernelTestBase
from tests.frameworks.test_config import (
    DesignParameters,
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
    ValidationConfig,
)

__all__ = [
    "KernelTestConfig",
    "ModelStructure",
    "DesignParameters",
    "PlatformConfig",
    "ValidationConfig",
    "KernelTestBase",
    "KernelTest",
    "KernelParityTest",
]
