"""Test frameworks for kernel testing (v4.0).

This package provides Arete-compliant test frameworks with compositional
configuration, eliminating bloat and complexity theater.

Architecture (v4.0):
- Composition over inheritance
- Config as data (ModelStructure, DesignParameters, PlatformConfig, ValidationConfig)
- Property accessors for flat access to nested structure
- No delegation bloat (config methods called directly)
- Support utilities via composition (not inheritance)

Frameworks:
- SingleKernelTest_v2: Test one kernel implementation vs golden reference
- DualKernelTest_v2: Test FINN vs Brainsmith parity + both vs golden

Configuration (v4.0):
- KernelTestConfig: Compositional configuration with sub-configs
- ModelStructure: What we're testing (operation, shapes, dtypes)
- DesignParameters: How we configure it (PE, SIMD, backend variants)
- PlatformConfig: Where we run it (FPGA part)
- ValidationConfig: How we validate it (tolerances)

Support Utilities (tests.support):
- PipelineRunner: Unified ONNX → Hardware pipeline execution
- GoldenValidator: Output validation against golden reference
- Executors: Backend execution (Python, cppsim, rtlsim)

Example (SingleKernelTest_v2):
    from tests.frameworks.single_kernel_test_v2 import SingleKernelTest
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

    class TestMyKernel(SingleKernelTest):
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

Example (DualKernelTest_v2):
    from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2

    class TestAddStreamsParity(DualKernelTest_v2):
        # Same make_test_model and kernel_test_config as above

        def get_kernel_inference_transform(self):
            # Brainsmith automatic inference
            from brainsmith.primitives.transforms.infer_kernels import InferKernels
            return InferKernels

        def get_manual_backend_variants(self):
            # FINN manual backend selection
            return ("hls",)
"""

from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    DesignParameters,
    PlatformConfig,
    ValidationConfig,
)
from tests.frameworks.kernel_test_base_v2 import KernelTestBase_v2
from tests.frameworks.single_kernel_test_v2 import SingleKernelTest
from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2

__all__ = [
    "KernelTestConfig",
    "ModelStructure",
    "DesignParameters",
    "PlatformConfig",
    "ValidationConfig",
    "KernelTestBase_v2",
    "SingleKernelTest",
    "DualKernelTest_v2",
]
