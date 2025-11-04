"""Test frameworks for kernel testing.

This package provides composition-based test frameworks that eliminate
duplication and complexity from the old inheritance-based approach.

Frameworks:
- SingleKernelTest: Test one kernel vs golden reference
- DualKernelTest: Test manual vs auto parity + both vs golden

Utilities (from tests.common):
- PipelineRunner: Unified ONNX → Hardware pipeline execution
- GoldenValidator: Output validation against golden reference
- Executors: Backend execution (Python, cppsim, rtlsim)

Example (SingleKernelTest):
    from tests.frameworks.single_kernel_test import SingleKernelTest
    from brainsmith.primitives.transforms.infer_kernels import InferKernels

    class TestMyKernel(SingleKernelTest):
        def make_test_model(self):
            # Create ONNX model
            return model, "Add_0"

        def get_kernel_inference_transform(self):
            return InferKernels

        def compute_golden_reference(self, inputs):
            return {"output": inputs["input0"] + inputs["input1"]}

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Example (DualKernelTest):
    from tests.frameworks.dual_kernel_test import DualKernelTest
    from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
    from brainsmith.primitives.transforms.infer_kernels import InferKernels

    class TestAddStreamsParity(DualKernelTest):
        # Same make_test_model, get_num_inputs, get_num_outputs as above

        def get_manual_transform(self):
            return InferAddStreamsLayer  # FINN

        def get_auto_transform(self):
            return InferKernels  # Brainsmith

        def compute_golden_reference(self, inputs):
            return {"output": inputs["input0"] + inputs["input1"]}
"""

from tests.frameworks.kernel_test_base import KernelTestConfig

__all__ = ["KernelTestConfig"]
