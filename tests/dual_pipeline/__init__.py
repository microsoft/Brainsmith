"""Dual Pipeline Parity Testing Framework - V2 Modular Architecture.

This module provides DualPipelineParityTest, a convenience wrapper combining:
1. Core parity testing (CoreParityTest) - shapes, widths, datatypes
2. HW estimation parity (HWEstimationParityTest) - resources, cycles
3. Golden reference validation (GoldenReferenceMixin) - execution correctness

Philosophy:
-----------
Run complete pipeline for BOTH manual (FINN) and auto (Brainsmith) implementations:
- Each validates against test-owned golden reference (absolute correctness)
- Hardware specs compared between implementations (migration safety)
- Structural properties validated (shapes, widths, datatypes)

Best of both worlds for compiler teams migrating from FINN to Brainsmith.

Usage:
------
    from tests.dual_pipeline import DualPipelineParityTest

    class TestMyKernelDualParity(DualPipelineParityTest):
        def make_test_model(self):
            return create_onnx_model(), "node_name"

        def get_manual_transform(self):
            return InferMyKernelLayer  # FINN

        def get_auto_transform(self):
            return InferKernelList  # Brainsmith

        def compute_golden_reference(self, inputs):
            # Test-owned golden reference
            return {"output": inputs["input0"] + inputs["input1"]}

        def get_num_inputs(self):
            return 2

        def get_num_outputs(self):
            return 1

Inherited Tests (16 automatic):
--------------------------------
- Core parity (7 tests): shapes, widths, datatypes
- HW estimation (5 tests): resources, cycles
- Golden execution (4 tests): manual/auto × Python/cppsim
"""

from .dual_pipeline_parity_test_v2 import DualPipelineParityTest

__all__ = ["DualPipelineParityTest"]
