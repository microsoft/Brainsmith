"""Dual Pipeline Parity Testing Framework.

This module provides DualPipelineParityTest, a unified testing framework that
combines golden reference validation with hardware parity validation.

Philosophy:
-----------
Run complete pipeline for BOTH manual (FINN) and auto (Brainsmith) implementations:
- Each validates against NumPy golden reference (absolute correctness)
- Hardware specs compared between implementations (migration safety)

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

        def get_kernel_class(self):
            return MyKernel  # For golden reference

Inherited Tests (~20 automatic):
--------------------------------
- Golden reference validation (4 tests)
- Hardware parity validation (12 tests)
- Integration validation (4 tests)
"""

from .base_dual_pipeline_test import DualPipelineParityTest

__all__ = ["DualPipelineParityTest"]
