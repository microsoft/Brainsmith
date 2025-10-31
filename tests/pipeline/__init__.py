"""Pipeline integration tests for complete ONNX → Hardware → Execution flows.

This module provides end-to-end integration testing for the complete transformation
pipeline from standard ONNX nodes to hardware-accelerated kernels, validating
correctness against golden reference implementations.

Key Features:
- Pipeline validation (ONNX → Shapes → Datatypes → Kernel → Backend)
- Golden reference validation (NumPy/PyTorch ground truth)
- Multi-backend testing (Python, HLS cppsim, RTL rtlsim)
- Property-based testing (parametric sweep)
- Cross-backend parity validation

Compared to Parity Tests (tests/parity/):
- Pipeline tests validate against golden reference (not manual vs auto)
- Pipeline tests validate complete pipeline (not just final node)
- Pipeline tests are kernel-focused (one test class per kernel)
- Parity tests compare two implementations, pipeline tests validate correctness

Compared to DSE Integration Tests (tests/integration/):
- Pipeline tests validate kernel correctness (not DSE framework)
- Pipeline tests focus on ONNX → Hardware transformation
- DSE tests focus on design space exploration logic

Usage:
    # Run all fast pipeline tests
    pytest tests/pipeline/ -v

    # Run slow tests (cppsim, rtlsim)
    pytest tests/pipeline/ -v --run-slow

    # Run specific kernel
    pytest tests/pipeline/test_addstreams_integration.py -v

See Also:
    tests/pipeline/README.md - Complete usage guide
    tests/frameworks/single_kernel_test.py - New composition-based framework
    tests/IMPLEMENTATION_STATUS.md - Migration status and history
"""

# NOTE: IntegratedPipelineTest was replaced by SingleKernelTest (tests/frameworks/)
# All pipeline tests have been migrated to the new composition-based framework.
# See tests/IMPLEMENTATION_STATUS.md for migration details.

__all__ = []
