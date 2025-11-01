"""Dual Pipeline Parity Testing - New Composition-Based Architecture.

NOTE: DualPipelineParityTest has been replaced by DualKernelTest (tests/frameworks/).
All dual pipeline tests have been migrated to the new composition-based framework.

New Framework Features:
-----------------------
- Composition over inheritance (no diamond inheritance)
- 20 inherited tests (vs 16 in old framework)
- Uses PipelineRunner, GoldenValidator, Executors utilities
- Cleaner, more maintainable architecture

Migration Guide:
----------------
Old:
    from tests.dual_pipeline import DualPipelineParityTest
    class TestMyKernel(DualPipelineParityTest):
        ...

New:
    from tests.frameworks.dual_kernel_test import DualKernelTest
    class TestMyKernel(DualKernelTest):
        ...

Key Changes:
- configure_kernel_node() no longer needs is_manual parameter
- Inherits 20 tests instead of 16 (more coverage)
- No diamond inheritance issues

See Also:
---------
- tests/frameworks/dual_kernel_test.py - New framework
- tests/IMPLEMENTATION_STATUS.md - Migration status and history
- tests/PHASE3_VALIDATION_SUMMARY.md - Migration validation report
"""

# All dual pipeline tests migrated to tests/frameworks/dual_kernel_test.py
__all__ = []
