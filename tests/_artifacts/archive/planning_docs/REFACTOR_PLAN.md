# Testing Infrastructure Refactor - Implementation Plan

**Goal:** Transform testing infrastructure from inheritance-driven complexity to composition-driven clarity.

**Estimated Reduction:** 37% fewer lines, 75% less duplication

**Timeline:** 4 phases, ~2-3 weeks

---

## Phase 1: Extract Reusable Components (Week 1)

**Goal:** Create new components without breaking existing tests

**Risk Level:** LOW - No changes to existing code

### Task 1.1: Create `tests/common/pipeline.py`

**File:** `tests/common/pipeline.py` (NEW)

**Content:**
```python
"""Unified pipeline execution for test frameworks."""

from typing import Callable, Optional, Tuple, Type
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import getHWCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from brainsmith.dataflow.kernel_op import KernelOp


class PipelineRunner:
    """Unified pipeline execution for manual/auto/custom kernel inference.

    Replaces duplicated run_manual_pipeline/run_auto_pipeline methods across
    CoreParityTest, HWEstimationParityTest, and IntegratedPipelineTest.

    Example:
        >>> runner = PipelineRunner()
        >>> op, model = runner.run(
        ...     model_factory=lambda: (create_model(), "Add_0"),
        ...     transform=InferKernelList,
        ...     configurator=lambda op, model: op.set_nodeattr("PE", 8)
        ... )
    """

    def run(
        self,
        model_factory: Callable[[], Tuple[ModelWrapper, str]],
        transform: Type[Transformation],
        configurator: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete ONNX → Hardware kernel pipeline.

        Pipeline stages:
        1. Create ONNX model (via model_factory)
        2. InferShapes
        3. InferDataTypes
        4. Kernel-specific inference (transform)
        5. Optional configuration (configurator)
        6. KernelOp initialization if applicable

        Args:
            model_factory: Function returning (ModelWrapper, node_name)
            transform: Transformation class (InferAddStreamsLayer, InferKernelList, etc.)
            configurator: Optional function to configure op after inference
                         Example: lambda op, model: op.set_nodeattr("PE", 8)

        Returns:
            (op, model): Hardware operator and transformed model

        Raises:
            RuntimeError: If hardware node not created
        """
        # Create model
        model, expected_node_name = model_factory()

        # Standard ONNX preprocessing
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Kernel-specific inference
        model = model.transform(transform())

        # Find hardware node
        hw_node = model.get_node_from_name(expected_node_name)
        if hw_node is None:
            # Transform may have renamed node, try first node
            if len(model.graph.node) > 0:
                hw_node = model.graph.node[0]
            else:
                raise RuntimeError(
                    f"Kernel inference failed to create hardware node. "
                    f"Transform: {transform.__name__}"
                )

        # Get op instance
        op = getHWCustomOp(hw_node, model)

        # Apply user configuration
        if configurator:
            configurator(op, model)

        # Initialize KernelOp design space if applicable
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        return op, model
```

**Validation:**
- Create simple test: `tests/common/test_pipeline.py`
- Verify PipelineRunner can run AddStreams inference
- Ensure no imports break

---

### Task 1.2: Create `tests/common/validator.py`

**File:** `tests/common/validator.py` (NEW)

**Content:**
```python
"""Unified golden reference validation."""

from typing import Dict
import numpy as np
from tests.parity.assertions import assert_arrays_close


class GoldenValidator:
    """Validates outputs against golden reference.

    Replaces:
    - IntegratedPipelineTest.validate_against_golden()
    - GoldenReferenceMixin.validate_against_golden()

    Handles ONNX tensor name vs golden standard name mismatches
    by comparing outputs by index position.

    Example:
        >>> validator = GoldenValidator()
        >>> validator.validate(
        ...     actual={"outp": np.array([1.0, 2.0])},
        ...     golden={"output": np.array([1.0, 2.0])},
        ...     backend_name="Python execution",
        ...     rtol=1e-7, atol=1e-9
        ... )
    """

    def validate(
        self,
        actual: Dict[str, np.ndarray],
        golden: Dict[str, np.ndarray],
        backend_name: str,
        rtol: float = 1e-5,
        atol: float = 1e-6
    ) -> None:
        """Validate actual outputs match golden reference.

        Compares by index position to handle name mismatches:
        - ONNX may use: {"inp1": ..., "outp": ...}
        - Golden uses: {"input0": ..., "output": ...}

        Args:
            actual: Outputs from backend execution
            golden: Expected outputs from golden reference
            backend_name: Backend identifier for error messages
                         Examples: "Python execution", "HLS cppsim", "RTL rtlsim"
            rtol: Relative tolerance for np.allclose()
            atol: Absolute tolerance for np.allclose()

        Raises:
            AssertionError: If output count or values don't match
        """
        # Convert to lists for index-based comparison
        actual_list = list(actual.items())
        golden_list = list(golden.items())

        # Validate output count
        if len(actual_list) != len(golden_list):
            raise AssertionError(
                f"{backend_name} output count mismatch.\n"
                f"Expected: {len(golden_list)} outputs {list(golden.keys())}\n"
                f"Actual: {len(actual_list)} outputs {list(actual.keys())}"
            )

        # Compare each output by position
        for i, ((actual_name, actual_array), (golden_name, golden_array)) in enumerate(
            zip(actual_list, golden_list)
        ):
            assert_arrays_close(
                actual_array,
                golden_array,
                f"{backend_name} output {i} ({actual_name} vs golden {golden_name})",
                rtol=rtol,
                atol=atol,
            )
```

**Validation:**
- Create test: `tests/common/test_validator.py`
- Test name mismatch handling
- Test multi-output validation

---

### Task 1.3: Refactor `tests/parity/executors.py`

**File:** `tests/parity/executors.py` (MODIFY)

**Changes:**

1. **Extract Executor Protocol:**
```python
"""Backend execution protocol and implementations.

Refactored to use Protocol pattern with single-responsibility executors.
Comparison logic moved to test frameworks where it belongs.
"""

from typing import Protocol, Dict
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Executor(Protocol):
    """Protocol for backend execution (Python/cppsim/rtlsim).

    Executors have ONE responsibility: execute backend and return outputs.
    They do NOT compare outputs - that belongs in test frameworks.
    """

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute backend and return outputs.

        Args:
            op: Hardware operator to execute
            model: Model containing the operator
            inputs: Input tensors (dict mapping names → arrays)

        Returns:
            Dict mapping output names → output arrays

        Raises:
            Exception: If execution fails
        """
        ...
```

2. **Simplify PythonExecutor:**
```python
class PythonExecutor:
    """Execute kernel via op.execute_node() (Python simulation).

    This is the fastest execution mode but least representative of hardware.
    Use for quick functional validation.
    """

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute via Python execute_node()."""
        context = dict(inputs)
        op.execute_node(context, model.graph)

        # Extract outputs
        outputs = {}
        for output_name in op.onnx_node.output:
            if output_name and output_name in context:
                outputs[output_name] = context[output_name]

        return outputs
```

3. **Simplify CppSimExecutor:**
```python
class CppSimExecutor:
    """Execute kernel via HLS C++ simulation (cppsim).

    Generates C++ code, compiles it, and runs native simulation.
    Validates code generation and HLS behavior.

    Requires: VITIS_PATH environment variable
    """

    def __init__(self, cleanup: bool = True):
        """Initialize executor.

        Args:
            cleanup: If True, delete temp directories after execution
        """
        self.cleanup = cleanup

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute via cppsim."""
        # Move current _prepare_and_execute logic here
        # Remove comparison logic
        # Return only outputs
        ...
```

4. **Simplify RTLSimExecutor:**
```python
class RTLSimExecutor:
    """Execute kernel via RTL simulation (rtlsim).

    For HLS backends: Synthesizes HLS → RTL using Vitis HLS first
    For RTL backends: Uses generated HDL directly

    Requires: Vivado installation with XSim
    """

    def __init__(self, cleanup: bool = True):
        """Initialize executor.

        Args:
            cleanup: If True, delete temp directories after execution
        """
        self.cleanup = cleanup

    def execute(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute via rtlsim."""
        # Move current _prepare_and_execute logic here
        # Remove comparison logic
        # Return only outputs
        ...
```

5. **Delete BackendExecutor.execute_and_compare():**
   - Remove entire BackendExecutor class
   - Comparison logic moves to test frameworks

**Validation:**
- Update existing tests to use new Executor API
- Run `pytest tests/parity/ -k executor` to verify

---

### Task 1.4: Validate Phase 1

**Success Criteria:**
- [ ] All existing tests pass unchanged
- [ ] New components importable and functional
- [ ] No performance regression
- [ ] Code coverage maintained or improved

**Commands:**
```bash
# Run full test suite
pytest tests/

# Verify new components
pytest tests/common/test_pipeline.py
pytest tests/common/test_validator.py

# Check imports
python -c "from tests.common.pipeline import PipelineRunner; print('OK')"
python -c "from tests.common.validator import GoldenValidator; print('OK')"
python -c "from tests.parity.executors import PythonExecutor, CppSimExecutor; print('OK')"
```

**Rollback:** Delete new files if validation fails

---

## Phase 2: Create New Test Frameworks (Week 2)

**Goal:** Implement composition-based test frameworks

**Risk Level:** MEDIUM - New code runs in parallel with old

### Task 2.1: Create `tests/frameworks/kernel_test_base.py`

**File:** `tests/frameworks/kernel_test_base.py` (NEW)

**Content:**
```python
"""Minimal shared interface for kernel test frameworks.

This module provides ONLY configuration hooks that both SingleKernelTest
and DualKernelTest need. No pipeline logic, no validation logic.

Design: Composition over inheritance
- Test frameworks compose PipelineRunner, GoldenValidator, Executors
- This base provides only abstract configuration methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class KernelTestConfig(ABC):
    """Minimal configuration interface for kernel tests.

    Provides ONLY abstract methods for test-specific configuration.
    Does NOT provide pipeline/validation/execution logic.
    """

    # ========================================================================
    # Model Creation - Required
    # ========================================================================

    @abstractmethod
    def make_test_model(self) -> Tuple[ModelWrapper, str]:
        """Create standard ONNX model for testing.

        Returns:
            (model, node_name): ModelWrapper and name of test node

        Example:
            def make_test_model(self):
                inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
                out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
                node = helper.make_node("Add", ["input", "bias"], ["output"])
                graph = helper.make_graph([node], "test_add", [inp], [out])
                return ModelWrapper(helper.make_model(graph)), "Add_0"
        """
        pass

    # ========================================================================
    # I/O Configuration - Required
    # ========================================================================

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of input tensors.

        Returns:
            Number of inputs (typically 1 or 2)
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of output tensors.

        Returns:
            Number of outputs (typically 1)
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def configure_kernel_node(
        self,
        op: HWCustomOp,
        model: ModelWrapper
    ) -> None:
        """Configure kernel node after inference (optional).

        Override to set non-default parameters (PE, SIMD, etc.).

        IMPORTANT: If you change dimension parameters on a KernelOp,
        call op._ensure_ready(model) afterwards.

        Args:
            op: Hardware operator instance
            model: Model containing the operator

        Example:
            def configure_kernel_node(self, op, model):
                from brainsmith.dataflow.kernel_op import KernelOp

                op.set_nodeattr("PE", 8)
                op.set_nodeattr("SIMD", 16)

                if isinstance(op, KernelOp):
                    op._ensure_ready(model)
        """
        pass

    def get_tolerance_python(self) -> Dict[str, float]:
        """Tolerance for Python execution vs golden.

        Returns:
            Dict with 'rtol' and 'atol' keys
        """
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self) -> Dict[str, float]:
        """Tolerance for C++ simulation vs golden.

        Returns:
            Dict with 'rtol' and 'atol' keys
        """
        return {"rtol": 1e-5, "atol": 1e-6}

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Tolerance for RTL simulation vs golden.

        Returns:
            Dict with 'rtol' and 'atol' keys
        """
        return self.get_tolerance_cppsim()  # Same as cppsim by default
```

**Validation:**
- Can be imported
- Abstract methods enforced

---

### Task 2.2: Create `tests/frameworks/single_kernel_test.py`

**File:** `tests/frameworks/single_kernel_test.py` (NEW)

**Content:** See detailed implementation in separate file

**Key Features:**
- Inherits from KernelTestConfig
- Composes PipelineRunner, GoldenValidator, PythonExecutor, CppSimExecutor
- 6 tests: pipeline creation, shapes, datatypes, Python golden, cppsim golden, rtlsim golden
- Abstract method: `compute_golden_reference(inputs) -> outputs`
- Abstract method: `get_kernel_inference_transform() -> Type[Transformation]`

**Validation:**
- Create example test: `tests/frameworks/test_single_kernel_example.py`
- Verify 6 tests discoverable

---

### Task 2.3: Create `tests/frameworks/dual_kernel_test.py`

**File:** `tests/frameworks/dual_kernel_test.py` (NEW)

**Content:** See detailed implementation in separate file

**Key Features:**
- Inherits from KernelTestConfig
- Composes PipelineRunner, GoldenValidator, all Executors
- 20 tests: 7 structural parity + 5 HW estimation + 8 golden execution
- Abstract methods: `get_manual_transform()`, `get_auto_transform()`, `compute_golden_reference()`
- Single `configure_kernel_node()` with consistent signature

**Validation:**
- Create example test: `tests/frameworks/test_dual_kernel_example.py`
- Verify 20 tests discoverable

---

### Task 2.4: Validate Phase 2

**Success Criteria:**
- [ ] New frameworks pass example tests
- [ ] No interference with existing tests
- [ ] Documentation clear
- [ ] Performance acceptable

**Commands:**
```bash
# Test new frameworks
pytest tests/frameworks/test_single_kernel_example.py -v
pytest tests/frameworks/test_dual_kernel_example.py -v

# Ensure old tests still pass
pytest tests/pipeline/
pytest tests/parity/
pytest tests/dual_pipeline/

# Coverage check
pytest --cov=tests/frameworks tests/frameworks/
```

**Rollback:** Keep new frameworks but don't migrate if issues found

---

## Phase 3: Migrate Tests (Week 2-3)

**Goal:** Port existing tests to new frameworks

**Risk Level:** MEDIUM - Dual execution to catch regressions

### Task 3.1: Port AddStreams to SingleKernelTest

**File:** `tests/pipeline/test_addstreams_single.py` (NEW)

**Content:**
```python
"""AddStreams integration test using SingleKernelTest framework."""

from tests.frameworks.single_kernel_test import SingleKernelTest
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
import numpy as np


class TestAddStreamsSingle(SingleKernelTest):
    """Test AddStreams kernel vs golden reference."""

    def make_test_model(self):
        # Use existing helper from test_addstreams_integration.py
        ...

    def get_kernel_inference_transform(self):
        return InferKernelList

    def compute_golden_reference(self, inputs):
        return {"output": inputs["input0"] + inputs["input1"]}

    def get_num_inputs(self):
        return 2

    def get_num_outputs(self):
        return 1
```

**Validation:**
```bash
# Run new test
pytest tests/pipeline/test_addstreams_single.py -v

# Run old test
pytest tests/pipeline/test_addstreams_integration.py -v

# Compare results - should be identical
pytest tests/pipeline/test_addstreams_single.py::test_python_execution_vs_golden -v
pytest tests/pipeline/test_addstreams_integration.py::test_python_execution_vs_golden -v
```

---

### Task 3.2: Port AddStreams to DualKernelTest

**File:** `tests/dual_pipeline/test_addstreams_dual.py` (NEW)

**Content:**
```python
"""AddStreams parity test using DualKernelTest framework."""

from tests.frameworks.dual_kernel_test import DualKernelTest
from finn.transformation.fpgadataflow.infer_addstreams_layer import InferAddStreamsLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList


class TestAddStreamsDual(DualKernelTest):
    """Test AddStreams manual vs auto parity."""

    def make_test_model(self):
        # Same as single kernel test
        ...

    def get_manual_transform(self):
        return InferAddStreamsLayer

    def get_auto_transform(self):
        return InferKernelList

    def compute_golden_reference(self, inputs):
        return {"output": inputs["input0"] + inputs["input1"]}

    def get_num_inputs(self):
        return 2

    def get_num_outputs(self):
        return 1
```

**Validation:**
```bash
# Run new test - should have 20 test methods
pytest tests/dual_pipeline/test_addstreams_dual.py -v --collect-only

# Run old test
pytest tests/dual_pipeline/test_addstreams_v2.py -v --collect-only

# Spot-check critical tests
pytest tests/dual_pipeline/test_addstreams_dual.py::test_shapes_parity -v
pytest tests/dual_pipeline/test_addstreams_dual.py::test_both_match_golden_python -v
```

---

### Task 3.3: Create Migration Validation Script

**File:** `tests/validate_migration.py` (NEW)

**Purpose:** Automated validation that new tests produce identical results to old tests

**Content:**
```python
"""Validate that migrated tests produce identical results to original tests.

Usage:
    python tests/validate_migration.py AddStreams
"""

import subprocess
import sys


def compare_test_outputs(old_test_path, new_test_path, test_method):
    """Run both tests and compare outputs."""
    # Run old test
    old_result = subprocess.run(
        ["pytest", old_test_path + "::" + test_method, "-v"],
        capture_output=True
    )

    # Run new test
    new_result = subprocess.run(
        ["pytest", new_test_path + "::" + test_method, "-v"],
        capture_output=True
    )

    # Compare exit codes
    if old_result.returncode != new_result.returncode:
        print(f"❌ {test_method}: Different exit codes")
        return False

    print(f"✅ {test_method}: Pass")
    return True


def validate_addstreams():
    """Validate AddStreams migration."""
    tests_to_compare = [
        ("test_python_execution_vs_golden",
         "tests/pipeline/test_addstreams_integration.py",
         "tests/pipeline/test_addstreams_single.py"),
        ("test_shapes_parity",
         "tests/dual_pipeline/test_addstreams_v2.py",
         "tests/dual_pipeline/test_addstreams_dual.py"),
    ]

    all_passed = True
    for test_method, old_path, new_path in tests_to_compare:
        if not compare_test_outputs(old_path, new_path, test_method):
            all_passed = False

    return all_passed


if __name__ == "__main__":
    kernel_name = sys.argv[1] if len(sys.argv) > 1 else "AddStreams"

    if kernel_name == "AddStreams":
        success = validate_addstreams()
    else:
        print(f"Unknown kernel: {kernel_name}")
        sys.exit(1)

    sys.exit(0 if success else 1)
```

---

### Task 3.4: Port Remaining Kernels

**Kernels to Migrate (Priority Order):**
1. ✅ AddStreams (done in 3.1-3.2)
2. Thresholding
3. ElementwiseBinary
4. DuplicateStreams
5. VVAU
6. Shuffle
7. LayerNorm
8. Softmax
9. Channelwise
10. Crop

**Process for Each Kernel:**
1. Create `test_{kernel}_single.py` using SingleKernelTest
2. Create `test_{kernel}_dual.py` using DualKernelTest
3. Run validation: `python tests/validate_migration.py {Kernel}`
4. Keep both old and new tests until Phase 4

**Validation:**
```bash
# After each kernel migration
python tests/validate_migration.py {KernelName}

# Full test suite should still pass
pytest tests/ -v

# Coverage should improve or stay same
pytest --cov=tests tests/
```

---

## Phase 4: Cleanup and Documentation (Week 3)

**Goal:** Remove deprecated code, update docs

**Risk Level:** LOW - Migration complete and validated

### Task 4.1: Mark Old Frameworks as Deprecated

**Files to Modify:**
- `tests/pipeline/base_integration_test.py`
- `tests/parity/core_parity_test.py`
- `tests/parity/hw_estimation_parity_test.py`
- `tests/dual_pipeline/dual_pipeline_parity_test_v2.py`

**Changes:**
Add deprecation warnings to each class:

```python
import warnings

class IntegratedPipelineTest(ABC):
    """DEPRECATED: Use SingleKernelTest instead.

    This class will be removed in v2.0.

    Migration guide:
        from tests.frameworks.single_kernel_test import SingleKernelTest

        # Old:
        class TestMyKernel(IntegratedPipelineTest):
            ...

        # New:
        class TestMyKernel(SingleKernelTest):
            ...  # Same interface!
    """

    def __init__(self):
        warnings.warn(
            "IntegratedPipelineTest is deprecated. "
            "Use SingleKernelTest from tests.frameworks.single_kernel_test",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
```

**Validation:**
```bash
# Run old tests - should show warnings
pytest tests/pipeline/test_addstreams_integration.py -v 2>&1 | grep DeprecationWarning
```

---

### Task 4.2: Delete Deprecated Code

**Files to DELETE:**
- `tests/pipeline/base_integration_test.py` (400 lines)
- `tests/parity/core_parity_test.py` (200 lines)
- `tests/parity/hw_estimation_parity_test.py` (200 lines)
- `tests/dual_pipeline/dual_pipeline_parity_test_v2.py` (200 lines)
- `tests/parity/backend_helpers.py` (150 lines) - functionality moved to PipelineRunner
- `tests/common/golden_reference_mixin.py` (150 lines) - replaced by GoldenValidator

**Files to DELETE (Old Test Implementations):**
- `tests/pipeline/test_addstreams_integration.py`
- `tests/dual_pipeline/test_addstreams_v2.py`
- (Repeat for each migrated kernel)

**Total Deletion:** ~1300 lines of test infrastructure + ~500 lines of test implementations = **1800 lines**

**Before Deletion:**
```bash
# Verify ALL tests migrated
grep -r "IntegratedPipelineTest" tests/ --include="*.py" | grep -v "DEPRECATED"
grep -r "CoreParityTest" tests/ --include="*.py" | grep -v "DEPRECATED"
# Should return nothing

# Final validation - all tests pass
pytest tests/ -v --tb=short

# Record line count before
cloc tests/
```

**After Deletion:**
```bash
# Record line count after
cloc tests/

# Verify reduction
# Should show ~1800 fewer lines
```

---

### Task 4.3: Update Documentation

**Files to CREATE/UPDATE:**

1. **`tests/README.md`** (UPDATE)
```markdown
# Brainsmith Testing Infrastructure

## Test Frameworks

### SingleKernelTest
Test a single kernel implementation against golden reference.

**Use when:**
- Testing new kernel implementation
- Validating against NumPy/PyTorch reference
- Testing single backend (HLS or RTL)

**Example:**
```python
from tests.frameworks.single_kernel_test import SingleKernelTest

class TestMyKernel(SingleKernelTest):
    def make_test_model(self):
        # Create ONNX Add node
        return model, "Add_0"

    def get_kernel_inference_transform(self):
        return InferKernelList

    def compute_golden_reference(self, inputs):
        return {"output": inputs["input0"] + inputs["input1"]}
```

Inherited tests (6):
- test_pipeline_creates_hw_node
- test_shapes_preserved
- test_datatypes_preserved
- test_python_execution_vs_golden
- test_cppsim_execution_vs_golden
- test_rtlsim_execution_vs_golden

### DualKernelTest
Test manual vs auto parity + both against golden reference.

**Use when:**
- Migrating FINN kernel to Brainsmith
- Validating backward compatibility
- Testing multiple backends

**Example:**
```python
from tests.frameworks.dual_kernel_test import DualKernelTest

class TestAddStreamsParity(DualKernelTest):
    def get_manual_transform(self):
        return InferAddStreamsLayer  # FINN

    def get_auto_transform(self):
        return InferKernelList  # Brainsmith
```

Inherited tests (20):
- 7 structural parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources)
- 8 golden execution tests (manual/auto × Python/cppsim/rtlsim)

## Migration from Old Frameworks

| Old Framework | New Framework | Notes |
|---------------|---------------|-------|
| IntegratedPipelineTest | SingleKernelTest | Same interface |
| CoreParityTest + HWEstimationParityTest + DualPipelineParityTest | DualKernelTest | Combined into one |

## Components

### PipelineRunner (`tests/common/pipeline.py`)
Unified ONNX → Hardware pipeline execution.

### GoldenValidator (`tests/common/validator.py`)
Output validation against golden reference.

### Executors (`tests/parity/executors.py`)
Backend execution: PythonExecutor, CppSimExecutor, RTLSimExecutor
```

2. **`tests/TESTING_GUIDE.md`** (NEW)
Complete guide on writing tests with new frameworks.

3. **`tests/MIGRATION_GUIDE.md`** (NEW)
Step-by-step migration from old to new frameworks.

**Validation:**
- Documentation builds without errors
- Examples in docs actually run
- Links valid

---

## Rollback Procedures

### Phase 1 Rollback
```bash
git rm tests/common/pipeline.py
git rm tests/common/validator.py
git checkout tests/parity/executors.py
git commit -m "Rollback Phase 1"
```

### Phase 2 Rollback
```bash
git rm tests/frameworks/
git commit -m "Rollback Phase 2"
```

### Phase 3 Rollback
```bash
# Keep old tests, delete new tests
git rm tests/pipeline/test_*_single.py
git rm tests/dual_pipeline/test_*_dual.py
git commit -m "Rollback Phase 3"
```

### Phase 4 Rollback
```bash
# Restore deleted files from git history
git checkout HEAD~1 tests/pipeline/base_integration_test.py
# (Repeat for each deleted file)
git commit -m "Rollback Phase 4"
```

---

## Success Metrics

### Code Metrics
- **Line Reduction:** ≥35% (target: 37%)
- **Duplication:** ≤25% (target: 25%, down from 60%)
- **Test Coverage:** ≥90% (maintain or improve)

### Quality Metrics
- **All Tests Pass:** 100% pass rate maintained
- **No Regressions:** Migrated tests produce identical results
- **Performance:** Test execution time within 10% of baseline

### Developer Experience
- **Clarity:** "Which framework?" answerable in 30 seconds from README
- **Onboarding:** New developer can write test in <1 hour
- **Maintenance:** Bug fixes require changes in 1 place (not 3)

---

## Timeline Summary

| Phase | Duration | Risk | Validation |
|-------|----------|------|------------|
| 1: Extract Components | 2-3 days | LOW | Old tests pass |
| 2: New Frameworks | 3-4 days | MEDIUM | Example tests pass |
| 3: Migrate Tests | 5-7 days | MEDIUM | validate_migration.py |
| 4: Cleanup | 1-2 days | LOW | Line count reduction |

**Total: 11-16 days (2-3 weeks)**

---

## Arete Achieved

✅ **Deletion**: 1800 lines removed (37% reduction)

✅ **Standards**: Protocol pattern (PEP 544), composition over inheritance

✅ **Clarity**: "SingleKernelTest tests one kernel" - obvious

✅ **Courage**: Replaced 4 frameworks with 2

✅ **Honesty**: `PipelineRunner.run()` - name says what it does

---

This refactor is **Arete** - testing infrastructure in its highest form.
