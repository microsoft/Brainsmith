# KernelParityTest Framework Guide

**Status:** Production Ready
**Last Updated:** 2025-11-07

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

---

## Overview

KernelParityTest is a **fixture-based test framework** for comparing two kernel implementations (e.g., FINN vs Brainsmith). It provides:

- **18 inherited tests** - 6 golden execution + 7 core parity + 5 HW estimation
- **Fixture-based parameterization** - 90% shared config, 10% per-kernel hooks
- **Session-scoped caching** - Computational reuse across tests
- **Expected failures are features** - Tests SHOULD fail when implementations differ
- **Asymmetric design** - Reference explicit, primary inherited

### When to Use

✅ **Use KernelParityTest when:**
- Comparing FINN vs Brainsmith implementations
- Testing two different inference transforms
- Validating parity across shapes, widths, datatypes, resources
- Need fixture-based test parameterization

❌ **Use KernelTest when:**
- Testing only ONE implementation
- Don't need parity comparisons

### What You Get

**18 Inherited Tests:**

```
Golden Execution (6 tests)
├─ test_reference_python_vs_golden
├─ test_primary_python_vs_golden
├─ test_reference_cppsim_vs_golden
├─ test_primary_cppsim_vs_golden
├─ test_reference_rtlsim_vs_golden
└─ test_primary_rtlsim_vs_golden

Core Parity (7 tests)
├─ test_normal_shapes_parity
├─ test_folded_shapes_parity
├─ test_stream_widths_parity
├─ test_stream_widths_padded_parity
├─ test_datatypes_parity
├─ test_datatype_inference_parity
└─ test_make_shape_compatible_op_parity

HW Estimation (5 tests)
├─ test_expected_cycles_parity
├─ test_number_output_values_parity
├─ test_resource_estimates_parity
├─ test_efficiency_metrics_parity
└─ test_operation_counts_parity
```

---

## Quick Start

### Minimal Example

```python
from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.test_config import KernelTestConfig, ModelStructure
from qonnx.core.datatype import DataType
import pytest
import numpy as np

class TestAddParity(KernelParityTest):
    """Compare FINN ElementwiseAdd vs Brainsmith ElementwiseBinaryOp."""

    # 1. Test Configuration (fixture-based)
    @pytest.fixture(
        params=[
            KernelTestConfig(
                test_id="add_int8",
                model=ModelStructure(
                    operation="Add",
                    input_shapes={"input0": (1, 64), "input1": (1, 64)},
                    input_dtypes={
                        "input0": DataType["INT8"],
                        "input1": DataType["INT8"]
                    },
                ),
            )
        ]
    )
    def kernel_test_config(self, request):
        return request.param

    # 2. Shared Model Creation
    def make_test_model(self, kernel_test_config):
        # Create ONNX Add node
        model = ...  # ONNX model creation
        return model, ["input0", "input1"]  # input names for annotation

    # 3. Kernel A (FINN)
    def infer_kernel_reference(self, model, target_node):
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferElementwiseBinaryOperation
        )
        model = model.transform(InferElementwiseBinaryOperation())

        # CRITICAL: FINN doesn't preserve node names!
        nodes = model.get_nodes_by_op_type("ElementwiseAdd")
        op = getCustomOp(nodes[0])
        return op, model

    def get_backend_variants_reference(self):
        from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
            ElementwiseAdd_hls
        )
        return [ElementwiseAdd_hls]

    # 4. Kernel B (Brainsmith) - uses defaults
    def get_kernel_op(self):
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
        return ElementwiseBinaryOp

    # 5. Test Structure
    def get_num_inputs(self):
        return 2

    def get_num_outputs(self):
        return 1

    # 6. Golden Reference
    def compute_golden_reference(self, inputs):
        return {"output": inputs["input0"] + inputs["input1"]}
```

**Result:** 18 tests collected, 10 passed, 4 skipped (no FPGA), 4 failed (expected differences)

---

## Architecture

### Design Philosophy

**90/10 Configuration Rule:**
- **90% shared** - Model, inputs, golden reference (via fixtures)
- **10% per-kernel** - Inference and backend selection (via method hooks)

**Fixture Dependency Graph:**

```
kernel_test_config (pytest fixture - parameterization)
    ↓
stage1_model (session-scoped - ONNX model)
    ├→ test_inputs (session-scoped - NumPy arrays)
    │      ↓
    │  golden_outputs (session-scoped - expected results)
    │
    ├→ stage2_model_a (session-scoped - Kernel A inference)
    │      ↓
    │  stage3_model_a (session-scoped - Kernel A backend)
    │
    └→ stage2_model_b (session-scoped - Kernel B inference)
           ↓
       stage3_model_b (session-scoped - Kernel B backend)
```

**Key Principles:**

1. **Session-scoped caching** - Each fixture computed once per test session
2. **Model immutability** - Each stage returns new model (no in-place mutation)
3. **Progressive specialization** - Stage 1 → 2 → 3 (ONNX → Kernel → Backend)
4. **Asymmetric API** - Explicit reference methods, no method swapping

### The Asymmetric API Pattern

**Problem:** Both kernels share the same base class, but need different backend variants.

**Solution:** Use explicit reference-specific methods:

```python
# Reference implementation (explicit methods)
def infer_kernel_reference(self, model, target_node):
    """Reference kernel inference (e.g., FINN)."""
    from finn.transformation.fpgadataflow.convert_to_hw_layers import InferElementwiseBinaryOperation
    model = model.transform(InferElementwiseBinaryOperation())
    nodes = model.get_nodes_by_op_type("ElementwiseAdd")
    from qonnx.custom_op.registry import getCustomOp
    return getCustomOp(nodes[0]), model

def get_backend_variants_reference(self):
    """Reference backends (e.g., FINN HLS)."""
    from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import ElementwiseAdd_hls
    return [ElementwiseAdd_hls]

# Primary implementation (inherited from base)
def get_kernel_op(self):
    """Primary kernel (e.g., Brainsmith)."""
    from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
    return ElementwiseBinaryOp
```

**Why This Works:**
- No method swapping or state mutation
- Clear asymmetry: reference explicit, primary inherited
- Each implementation uses its own methods
- Fixtures call appropriate method for each path

---

## API Reference

### Abstract Methods (5 - MUST implement)

#### `make_test_model(kernel_test_config) -> Tuple[ModelWrapper, List[str]]`

Create shared ONNX model for both kernels.

**Args:**
- `kernel_test_config`: Test configuration with shapes/datatypes

**Returns:**
- `(model, input_names)` - ModelWrapper and list of input names for annotation

**Example:**
```python
def make_test_model(self, kernel_test_config):
    shape = kernel_test_config.input_shapes["input"]
    node = helper.make_node("Add", ["in0", "in1"], ["out"])
    # ... create graph and model ...
    return model, ["in0", "in1"]
```

#### `infer_kernel_reference(model, target_node) -> Tuple[CustomOp, ModelWrapper]`

Apply Kernel A inference transform (usually FINN).

**Args:**
- `model`: Stage 1 ONNX model
- `target_node`: Name of target node (before transformation)

**Returns:**
- `(op, model)` - Custom op instance and transformed model

**Critical Pattern - FINN Node Naming:**

```python
def infer_kernel_reference(self, model, target_node):
    # Apply FINN transform
    model = model.transform(InferElementwiseBinaryOperation())

    # CRITICAL: FINN doesn't preserve node names!
    # Search by op_type instead of name
    nodes_by_op_type = model.get_nodes_by_op_type("ElementwiseAdd")
    assert len(nodes_by_op_type) == 1, (
        f"Expected exactly 1 ElementwiseAdd node, found {len(nodes_by_op_type)}"
    )

    # Wrap with custom op
    from qonnx.custom_op.registry import getCustomOp
    op = getCustomOp(nodes_by_op_type[0])
    return op, model
```

**Why:** FINN's transforms create new nodes without preserving original names. Always search by `op_type` when using FINN transforms.

#### `get_backend_variants_reference() -> List[Type]`

Return Kernel A backend classes (explicit).

**Returns:**
- List of backend classes (e.g., `[ElementwiseAdd_hls]`)

**Example:**
```python
def get_backend_variants_reference(self):
    from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
        ElementwiseAdd_hls
    )
    return [ElementwiseAdd_hls]
```

#### `get_num_inputs() -> int`

Return number of inputs for this operation.

**Returns:**
- Integer count of inputs

#### `get_num_outputs() -> int`

Return number of outputs for this operation.

**Returns:**
- Integer count of outputs

### Optional Overrides (4 - with defaults)

#### `infer_primary(model, target_node) -> Tuple[CustomOp, ModelWrapper]`

Apply Kernel B inference transform (usually Brainsmith).

**Default:** Calls `self.infer_kernel()` (from KernelTestBase)

**Override when:** Need custom Kernel B inference logic

**Example (using default):**
```python
# No override needed!
# Default calls get_kernel_op() automatically

def get_kernel_op(self):
    from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
    return ElementwiseBinaryOp
```

#### `get_backend_variants_b() -> List[Type] | None`

Return Kernel B backend classes.

**Default:** `None` (auto-detect from registry)

**Override when:** Need explicit Kernel B backend selection

**Example:**
```python
def get_backend_variants_b(self):
    from brainsmith.kernels.elementwise_binary.elementwise_binary_hls import (
        ElementwiseBinaryOp_hls
    )
    return [ElementwiseBinaryOp_hls]
```

#### `configure_kernel_reference(op, model) -> None`

Configure Kernel A node attributes.

**Default:** Calls `auto_configure_from_fixture(op, model)`

**Override when:** Need custom Kernel A configuration

**Example:**
```python
def configure_kernel_reference(self, op, model):
    # Custom FINN configuration
    op.set_nodeattr("PE", 4)
    op.set_nodeattr("SIMD", 8)
```

#### `configure_primary(op, model) -> None`

Configure Kernel B node attributes.

**Default:** Calls `auto_configure_from_fixture(op, model)`

**Override when:** Need custom Kernel B configuration

### Backend Specialization (2 - v5.0)

#### `specialize_to_backend_a(op, model, config)`

Specialize Kernel A to backend (Stage 2 → Stage 3).

**Default:** Calls base class `specialize_to_backend()`

**Override when:** Need custom Kernel A backend logic

#### `specialize_to_backend_b(op, model, config)`

Specialize Kernel B to backend (Stage 2 → Stage 3).

**Default:** Calls base class `specialize_to_backend()`

**Override when:** Need custom Kernel B backend logic

### Helper Methods (3 - framework internal)

These are **framework internal** - you don't override them, but they were added during Phase 3 integration:

#### `_prepare_model_with_annotations(kernel_test_config)`

Create model with QONNX DataType annotations (NO Quant nodes).

#### `_generate_test_inputs(kernel_test_config)`

Generate test data with correct shapes and datatypes.

#### `_compute_golden_reference(quant_model, inputs)`

Compute golden reference using test's `compute_golden_reference()`.

---

## Examples

### Example 1: Minimal Parity Test

See [Quick Start](#quick-start) above.

### Example 2: Multiple Test Configurations

```python
class TestAddParity(KernelParityTest):
    @pytest.fixture(
        params=[
            # Test 1: INT8
            KernelTestConfig(
                test_id="add_int8",
                model=ModelStructure(
                    operation="Add",
                    input_shapes={"input0": (1, 64), "input1": (1, 64)},
                    input_dtypes={
                        "input0": DataType["INT8"],
                        "input1": DataType["INT8"]
                    },
                ),
            ),
            # Test 2: INT16
            KernelTestConfig(
                test_id="add_int16",
                model=ModelStructure(
                    operation="Add",
                    input_shapes={"input0": (1, 128), "input1": (1, 128)},
                    input_dtypes={
                        "input0": DataType["INT16"],
                        "input1": DataType["INT16"]
                    },
                ),
            ),
        ]
    )
    def kernel_test_config(self, request):
        return request.param

    # ... rest of implementation ...
```

**Result:** 18 tests × 2 configurations = 36 tests collected

### Example 3: Custom Kernel A Configuration

```python
class TestMVAUParity(KernelParityTest):
    # ... fixtures and required methods ...

    def configure_kernel_reference(self, op, model):
        """Custom FINN MVAU configuration."""
        # Don't call auto_configure - do it manually
        op.set_nodeattr("PE", 4)
        op.set_nodeattr("SIMD", 8)
        op.set_nodeattr("MVAU_ACT", 0)  # FINN-specific
```

### Example 4: Different Backend Types

```python
class TestConvParity(KernelParityTest):
    # ... fixtures and required methods ...

    def get_backend_variants_reference(self):
        """Kernel A uses HLS backend."""
        from finn.custom_op.fpgadataflow.hls.conv_hls import Conv_hls
        return [Conv_hls]

    def get_backend_variants_b(self):
        """Kernel B uses RTL backend."""
        from brainsmith.kernels.conv.conv_rtl import ConvOp_rtl
        return [ConvOp_rtl]
```

---

## Troubleshooting

### Test Failures Are Expected!

**Parity tests SHOULD fail when implementations differ.** This is a **feature, not a bug**.

**Example from test_add_parity.py:**

```
FAILED test_datatypes_parity - AssertionError: Output datatypes don't match
  Kernel A (FINN): INT9    # Prevents overflow
  Kernel B (Brainsmith): INT8  # Assumes saturation
```

**This failure reveals a real implementation difference worth investigating!**

### Common Issues

#### Issue 1: "Expected exactly 1 node, found 0"

**Cause:** FINN transform didn't create expected node type

**Solution:** Check FINN transform output:
```python
print(f"Node types: {[n.op_type for n in model.graph.node]}")
```

#### Issue 2: "Backend specialization requires fpgapart"

**Cause:** Backend tests need fpgapart configured in test config

**Solution:** Add platform config to your test configuration:
```python
from tests.frameworks.test_config import PlatformConfig

@pytest.fixture(params=[
    KernelTestConfig(
        test_id="my_test_with_backend",
        model=ModelStructure(...),
        platform=PlatformConfig(fpgapart="xc7z020clg400-1")  # Enable backend!
    )
])
def kernel_test_config(self, request):
    return request.param
```

**Control test execution with pytest marks:**
```bash
# Skip backend tests
pytest test_my_parity.py -m "not cppsim and not rtlsim" -v

# Run ONLY rtlsim
pytest test_my_parity.py -m "rtlsim" -v
```

#### Issue 3: "Fixture not found: kernel_test_config"

**Cause:** Missing pytest fixture definition

**Solution:** Add fixture to your test class:
```python
@pytest.fixture(params=[...])
def kernel_test_config(self, request):
    return request.param
```

#### Issue 4: "Node name not preserved after transform"

**Cause:** FINN transforms don't preserve node names

**Solution:** Search by `op_type` instead:
```python
# Don't do this:
op = self._find_hw_node(model, "Add_0")  # Fails!

# Do this:
nodes = model.get_nodes_by_op_type("ElementwiseAdd")
op = getCustomOp(nodes[0])  # Works!
```

---

## See Also

- **Implementation:** `tests/frameworks/kernel_parity_test.py`
- **Working Example:** `tests/kernels/elementwise_binary/test_add_parity.py`
- **Base Class:** `tests/frameworks/kernel_test_base_v2.py`
- **Flow Diagram:** `_artifacts/phase3_kernelparitytest_flow.md`
- **Status:** `_artifacts/PHASE_STATUS.md`

---

**KernelParityTest v5.0**
Fixture-Based Architecture • 18 Inherited Tests • Production Ready
