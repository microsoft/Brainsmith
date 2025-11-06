# ElementwiseBinary Add Parity Test - Implementation Plan

**Status**: Pilot Implementation
**Date**: 2025-11-05
**Purpose**: Establish pattern for FINN vs Brainsmith parity testing before scaling to all 17 operations

---

## Overview

Create a **single parity test** for the Add operation to:
1. Verify DualKernelTest_v2 framework works for ElementwiseBinary operations
2. Establish the implementation pattern
3. Validate FINN's ElementwiseAdd vs Brainsmith's ElementwiseBinaryOp produce identical results

Once this pilot test passes, we'll extend to the remaining 16 operations.

---

## Directory Structure

```
tests/kernel-migration/                          # NEW
└── elementwise-binary/                          # NEW
    ├── __init__.py                              # Empty marker file
    ├── IMPLEMENTATION_PLAN.md                   # This file
    └── test_add_parity.py                       # Add operation parity test (PILOT)
```

**Future expansion** (after Add passes):
```
tests/kernel-migration/elementwise-binary/
├── __init__.py
├── IMPLEMENTATION_PLAN.md
├── test_add_parity.py           # Pilot (completed)
├── test_sub_parity.py           # Next: Subtraction
├── test_mul_parity.py           # Next: Multiplication
├── test_div_parity.py           # Next: Division
└── ...                          # 13 more operations
```

---

## Implementation: `test_add_parity.py`

### Complete File Content

```python
"""Parity test for ElementwiseBinary Add operation.

Compares FINN's ElementwiseAdd (manual) vs Brainsmith's ElementwiseBinaryOp (auto)
for the Add operation.

This is the pilot implementation to establish the pattern before extending to
all 17 elementwise binary operations.
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.dual_kernel_test_v2 import DualKernelTest_v2


class TestElementwiseBinaryAdd_Parity(DualKernelTest_v2):
    """Test Add operation parity between FINN and Brainsmith.

    Pipeline comparison:
    - Manual (FINN): Add (ONNX) → ElementwiseAdd → ElementwiseAdd_hls
    - Auto (Brainsmith): Add (ONNX) → ElementwiseBinaryOp → ElementwiseBinaryOp_hls

    Provides 18 inherited tests:
    - 7 core parity tests (shapes, widths, datatypes)
    - 5 HW estimation tests (cycles, resources)
    - 6 golden execution tests (2 Python + 4 backend if enabled)
    """

    # ========================================================================
    # Test Configuration (Attribute-based)
    # ========================================================================

    batch = 1
    channels = 64
    input_dtype = DataType["INT8"]

    # ========================================================================
    # Model Creation (v2.3 interface with direct DataType annotations)
    # ========================================================================

    def make_test_model(self, input_shapes):
        """Create Add operation test model.

        Args:
            input_shapes: Dict with input shapes (not used - we use class attributes)

        Returns:
            (model, input_names): Model with Add ONNX node and list of inputs to annotate
        """
        # Use shapes from class attributes
        shape = [self.batch, self.channels]

        # Create inputs with FLOAT container type (QONNX convention)
        inp0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, shape)
        inp1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        # Create Add ONNX node
        node = helper.make_node(
            "Add",
            ["input0", "input1"],
            ["output"],
            name="Add_0"
        )

        graph = helper.make_graph(
            [node],
            "test_elementwise_binary_add",
            [inp0, inp1],
            [out]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names for DataType annotation
        # Framework will annotate these with datatypes from _get_input_datatypes()
        return model, ["input0", "input1"]

    # ========================================================================
    # Input/Output Configuration
    # ========================================================================

    def _get_input_shapes(self):
        """Return input shapes from class attributes."""
        shape = (self.batch, self.channels)
        return {
            "input0": shape,
            "input1": shape,
        }

    def _get_input_datatypes(self):
        """Return input datatypes for QONNX annotation."""
        return {
            "input0": self.input_dtype,
            "input1": self.input_dtype,
        }

    def get_num_inputs(self):
        """ElementwiseBinary Add has 2 inputs."""
        return 2

    def get_num_outputs(self):
        """ElementwiseBinary Add has 1 output."""
        return 1

    # ========================================================================
    # Transform Configuration (Manual vs Auto)
    # ========================================================================

    def get_manual_transform(self):
        """Return FINN's manual ElementwiseBinary transform.

        FINN uses a unified InferElementwiseBinaryOperation transform that
        auto-detects the operation type from the ONNX node's op_type.

        Returns:
            InferElementwiseBinaryOperation class
        """
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferElementwiseBinaryOperation,
        )
        return InferElementwiseBinaryOperation

    def get_auto_transform(self):
        """Return Brainsmith's unified kernel inference transform.

        Brainsmith uses InferKernels with ElementwiseBinaryOp, which creates
        a polymorphic kernel that handles all binary operations via the 'func'
        parameter.

        Returns:
            Callable that creates InferKernels([ElementwiseBinaryOp])
        """
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return a lambda that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    def get_manual_backend_variants(self):
        """Return FINN backend for manual pipeline.

        REQUIRED: Must use FINN's ElementwiseAdd_hls backend because the manual
        pipeline creates nodes with FINN-specific attributes (e.g., "Func" vs "func").

        Returns:
            List containing ElementwiseAdd_hls class
        """
        from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
            ElementwiseAdd_hls,
        )
        return [ElementwiseAdd_hls]

    # get_auto_backend_variants() not needed - auto-detects from Brainsmith registry

    # ========================================================================
    # Backend Testing (Optional)
    # ========================================================================

    # Uncomment to enable cppsim/rtlsim tests (adds 4 tests: 2 cppsim + 2 rtlsim)
    # def get_backend_fpgapart(self):
    #     return "xc7z020clg400-1"

    # ========================================================================
    # Golden Reference
    # ========================================================================

    def compute_golden_reference(self, inputs):
        """Compute expected output using NumPy.

        Args:
            inputs: Dict mapping input names to numpy arrays
                   {"input0": ndarray, "input1": ndarray}

        Returns:
            Dict mapping output names to expected numpy arrays
            {"output": ndarray}
        """
        return {"output": inputs["input0"] + inputs["input1"]}
```

---

## Expected Test Output

When running `pytest tests/kernel-migration/elementwise-binary/test_add_parity.py -v`:

```
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_normal_shapes_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_folded_shapes_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_stream_widths_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_stream_widths_padded_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_datatypes_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_datatype_inference_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_make_shape_compatible_op_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_expected_cycles_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_number_output_values_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_resource_estimates_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_efficiency_metrics_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_operation_counts_parity PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_manual_python_vs_golden PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_auto_python_vs_golden PASSED
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_manual_cppsim_vs_golden SKIPPED (backend not configured)
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_auto_cppsim_vs_golden SKIPPED (backend not configured)
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_manual_rtlsim_vs_golden SKIPPED (backend not configured)
tests/kernel-migration/elementwise-binary/test_add_parity.py::TestElementwiseBinaryAdd_Parity::test_auto_rtlsim_vs_golden SKIPPED (backend not configured)

==================== 14 passed, 4 skipped in X.XXs ====================
```

### Test Breakdown
- ✅ **12 parity tests** (7 core + 5 HW estimation) - Validate FINN ↔ Brainsmith equivalence
- ✅ **2 Python golden tests** - Validate both implementations vs NumPy reference
- ⏭️ **4 backend tests** (skipped without fpgapart) - Would validate cppsim/rtlsim if enabled

---

## Verification Steps

### Step 1: Run Basic Tests
```bash
pytest tests/kernel-migration/elementwise-binary/test_add_parity.py -v
```

**Expected**: 14 passed, 4 skipped

### Step 2: Verify Parity Tests Pass
All 12 parity tests should **PASS**, confirming:
- ✅ Input/output shapes match (normal and folded)
- ✅ Stream widths match (raw and AXI-padded)
- ✅ Datatypes match (inputs and outputs)
- ✅ Cycle counts match
- ✅ Resource estimates match (LUT, DSP, BRAM, URAM)

### Step 3: Verify Golden Tests Pass
Both Python golden tests should **PASS**, confirming:
- ✅ FINN's ElementwiseAdd produces correct results
- ✅ Brainsmith's ElementwiseBinaryOp produces correct results

### Step 4: (Optional) Enable Backend Tests

Uncomment in `test_add_parity.py`:
```python
def get_backend_fpgapart(self):
    return "xc7z020clg400-1"
```

Then run with HLS environment:
```bash
source .brainsmith/env.sh
pytest tests/kernel-migration/elementwise-binary/test_add_parity.py -v --tb=short
```

**Expected**: 18 tests (14 pass + 4 backend tests)

**Warning**: Backend tests are **SLOW** (2-10 minutes per test for compilation)

---

## Key Implementation Details

### 1. FINN Transform: Unified for All Operations
```python
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
)
```

FINN uses **ONE transform** for all 17 operations. It auto-detects the operation type from the ONNX node's `op_type` attribute (e.g., "Add", "Sub", "Mul").

### 2. Backend Class Selection: Operation-Specific
Each operation has its own FINN backend class:
- `ElementwiseAdd_hls` - Addition
- `ElementwiseSub_hls` - Subtraction
- `ElementwiseMul_hls` - Multiplication
- `ElementwiseDiv_hls` - Division
- ... (13 more)

All located in: `deps/finn/src/finn/custom_op/fpgadataflow/hls/elementwise_binary_hls.py`

### 3. Brainsmith: Polymorphic Kernel
Brainsmith uses a **single polymorphic kernel** (`ElementwiseBinaryOp`) that handles all 17 operations via the `func` parameter:
```python
ElementwiseBinaryOp(func="Add")   # Addition
ElementwiseBinaryOp(func="Sub")   # Subtraction
ElementwiseBinaryOp(func="Mul")   # Multiplication
# ... etc
```

### 4. Golden Reference: NumPy Operations
Use NumPy operations that match hardware behavior:
```python
def compute_golden_reference(self, inputs):
    return {"output": inputs["input0"] + inputs["input1"]}  # Simple addition
```

### 5. Configuration Attributes
```python
batch = 1           # Batch size
channels = 64       # Channel dimension (must be divisible by PE if using folding)
input_dtype = DataType["INT8"]  # Input datatype (INT8, INT16, BIPOLAR, etc.)
```

**Datatype recommendations for Add**:
- INT8, INT16 (integer types)
- BIPOLAR (-1, 1)
- Avoid FLOAT types for now (focus on integer validation first)

---

## Architecture: FINN vs Brainsmith

### FINN Pipeline (Manual)
```
Add (ONNX)
  ↓ InferElementwiseBinaryOperation
ElementwiseAdd (Stage 2: Base kernel)
  ↓ SpecializeLayers
ElementwiseAdd_hls (Stage 3: HLS backend)
```

**Attributes**: `Func="Add"` (capital F), `lhs_dtype`, `rhs_dtype`, `out_dtype`

### Brainsmith Pipeline (Auto)
```
Add (ONNX)
  ↓ InferKernels([ElementwiseBinaryOp])
ElementwiseBinaryOp (Stage 2: Base kernel)
  ↓ SpecializeLayers
ElementwiseBinaryOp_hls (Stage 3: HLS backend)
```

**Attributes**: `func="Add"` (lowercase f), `input0Datatype`, `input1Datatype`, `output0Datatype`

### Key Difference: Attribute Naming
- FINN uses: `Func`, `lhs_dtype`, `rhs_dtype`, `out_dtype`
- Brainsmith uses: `func`, `input0Datatype`, `input1Datatype`, `output0Datatype`

This is why **backend variants MUST be explicitly specified** for the manual pipeline - auto-detection would use Brainsmith backends (wrong attributes).

---

## Troubleshooting

### Issue: Transform creates wrong node type
**Symptom**: FINN transform creates `ElementwiseAdd` but Brainsmith creates different node
**Cause**: Transform configuration mismatch
**Fix**: Verify `get_manual_transform()` returns `InferElementwiseBinaryOperation`

### Issue: Backend specialization fails
**Symptom**: `get_manual_backend_variants()` doesn't find FINN backend
**Cause**: Backend variants not explicitly specified
**Fix**: Always specify `[ElementwiseAdd_hls]` for manual pipeline

### Issue: Datatype mismatch in parity tests
**Symptom**: Parity test fails on datatype comparison
**Cause**: FINN and Brainsmith use different datatype derivation rules
**Fix**: Review output datatype derivation in both implementations (should match UG1399)

### Issue: Golden reference doesn't match
**Symptom**: Python golden test fails
**Cause**: NumPy operation doesn't match hardware behavior
**Fix**: Verify golden reference uses correct NumPy operation (simple `+` for Add)

---

## Next Steps (After Add Passes)

Once all 14 tests pass:

### Phase 1: Arithmetic Operations
1. **Copy `test_add_parity.py` → `test_sub_parity.py`**
   - Change ONNX node: `"Add"` → `"Sub"`
   - Change backend: `ElementwiseAdd_hls` → `ElementwiseSub_hls`
   - Change golden: `inputs["input0"] + inputs["input1"]` → `inputs["input0"] - inputs["input1"]`

2. **Copy → `test_mul_parity.py`**
   - ONNX node: `"Mul"`
   - Backend: `ElementwiseMul_hls`
   - Golden: `inputs["input0"] * inputs["input1"]`

3. **Copy → `test_div_parity.py`**
   - ONNX node: `"Div"`
   - Backend: `ElementwiseDiv_hls`
   - Golden: `np.trunc(np.divide(inputs["input0"], inputs["input1"]))` (integer division)

### Phase 2: Logical Operations
4. **Create `test_and_parity.py`, `test_or_parity.py`, `test_xor_parity.py`**
   - ONNX nodes: `"And"`, `"Or"`, `"Xor"`
   - Backends: `ElementwiseAnd_hls`, `ElementwiseOr_hls`, `ElementwiseXor_hls`
   - Golden: `np.logical_and`, `np.logical_or`, `np.logical_xor`
   - **Note**: Output datatype should be BINARY (0 or 1)

### Phase 3: Comparison Operations
5. **Create comparison tests** (5 operations)
   - Equal, Less, LessOrEqual, Greater, GreaterOrEqual
   - All return BINARY output

### Phase 4: Bitwise Operations
6. **Create bitwise tests** (3 operations)
   - BitwiseAnd, BitwiseOr, BitwiseXor
   - Use `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`

### Phase 5: BitShift (Special Case)
7. **Create `test_bitshift_parity.py`** with **TWO** test classes:
   - `TestBitShiftLeft_Parity` - Set `direction="LEFT"` attribute
   - `TestBitShiftRight_Parity` - Set `direction="RIGHT"` attribute

### Phase 6: Organization (Optional)
8. **Group into category files**:
   - `test_arithmetic_parity.py` - Combine Add, Sub, Mul, Div
   - `test_logical_parity.py` - Combine And, Or, Xor
   - `test_comparison_parity.py` - Combine all 5 comparison ops
   - `test_bitwise_parity.py` - Combine BitwiseAnd, BitwiseOr, BitwiseXor
   - `test_bitshift_parity.py` - Both LEFT and RIGHT

---

## Success Criteria

This pilot implementation is considered **successful** when:

1. ✅ All 12 parity tests **PASS** (FINN ↔ Brainsmith equivalence confirmed)
2. ✅ Both Python golden tests **PASS** (correctness confirmed)
3. ✅ Test pattern is clear and reusable for other operations
4. ✅ Documentation is complete (this file)
5. ✅ Ready to extend to remaining 16 operations

---

## References

- **Test Framework**: `tests/frameworks/dual_kernel_test_v2.py`
- **Example Test**: `tests/kernels/test_addstreams_parity_poc.py`
- **FINN Transform**: `deps/finn/src/finn/transformation/fpgadataflow/convert_to_hw_layers.py:1753-1841`
- **FINN Backends**: `deps/finn/src/finn/custom_op/fpgadataflow/hls/elementwise_binary_hls.py`
- **Brainsmith Kernel**: `brainsmith/kernels/elementwise_binary/elementwise_binary.py`
- **Brainsmith Backend**: `brainsmith/kernels/elementwise_binary/elementwise_binary_hls.py`

---

**Document Version**: 1.0
**Last Updated**: 2025-11-05
**Status**: Ready for implementation
