# ChannelwiseOp Test Design: Handling Backend-Only Operations

**Date:** 2025-10-31
**Problem:** FINN supports LessOrEqual/GreaterOrEqual in cppsim/rtlsim but NOT in Python execution
**Solution:** Conditional test execution in DualKernelTest framework

---

## The Challenge

ChannelwiseOp has 4 operation modes with **different execution capabilities**:

| Operation | Python Execute | cppsim | rtlsim | Notes |
|-----------|----------------|--------|--------|-------|
| Add | ✅ | ✅ | ✅ | Full pipeline support |
| Mul | ✅ | ✅ | ✅ | Full pipeline support |
| LessOrEqual | ❌ | ✅ | ✅ | Backend only |
| GreaterOrEqual | ❌ | ✅ | ✅ | Backend only |

**Root Cause:** FINN's `ChannelwiseOp.execute_node()` only implements Add/Mul logic. Comparison operations throw errors in Python but work fine in compiled backends.

**Old Test Approach:** Skip entire test classes for leq/geq:
```python
@pytest.mark.skip(reason="FINN doesn't support LessOrEqual")
class TestChannelwiseLessOrEqualParity:
    # All 32 tests skipped
```

**Problem:** This loses ALL coverage for leq/geq, including parity tests and backend execution.

---

## The Solution: Conditional Test Execution

**Key Insight:** Most tests don't execute Python code - they query methods (shapes, widths, etc.). Only **3 of 20** tests execute Python.

### Test Breakdown

**DualKernelTest provides 20 tests:**

**Category 1: Structural Parity (12 tests) - NO EXECUTION**
1. `test_normal_shapes_parity` - Queries `get_normal_input_shape()`, `get_normal_output_shape()`
2. `test_folded_shapes_parity` - Queries `get_folded_input_shape()`, `get_folded_output_shape()`
3. `test_stream_widths_parity` - Queries `get_instream_width()`, `get_outstream_width()`
4. `test_stream_widths_padded_parity` - Queries `get_instream_width_padded()`, etc.
5. `test_datatypes_parity` - Queries `get_input_datatype()`, `get_output_datatype()`
6. `test_datatype_inference_parity` - Queries `infer_node_datatype()`
7. `test_make_shape_compatible_op_parity` - Queries `make_shape_compatible_op()`
8. `test_expected_cycles_parity` - Queries `get_exp_cycles()`
9. `test_number_output_values_parity` - Queries `get_number_output_values()`
10. `test_resource_estimates_parity` - Queries `lut_estimation()`, `bram_estimation()`, etc.
11. `test_efficiency_metrics_parity` - Queries `bram_efficiency_estimation()`, etc.
12. `test_operation_counts_parity` - Queries `get_op_and_param_counts()`

**✅ These work for ALL operations** (no execution, just method calls)

**Category 2: Python Execution (3 tests) - REQUIRES PYTHON EXECUTE**
13. `test_manual_python_vs_golden` - Calls `PythonExecutor.execute(manual_op)`
14. `test_auto_python_vs_golden` - Calls `PythonExecutor.execute(auto_op)`
15. `test_manual_auto_parity_python` - Calls `PythonExecutor.execute()` on both

**❌ These FAIL for leq/geq** (FINN doesn't support Python execution)

**Category 3: Backend Execution (5 tests) - REQUIRES CPPSIM/RTLSIM**
16. `test_manual_cppsim_vs_golden` - Calls `CppSimExecutor.execute(manual_op)`
17. `test_auto_cppsim_vs_golden` - Calls `CppSimExecutor.execute(auto_op)`
18. `test_manual_rtlsim_vs_golden` - Calls `RTLSimExecutor.execute(manual_op)`
19. `test_auto_rtlsim_vs_golden` - Calls `RTLSimExecutor.execute(auto_op)`
20. `test_manual_auto_parity_cppsim` - Calls `CppSimExecutor.execute()` on both

**✅ These work for ALL operations** (backend execution, leq/geq supported)

---

## Implementation Design

### Step 1: Add Capability Detection

```python
class ChannelwiseParityBase(DualKernelTest):
    operation_type: str = "Add"  # Override in subclasses

    def supports_python_execution(self) -> bool:
        """Check if this operation supports Python execution.

        FINN's ChannelwiseOp execute_node() only supports Add/Mul.
        LessOrEqual/GreaterOrEqual only work in cppsim/rtlsim.
        """
        return self.operation_type in ["Add", "Mul"]
```

### Step 2: Override Python Execution Tests

```python
def test_manual_python_vs_golden(self):
    """Test manual Python execution vs golden (skip for comparison ops)."""
    if not self.supports_python_execution():
        pytest.skip(
            f"{self.operation_type} not supported in FINN Python execution "
            "(only cppsim/rtlsim)"
        )
    # Call parent implementation
    super().test_manual_python_vs_golden()

def test_auto_python_vs_golden(self):
    """Test auto Python execution vs golden (skip for comparison ops)."""
    if not self.supports_python_execution():
        pytest.skip(
            f"{self.operation_type} not supported in FINN Python execution "
            "(only cppsim/rtlsim)"
        )
    super().test_auto_python_vs_golden()

def test_manual_auto_parity_python(self):
    """Test manual vs auto Python parity (skip for comparison ops)."""
    if not self.supports_python_execution():
        pytest.skip(
            f"{self.operation_type} not supported in FINN Python execution "
            "(only cppsim/rtlsim)"
        )
    super().test_manual_auto_parity_python()
```

### Step 3: Create 4 Test Classes

```python
class TestChannelwiseAddParity(ChannelwiseParityBase):
    """20 tests: All pass (full pipeline support)"""
    operation_type = "Add"

class TestChannelwiseMulParity(ChannelwiseParityBase):
    """20 tests: All pass (full pipeline support)"""
    operation_type = "Mul"

class TestChannelwiseLessOrEqualParity(ChannelwiseParityBase):
    """20 tests: 17 pass, 3 skip (backend only)"""
    operation_type = "LessOrEqual"

class TestChannelwiseGreaterOrEqualParity(ChannelwiseParityBase):
    """20 tests: 17 pass, 3 skip (backend only)"""
    operation_type = "GreaterOrEqual"
```

---

## Test Coverage Summary

### Add Mode (20 tests)
- ✅ 12 structural parity tests (no execution)
- ✅ 3 Python execution tests
- ✅ 5 backend execution tests

### Mul Mode (20 tests)
- ✅ 12 structural parity tests (no execution)
- ✅ 3 Python execution tests
- ✅ 5 backend execution tests

### LessOrEqual Mode (20 tests: 17 run, 3 skip)
- ✅ 12 structural parity tests (no execution)
- ⏭️ 3 Python execution tests (SKIPPED - not supported)
- ✅ 5 backend execution tests

### GreaterOrEqual Mode (20 tests: 17 run, 3 skip)
- ✅ 12 structural parity tests (no execution)
- ⏭️ 3 Python execution tests (SKIPPED - not supported)
- ✅ 5 backend execution tests

**Total: 80 tests (68 run, 12 skip)**

---

## Running Tests

```bash
# Run all ChannelwiseOp tests (80 tests: 68 run, 12 skip)
pytest tests/kernels/test_channelwise_backend.py -v

# Run only Add mode (20 tests: all pass)
pytest tests/kernels/test_channelwise_backend.py::TestChannelwiseAddParity -v

# Run only comparison operations (40 tests: 34 run, 6 skip)
pytest tests/kernels/test_channelwise_backend.py -k "LessOrEqual or GreaterOrEqual" -v

# Run only backend tests across all modes (20 tests)
pytest tests/kernels/test_channelwise_backend.py -m "cppsim or rtlsim" -v

# Run fast tests only (no cppsim/rtlsim - 60 tests: 48 run, 12 skip)
pytest tests/kernels/test_channelwise_backend.py -m "not slow" -v
```

---

## Example Output

```bash
$ pytest tests/kernels/test_channelwise_backend.py::TestChannelwiseLessOrEqualParity -v

test_normal_shapes_parity                        PASSED  # Structural test
test_folded_shapes_parity                        PASSED  # Structural test
test_stream_widths_parity                        PASSED  # Structural test
test_datatypes_parity                            PASSED  # Structural test
test_expected_cycles_parity                      PASSED  # Estimation test
test_resource_estimates_parity                   PASSED  # Estimation test
test_manual_python_vs_golden                     SKIPPED # Not supported
test_auto_python_vs_golden                       SKIPPED # Not supported
test_manual_cppsim_vs_golden                     PASSED  # Backend test
test_auto_cppsim_vs_golden                       PASSED  # Backend test
test_manual_rtlsim_vs_golden                     PASSED  # Backend test
test_auto_rtlsim_vs_golden                       PASSED  # Backend test
test_manual_auto_parity_python                   SKIPPED # Not supported
test_manual_auto_parity_cppsim                   PASSED  # Backend test

17 passed, 3 skipped
```

---

## Advantages Over Old Approach

### Old Framework (tests/parity/)
```python
@pytest.mark.skip(reason="FINN doesn't support LessOrEqual")
class TestChannelwiseLessOrEqualParity:
    # 32 tests ALL SKIPPED
    # 0% coverage for leq/geq
```

**Coverage:** 0 tests for leq/geq (100% skipped)

### New Framework (tests/kernels/)
```python
class TestChannelwiseLessOrEqualParity(ChannelwiseParityBase):
    operation_type = "LessOrEqual"
    # 20 tests: 17 run, 3 skip
    # 85% coverage for leq/geq
```

**Coverage:** 17 tests for leq/geq (85% run, 15% skipped)

### Comparison

| Metric | Old Framework | New Framework |
|--------|---------------|---------------|
| Tests run for leq/geq | 0 (0%) | 17 (85%) |
| Structural parity validated | ❌ No | ✅ Yes |
| Backend execution validated | ❌ No | ✅ Yes |
| Golden reference tested | ❌ No | ✅ Yes (cppsim/rtlsim) |
| Manual vs auto parity | ❌ No | ✅ Yes (12 structural + 5 backend) |
| Code duplication | High | None (reuses DualKernelTest) |

---

## Design Pattern: Backend-Only Operations

This pattern can be reused for ANY operation that only works in backends:

```python
class MyBackendOnlyKernelTest(DualKernelTest):
    def supports_python_execution(self) -> bool:
        """This kernel only works in cppsim/rtlsim."""
        return False

    def test_manual_python_vs_golden(self):
        if not self.supports_python_execution():
            pytest.skip("Backend-only kernel")
        super().test_manual_python_vs_golden()

    def test_auto_python_vs_golden(self):
        if not self.supports_python_execution():
            pytest.skip("Backend-only kernel")
        super().test_auto_python_vs_golden()

    def test_manual_auto_parity_python(self):
        if not self.supports_python_execution():
            pytest.skip("Backend-only kernel")
        super().test_manual_auto_parity_python()
```

**Result:** Still get 17 of 20 tests (85% coverage) instead of 0%.

---

## Validation Tests

The test file includes 4 meta-tests to validate the design:

1. `test_all_operation_modes_present()` - Verify all 4 modes have test classes
2. `test_add_mul_support_python()` - Verify Add/Mul support Python
3. `test_comparison_ops_backend_only()` - Verify leq/geq are backend-only
4. `test_backend_enabled_for_all()` - Verify backend enabled for all modes

---

## Migration Path

### Old File (BROKEN)
```
brainsmith/kernels/channelwise/tests/test_channelwise_parity.py
- Imports deleted framework (tests/parity/)
- 128 tests (64 effective, rest skipped)
- Cannot run
```

### New File (WORKING)
```
tests/kernels/test_channelwise_backend.py
- Uses DualKernelTest framework
- 80 tests (68 run, 12 skip)
- Runs successfully
- Better coverage (17 tests for leq/geq vs 0)
```

**Action:** Delete old file, use new file.

---

## Conclusion

By using **conditional test execution** instead of **blanket skipping**, we achieve:

- ✅ 85% coverage for backend-only operations (vs 0%)
- ✅ Full structural parity validation
- ✅ Full backend execution validation
- ✅ No code duplication (inherits from DualKernelTest)
- ✅ Clear, understandable skip messages
- ✅ Reusable pattern for other backend-only operations

**Total Test Count:** 80 tests across 4 operation modes
**Effective Coverage:** 68 tests run (12 gracefully skipped)
**Coverage Improvement:** From 64 tests (old) to 68 tests (new) with cleaner design

---

**Status:** Design complete, implementation ready
**File:** `tests/kernels/test_channelwise_backend.py`
**Ready to replace:** `brainsmith/kernels/channelwise/tests/test_channelwise_parity.py`
