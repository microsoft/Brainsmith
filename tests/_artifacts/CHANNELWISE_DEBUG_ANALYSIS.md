# ChannelwiseOp Test Failures: Root Cause Analysis

**Date:** 2025-10-31

---

## Summary of Findings

Running the debug script reveals **3 fundamental issues**:

### Issue 1: ChannelwiseOp has `actual_layouts` Bug ❌

```python
TypeError: TransformationResult.__init__() got an unexpected keyword argument 'actual_layouts'
  File "brainsmith/kernels/channelwise/channelwise.py", line 179
```

**Same bug as ElementwiseBinaryOp!** Needs same fix.

### Issue 2: Different Kernel Inference ❌

**FINN:**
```
Add (with static param) → ChannelwiseOp (domain: finn.custom_op.fpgadataflow)
```

**Brainsmith:**
```
Add (with static param) → ElementwiseBinaryOp (domain: brainsmith.kernels)
                          with input_pattern="dynamic_static"
```

**Problem:** We're comparing DIFFERENT kernels!
- ChannelwiseOp = channelwise operations with per-channel parameters
- ElementwiseBinaryOp = elementwise operations with broadcasting

### Issue 3: FINN Doesn't Support leq/geq Inference ❌

**FINN Manual Transform:**
```
LessOrEqual → LessOrEqual (domain: "")  # NOT CONVERTED!
```

**Brainsmith Auto Transform:**
```
LessOrEqual → ElementwiseBinaryOp (func="LessOrEqual")  # CONVERTED!
```

**Problem:** FINN `InferChannelwiseLinearLayer` doesn't support leq/geq at all!

---

## Detailed Analysis

### Issue 1: actual_layouts Bug

**Location:** `brainsmith/kernels/channelwise/channelwise.py:179`

**Code:**
```python
return TransformationResult(
    nodes_to_remove=[node],
    nodes_to_insert=[hw_node],
    actual_layouts={...},  # ← WRONG!
)
```

**Fix:** Remove `actual_layouts` (same as ElementwiseBinaryOp fix)

```python
return TransformationResult(
    nodes_to_remove=[node],
    nodes_to_insert=[hw_node],
)
```

---

### Issue 2: ChannelwiseOp vs ElementwiseBinaryOp

The fundamental problem: **ChannelwiseOp and ElementwiseBinaryOp are DIFFERENT kernels** that handle the same ONNX operations differently.

**ChannelwiseOp (FINN):**
- Handles: Add/Mul with **per-channel** static parameters
- Pattern: `input + channel_param` where `channel_param.shape == (C,)`
- Example: Bias addition, scale multiplication
- Domain: `finn.custom_op.fpgadataflow`

**ElementwiseBinaryOp (Brainsmith):**
- Handles: Add/Mul/Sub/etc with **broadcasting**
- Pattern: `input0 ⊕ input1` with numpy-style broadcasting
- Example: Any binary operation with broadcasting
- Domain: `brainsmith.kernels`
- Has special mode: `input_pattern="dynamic_static"` for one static input

**Why They're Different:**

| Aspect | ChannelwiseOp | ElementwiseBinaryOp |
|--------|---------------|---------------------|
| Operations | Add, Mul, cmp_le, cmp_ge | Add, Sub, Mul, Div, Min, Max, etc. |
| Input Pattern | 1 dynamic + 1 static (channelwise) | 2 dynamic OR 1 dynamic + 1 static |
| Broadcasting | Per-channel only | Full numpy broadcasting |
| Specialization | Channelwise operations | General elementwise |

**Implication:** We **CANNOT** do direct parity testing between them!

---

### Issue 3: FINN Doesn't Support leq/geq

**Debug Output:**
```
--- FINN Manual Transform (InferChannelwiseLinearLayer) ---
Resulting node: op_type=LessOrEqual, domain=
```

The node **stays as ONNX LessOrEqual** with empty domain - no conversion happens!

**Why?** FINN's `InferChannelwiseLinearLayer` only supports:
- Add → ChannelwiseOp (Func="add")
- Mul → ChannelwiseOp (Func="mul")

It does NOT support:
- LessOrEqual
- GreaterOrEqual

**User's insight was partially correct:**
- LessOrEqual/GreaterOrEqual DO work in cppsim/rtlsim (if manually created)
- BUT FINN doesn't have an inference transform for them
- So we can't test "manual vs auto" because manual doesn't exist!

---

## Why Tests Fail

### Add/Mul Tests Fail

1. **Python tests PASS** ✅
   - Both ChannelwiseOp and ElementwiseBinaryOp support Python execution
   - Golden reference is correct

2. **cppsim tests FAIL** ❌
   - Error: "ElementwiseBinaryOp has no hw implementation variant"
   - Reason: ElementwiseBinaryOp doesn't have `_hls` backend yet
   - SpecializeLayers can't convert ElementwiseBinaryOp → ElementwiseBinaryOp_hls

3. **rtlsim tests FAIL** ❌
   - Same reason as cppsim

### leq/geq Tests Fail

1. **Manual pipeline FAILS** ❌
   - FINN transform doesn't convert LessOrEqual → ChannelwiseOp
   - Node stays as "LessOrEqual" with empty domain
   - When trying to get HWCustomOp, it tries to import from empty module
   - Error: "ValueError: Empty module name"

2. **Auto pipeline creates ElementwiseBinaryOp**
   - Same backend issue as Add/Mul

### Mul Datatype Mismatches

FINN ChannelwiseOp and Brainsmith ElementwiseBinaryOp compute different output datatypes for Mul:

**Example:**
- Input: INT8 * INT4
- FINN: Might return INT12
- Brainsmith: Might return different width

This causes parity test failures.

---

## Root Cause Summary

| Issue | Impact | Fix Complexity |
|-------|--------|----------------|
| actual_layouts bug | All ChannelwiseOp inference fails | Easy (same fix as ElementwiseBinaryOp) |
| Different kernels | Can't test ChannelwiseOp vs ElementwiseBinaryOp parity | Fundamental design issue |
| leq/geq not supported in FINN | Manual pipeline fails | Cannot fix (FINN limitation) |
| ElementwiseBinaryOp no backend | cppsim/rtlsim fail | Need to implement ElementwiseBinaryOp_hls |

---

## Recommendations

### Option A: Fix actual_layouts Bug Only

**Fix:** Remove `actual_layouts` from ChannelwiseOp.infer_from()

**Result:** ChannelwiseOp inference works

**But:** Tests still fail because we're comparing different kernels

---

### Option B: Rethink Test Design (RECOMMENDED)

**Problem:** The current test tries to compare ChannelwiseOp vs ElementwiseBinaryOp, which are fundamentally different kernels.

**Solution:** Create TWO separate test suites:

1. **Test ChannelwiseOp Alone (SingleKernelTest)**
   - Tests ONLY Brainsmith's ChannelwiseOp implementation
   - No parity comparison (FINN's is different)
   - Validates: schema, inference, execution, backend

2. **Test ElementwiseBinaryOp Separately**
   - Already have this: `tests/kernels/test_elementwise_add_backend.py`
   - Tests ElementwiseBinaryOp with various functions

**Why This Makes Sense:**
- They serve different purposes (channelwise vs elementwise)
- They have different input patterns
- Direct comparison is meaningless
- Each should be validated independently

---

### Option C: Delete ChannelwiseOp Tests

**If:** We don't need ChannelwiseOp (ElementwiseBinaryOp covers it)

**Then:** Delete the test, document that ElementwiseBinaryOp replaces it

---

## Immediate Actions Required

### 1. Fix actual_layouts Bug (5 minutes)

```python
# File: brainsmith/kernels/channelwise/channelwise.py

# BEFORE (line 179):
return TransformationResult(
    nodes_to_remove=[node],
    nodes_to_insert=[hw_node],
    actual_layouts={
        "data": None,
        "param": None,
        "output": None,
    },
)

# AFTER:
return TransformationResult(
    nodes_to_remove=[node],
    nodes_to_insert=[hw_node],
)
```

### 2. Delete or Redesign Test (10 minutes)

**Delete:**
```bash
rm tests/kernels/test_channelwise_backend.py
```

**Or Redesign:** Convert to SingleKernelTest for ChannelwiseOp only

---

## Questions for User

1. **Do we need ChannelwiseOp at all?**
   - If ElementwiseBinaryOp covers the same cases, we might not need it

2. **Should we implement ChannelwiseOp_hls backend?**
   - Required if we want to keep ChannelwiseOp

3. **What's the relationship between ChannelwiseOp and ElementwiseBinaryOp?**
   - Are they meant to coexist or is one replacing the other?

---

**Status:** Root causes identified, ready for fix/redesign decision
