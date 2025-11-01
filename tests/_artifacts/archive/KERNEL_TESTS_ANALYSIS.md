# Kernel-Embedded Tests Analysis

**Date:** 2025-10-31
**Question:** Can brainsmith/kernels/*/tests/ directories be safely deleted?

---

## Executive Summary

**Answer:** **PARTIAL** - Some can be deleted, some should be preserved and migrated.

**Breakdown:**
- **Delete:** Parity tests using old framework (BROKEN - depend on deleted tests/parity/)
- **Preserve:** Unit tests for kernel implementation logic (schema, can_infer_from, etc.)

---

## Current State

### Kernel Directories with Tests

```bash
brainsmith/kernels/addstreams/tests/
brainsmith/kernels/channelwise/tests/
brainsmith/kernels/duplicate_streams/tests/
brainsmith/kernels/elementwise_binary/tests/
```

Each contains 2 files:
1. `test_<kernel>.py` - Unit tests for kernel implementation
2. `test_<kernel>_parity.py` - Parity tests using OLD framework

---

## Analysis by File Type

### 1. Parity Tests (BROKEN - Can Delete)

**Files:**
- `addstreams/tests/test_addstreams_parity.py`
- `channelwise/tests/test_channelwise_parity.py`
- `duplicate_streams/tests/test_duplicatestreams_parity.py`
- `elementwise_binary/tests/test_elementwise_binary_parity.py`

**Problem:** All import from deleted framework:
```python
from tests.parity.base_parity_test import ParityTestBase
from tests.parity.hls_codegen_parity import HLSCodegenParityMixin
```

**Status:** ❌ **BROKEN** (tests/parity/ was deleted in consolidation)

**Replacement:** ✅ **COVERED** by new framework:
- `tests/frameworks/test_addstreams_dual_backend.py` (AddStreams parity)
- `tests/kernels/test_duplicate_streams_backend.py` (DuplicateStreams parity)
- `tests/kernels/test_elementwise_add_backend.py` (ElementwiseBinary parity)

**Recommendation:** **DELETE** (superseded by new framework)

---

### 2. Unit Tests (VALID - Should Preserve)

**Files:**
- `addstreams/tests/test_addstreams.py`
- `channelwise/tests/test_channelwise.py`
- `duplicate_streams/tests/test_duplicatestreams.py`
- `elementwise_binary/tests/test_elementwise_binary.py`

**What They Test:**
- Schema structure validation
- `can_infer_from()` logic (when kernel SHOULD/SHOULDN'T infer)
- `infer_from()` correctness (node creation)
- `build_schema()` behavior
- Edge cases (static inputs, float inputs, shape mismatches)
- Direct `execute_node()` calls

**Example from test_addstreams.py:**
```python
def test_addstreams_schema():
    """Test AddStreams schema structure."""
    assert ADDSTREAMS_SCHEMA.name == "AddStreams"
    assert len(ADDSTREAMS_SCHEMA.inputs) == 2
    # ...

def test_addstreams_cannot_infer_from_static_inputs():
    """Test can_infer_from() rejects Add with static inputs."""
    # Should reject because in1 is static (has initializer)
    assert AddStreams.can_infer_from(add_node, model) is False

def test_addstreams_cannot_infer_from_float_inputs():
    """Test can_infer_from() rejects Add with float inputs."""
    # Should reject because inputs are not integer
    assert AddStreams.can_infer_from(add_node, model) is False
```

**Status:** ✅ **VALID** (self-contained unit tests)

**Covered by New Framework?** ⚠️ **PARTIALLY**
- New framework tests HAPPY PATH (valid inference, correct execution)
- These tests EDGE CASES (rejection logic, schema validation)

**Recommendation:** **MIGRATE** to tests/unit/

---

## Coverage Gap Analysis

### What New Framework Tests

**DualKernelTest (20 tests):**
- ✅ Normal/folded shapes
- ✅ Stream widths
- ✅ Datatypes
- ✅ Resource estimation
- ✅ Execution correctness (Python, cppsim, rtlsim)
- ✅ Manual vs auto parity

**What it DOESN'T test:**
- ❌ Schema structure (inputs, outputs, params)
- ❌ can_infer_from() rejection logic (invalid inputs)
- ❌ Edge cases (static inputs, float inputs, shape mismatches)
- ❌ build_schema() behavior
- ❌ Direct execute_node() calls

### What Unit Tests Provide

**Unique Coverage:**
1. **Schema Validation** - Structure correctness
2. **Inference Logic** - When to accept/reject nodes
3. **Edge Cases** - Boundary conditions, invalid inputs
4. **Direct API Testing** - Bypassing transforms

**Value:** These are **unit tests** for kernel implementation, not integration tests.

---

## Recommendation

### Option A: Full Delete (Risky)

**Delete all brainsmith/kernels/*/tests/ directories**

**Pros:**
- Clean separation (all tests in tests/)
- No kernel-embedded tests

**Cons:**
- ❌ Lose unit test coverage for kernel logic
- ❌ Lose edge case validation
- ❌ Lose schema structure tests

**Risk:** Medium (gaps in unit test coverage)

---

### Option B: Migrate Unit Tests (Recommended)

**Step 1: Delete parity tests** (already covered)
```bash
rm brainsmith/kernels/addstreams/tests/test_addstreams_parity.py
rm brainsmith/kernels/channelwise/tests/test_channelwise_parity.py
rm brainsmith/kernels/duplicate_streams/tests/test_duplicatestreams_parity.py
rm brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py
```

**Step 2: Migrate unit tests to tests/unit/**
```bash
mv brainsmith/kernels/addstreams/tests/test_addstreams.py \
   tests/unit/test_addstreams_kernel.py
mv brainsmith/kernels/channelwise/tests/test_channelwise.py \
   tests/unit/test_channelwise_kernel.py
mv brainsmith/kernels/duplicate_streams/tests/test_duplicatestreams.py \
   tests/unit/test_duplicate_streams_kernel.py
mv brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary.py \
   tests/unit/test_elementwise_binary_kernel.py
```

**Step 3: Delete empty test directories**
```bash
rmdir brainsmith/kernels/addstreams/tests/
rmdir brainsmith/kernels/channelwise/tests/
rmdir brainsmith/kernels/duplicate_streams/tests/
rmdir brainsmith/kernels/elementwise_binary/tests/
```

**Pros:**
- ✅ Preserves unit test coverage
- ✅ Clean separation (all tests in tests/)
- ✅ No broken imports

**Cons:**
- Requires minor import path updates in unit tests

---

### Option C: Leave Unit Tests In Place

**Keep brainsmith/kernels/*/tests/test_*.py** (NOT parity files)

**Pros:**
- No migration needed
- Tests live near kernel code

**Cons:**
- ❌ Mixed location (some tests in tests/, some in brainsmith/)
- ❌ Violates test ownership principle
- ❌ Still have broken parity tests

**Recommendation:** ❌ Not recommended

---

## Final Recommendation

**Execute Option B: Migrate Unit Tests**

1. **DELETE** broken parity tests (4 files)
   - Superseded by new DualKernelTest framework
   - Depend on deleted tests/parity/ framework

2. **MIGRATE** unit tests to tests/unit/ (4 files)
   - Preserve valuable unit test coverage
   - Maintain clean test ownership

3. **DELETE** empty test directories (4 directories)
   - Clean up kernel code structure

**Result:**
- All tests in tests/ (single location)
- Zero broken imports
- Preserved unit test coverage
- Clean kernel code (no embedded tests)

---

## Migration Commands

```bash
# Delete broken parity tests
rm brainsmith/kernels/addstreams/tests/test_addstreams_parity.py
rm brainsmith/kernels/channelwise/tests/test_channelwise_parity.py
rm brainsmith/kernels/duplicate_streams/tests/test_duplicatestreams_parity.py
rm brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary_parity.py

# Migrate unit tests
mv brainsmith/kernels/addstreams/tests/test_addstreams.py \
   tests/unit/test_addstreams_kernel.py
mv brainsmith/kernels/channelwise/tests/test_channelwise.py \
   tests/unit/test_channelwise_kernel.py
mv brainsmith/kernels/duplicate_streams/tests/test_duplicatestreams.py \
   tests/unit/test_duplicate_streams_kernel.py
mv brainsmith/kernels/elementwise_binary/tests/test_elementwise_binary.py \
   tests/unit/test_elementwise_binary_kernel.py

# Delete empty directories
rmdir brainsmith/kernels/addstreams/tests/
rmdir brainsmith/kernels/channelwise/tests/
rmdir brainsmith/kernels/duplicate_streams/tests/
rmdir brainsmith/kernels/elementwise_binary/tests/
```

---

## Testing After Migration

```bash
# Run migrated unit tests
cd /home/tafk/dev/brainsmith-1
pytest tests/unit/ -v

# Verify all kernel tests pass
pytest tests/kernels/ -v
pytest tests/frameworks/ -v
```

---

**Status:** Ready for migration
**Risk:** Low (unit tests are self-contained)
**Impact:** Clean test organization, preserved coverage
