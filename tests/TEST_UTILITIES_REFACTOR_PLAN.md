# Test Utilities Refactor Plan

**Date**: 2025-10-31
**Purpose**: Consolidate test utilities for better organization, discoverability, and maintainability
**Priority**: Medium (Quality of life improvement, no functional changes)

---

## Executive Summary

Current test utilities are spread across 3 directories (`parity/`, `utils/`, `common/`) with **confusing names** and **poor separation of concerns**. Recommendation: **Consolidate into single `support/` directory** organized by purpose, matching industry standards and improving discoverability.

**Impact**:
- Eliminate 2 confusing directories (parity, utils)
- Reduce from 3 locations to 1 for test utilities
- Clear naming: tests/support/ = "test support code"
- Minimal changes: Just file moves and import updates

---

## Current State Analysis

### Directory Structure (3 directories, 9 files, 2,418 lines)

#### tests/parity/ (484 lines + __init__)
**Purpose**: Kernel parity testing utilities (Manual vs Auto comparison)
**Problem**: Name "parity" is misleading - sounds like generic equality testing, but is kernel-specific

Files:
- `assertions.py` (347 lines) - ParityAssertion + specialized helpers
  - assert_shapes_match, assert_datatypes_match, assert_widths_match
  - assert_values_match, assert_arrays_close, assert_model_tensors_match
- `test_fixtures.py` (137 lines) - make_execution_context() for random test data

**Used by**: DualKernelTest, SingleKernelTest, executors

#### tests/utils/ (598 lines + __init__)
**Purpose**: DSE-specific assertion utilities
**Problem**: Name "utils" is too generic, but content is DSE-specific (tree, execution, blueprints)

Files:
- `assertions.py` (598 lines) - TreeAssertions, ExecutionAssertions, BlueprintAssertions
  - Tree structure validation (nodes, leaves, branches, execution order)
  - Execution result validation (stats, segment status, timing)
  - Blueprint parsing validation (design space, config, inheritance)

**Used by**: DSE integration tests, tree building tests

#### tests/common/ (1,336 lines + __init__)
**Purpose**: Shared utilities across all tests
**Problem**: Unclear distinction from "utils", name doesn't convey "test support"

Files:
- `assertions.py` (159 lines) - AssertionHelper base class
- `constants.py` (103 lines) - All test constants (DSE, parity, data generation)
- `executors.py` (455 lines) - PythonExecutor, CppSimExecutor, RTLSimExecutor
- `pipeline.py` (201 lines) - PipelineRunner for transformation execution
- `validator.py` (216 lines) - GoldenValidator for output comparison
- `tensor_mapping.py` (204 lines) - ONNX to golden tensor name mapping

**Used by**: All test frameworks (kernel + DSE)

---

## Problems Identified

### 1. Confusing Directory Names ⚠️

| Directory | Name Suggests | Actually Contains | Users Think |
|-----------|---------------|-------------------|-------------|
| `parity/` | Generic equality testing | Kernel parity utilities (Manual vs Auto) | "Is this for any parity test?" |
| `utils/` | General utilities | DSE-specific assertions | "Is this shared across all tests?" |
| `common/` | Shared utilities | Shared utilities (correct!) | "How is this different from utils?" |

**Impact**: New contributors waste time searching wrong directories

### 2. Poor Separation of Concerns ⚠️

**Problem**: Related functionality split across directories by arbitrary boundaries

Example 1: **Assertions split across 3 files**
```
tests/common/assertions.py       # Base: AssertionHelper
tests/parity/assertions.py       # Kernel: ParityAssertion + helpers
tests/utils/assertions.py        # DSE: TreeAssertions, ExecutionAssertions, BlueprintAssertions
```

Why? All are assertions. Should be together or clearly domain-separated.

Example 2: **Circular dependency smell**
```
tests/parity/test_fixtures.py    # make_execution_context()
     ↓ (used by)
tests/common/executors.py         # Executors
     ↓ (imported by)
tests/frameworks/single_kernel_test.py  # Uses both
```

`make_execution_context()` is in "parity" but used by "common" executors. Should be in common or executors.

### 3. Against Industry Standards ⚠️

Python test suite organization (pytest, Django, Flask):

**Standard patterns:**
```
tests/
  conftest.py           # Fixtures
  support/              # OR helpers/, test_utils/, testing_utils/
    assertions.py
    builders.py
    executors.py
  test_*.py             # Actual tests
```

**NOT standard:**
```
tests/
  parity/               # Domain-specific utils (non-standard)
  utils/                # Too generic
  common/               # Redundant with utils
```

**Examples from popular projects:**
- pytest: `testing/` directory for test support code
- Django: `tests/test_utils/` for test helpers
- Flask: `tests/helpers.py` for test utilities
- numpy: `numpy/testing/` for test support

### 4. Poor Discoverability ⚠️

**Current**: "I need to validate model outputs, where do I look?"
- ❌ Is it in parity/? (no)
- ❌ Is it in utils/? (no)
- ✅ It's in common/validator.py (found after checking 3 places)

**Better**: "I need to validate outputs"
- ✅ Look in tests/support/validator.py (obvious location)

---

## Recommended Solution

### Consolidate into `tests/support/` Directory

**Rationale**:
1. ✅ **Clear naming**: "support" clearly means "test support code"
2. ✅ **Industry standard**: Matches pytest/Django/Flask patterns
3. ✅ **Single location**: One place to find all test utilities
4. ✅ **Logical grouping**: Assertions together, executors together, etc.
5. ✅ **Easy discovery**: "I need X" → look in support/
6. ✅ **Fewer components**: 1 directory instead of 3

### Proposed Structure

```
tests/
  support/              # All test support code (replaces common/, parity/, utils/)
    assertions.py       # ALL assertion classes (Base + Kernel + DSE)
    constants.py        # All test constants
    executors.py        # All executors (Python, CppSim, RTLSim)
    pipeline.py         # PipelineRunner
    validator.py        # GoldenValidator
    context.py          # make_execution_context() [renamed from test_fixtures.py]
    tensor_mapping.py   # ONNX to golden name mapping
    __init__.py         # Exports key classes for convenient imports

  fixtures/             # Test data builders (keep as-is)
    kernel_test_helpers.py
    models.py
    design_spaces.py
    blueprints.py

  frameworks/           # Test frameworks (keep as-is)
    single_kernel_test.py
    dual_kernel_test.py
```

### What Changes

#### 1. File Consolidation

**assertions.py** - Merge 3 files into 1:
```python
# tests/support/assertions.py (combined from 3 files)

"""Test assertion utilities.

Provides:
- AssertionHelper: Base class with consistent error formatting
- ParityAssertion: Kernel parity testing (Manual vs Auto)
- TreeAssertions: DSE tree structure validation
- ExecutionAssertions: DSE execution result validation
- BlueprintAssertions: DSE blueprint parsing validation
- Specialized helpers: assert_shapes_match, assert_arrays_close, etc.
"""

# Base class (from tests/common/assertions.py)
class AssertionHelper:
    ...

# Kernel testing (from tests/parity/assertions.py)
class ParityAssertion(AssertionHelper):
    ...

def assert_shapes_match(...):
    ...

# DSE testing (from tests/utils/assertions.py)
class TreeAssertions(AssertionHelper):
    ...

class ExecutionAssertions(AssertionHelper):
    ...

class BlueprintAssertions(AssertionHelper):
    ...
```

**Benefits**:
- All assertions in one place
- Clear organization: base → domain-specific → helpers
- Single import: `from tests.support.assertions import ParityAssertion, TreeAssertions`

**context.py** - Rename test_fixtures.py:
```python
# tests/support/context.py (from tests/parity/test_fixtures.py)

"""Test execution context generation.

Provides make_execution_context() for generating random test data
based on operator specifications.
"""

def make_execution_context(model, op, seed=None):
    ...
```

**Benefits**:
- Clearer name: "context" conveys purpose better than "test_fixtures"
- Logical location: Next to executors that use it
- Single source of truth for test data generation

**Other files** - Move as-is:
- tests/common/*.py → tests/support/*.py (rename directory)
- tests/utils/assertions.py → merge into tests/support/assertions.py
- tests/parity/assertions.py → merge into tests/support/assertions.py
- tests/parity/test_fixtures.py → tests/support/context.py (rename)

#### 2. Import Updates

**Old imports:**
```python
# Scattered across 3 directories
from tests.common.assertions import AssertionHelper
from tests.parity.assertions import ParityAssertion, assert_shapes_match
from tests.utils.assertions import TreeAssertions, ExecutionAssertions
from tests.common.executors import CppSimExecutor
from tests.common.validator import GoldenValidator
from tests.parity.test_fixtures import make_execution_context
```

**New imports:**
```python
# Single directory, clear purpose
from tests.support.assertions import (
    AssertionHelper, ParityAssertion, TreeAssertions, ExecutionAssertions,
    assert_shapes_match
)
from tests.support.executors import CppSimExecutor
from tests.support.validator import GoldenValidator
from tests.support.context import make_execution_context
```

**OR with convenience __init__.py:**
```python
# Even simpler
from tests.support import (
    ParityAssertion, TreeAssertions, CppSimExecutor,
    GoldenValidator, make_execution_context
)
```

#### 3. Affected Files (~15-20 files need import updates)

**Test frameworks:**
- tests/frameworks/single_kernel_test.py
- tests/frameworks/dual_kernel_test.py

**Integration tests:**
- tests/pipeline/test_addstreams_integration.py
- tests/dual_pipeline/test_addstreams_v2.py
- tests/integration/finn/*.py (if any use these)

**Support modules:**
- tests/support/executors.py (internal imports)
- tests/support/validator.py (internal imports)
- tests/fixtures/*.py (if they import assertions)

---

## Implementation Plan

### Phase 1: Create tests/support/ and Merge Files

**Step 1.1**: Create directory and move simple files
```bash
mkdir tests/support
mv tests/common/constants.py tests/support/
mv tests/common/executors.py tests/support/
mv tests/common/pipeline.py tests/support/
mv tests/common/validator.py tests/support/
mv tests/common/tensor_mapping.py tests/support/
```

**Step 1.2**: Rename test_fixtures.py → context.py
```bash
mv tests/parity/test_fixtures.py tests/support/context.py
# Update docstring to reflect new name
```

**Step 1.3**: Merge assertions.py (3 files → 1)
```python
# Create tests/support/assertions.py with contents:
# 1. Base class from tests/common/assertions.py
# 2. Kernel classes from tests/parity/assertions.py
# 3. DSE classes from tests/utils/assertions.py
```

**Step 1.4**: Create convenience __init__.py
```python
# tests/support/__init__.py
"""Test support utilities - assertions, executors, validators, context."""

# Export commonly used classes for convenient imports
from .assertions import (
    AssertionHelper,
    ParityAssertion,
    TreeAssertions,
    ExecutionAssertions,
    BlueprintAssertions,
    assert_shapes_match,
    assert_datatypes_match,
    assert_widths_match,
    assert_arrays_close,
)
from .executors import PythonExecutor, CppSimExecutor, RTLSimExecutor
from .validator import GoldenValidator, TolerancePresets
from .pipeline import PipelineRunner
from .context import make_execution_context
from .constants import *

__all__ = [
    # Assertions
    "AssertionHelper",
    "ParityAssertion",
    "TreeAssertions",
    "ExecutionAssertions",
    "BlueprintAssertions",
    "assert_shapes_match",
    "assert_datatypes_match",
    "assert_widths_match",
    "assert_arrays_close",
    # Execution
    "PythonExecutor",
    "CppSimExecutor",
    "RTLSimExecutor",
    "PipelineRunner",
    "make_execution_context",
    # Validation
    "GoldenValidator",
    "TolerancePresets",
]
```

### Phase 2: Update Imports

**Step 2.1**: Update test frameworks
```bash
# Find and replace in frameworks/
sed -i 's/from tests\.common\./from tests.support./g' tests/frameworks/*.py
sed -i 's/from tests\.parity\./from tests.support./g' tests/frameworks/*.py
sed -i 's/test_fixtures import/context import/g' tests/frameworks/*.py
```

**Step 2.2**: Update integration tests
```bash
# Find all files importing from old locations
grep -r "from tests.common" tests --include="*.py" -l | xargs sed -i 's/from tests\.common\./from tests.support./g'
grep -r "from tests.parity" tests --include="*.py" -l | xargs sed -i 's/from tests\.parity\./from tests.support./g'
grep -r "from tests.utils" tests --include="*.py" -l | xargs sed -i 's/from tests\.utils\./from tests.support./g'
```

**Step 2.3**: Update test_fixtures → context renames
```bash
grep -r "test_fixtures import" tests --include="*.py" -l | xargs sed -i 's/test_fixtures import/context import/g'
```

### Phase 3: Cleanup

**Step 3.1**: Delete old directories
```bash
rm -rf tests/common
rm -rf tests/parity  # After moving files
rm -rf tests/utils
```

**Step 3.2**: Run tests to verify
```bash
pytest tests/ -v
```

**Step 3.3**: Update documentation
- Update TIER1_DELETION_SUMMARY.md to mention this refactor
- Update TEST_SUITE_ARCHITECTURE_MAP.md with new structure
- Update any README files referencing old structure

### Phase 4: Validation

**Step 4.1**: Verify zero regressions
```bash
pytest tests/ --tb=short
# Expect same pass/fail/skip counts as before refactor
```

**Step 4.2**: Check import errors
```bash
# Should find no references to old locations
grep -r "from tests\.common" tests --include="*.py"
grep -r "from tests\.parity" tests --include="*.py"
grep -r "from tests\.utils" tests --include="*.py"
```

**Step 4.3**: Verify convenience imports work
```python
# Should work cleanly
from tests.support import ParityAssertion, CppSimExecutor, GoldenValidator
```

---

## Benefits Analysis

### Before Refactor

**Directory structure:**
```
tests/
  common/          # "Shared utilities" (vague)
  parity/          # "Parity utilities" (misleading - kernel-specific)
  utils/           # "General utilities" (misleading - DSE-specific)
```

**Developer experience:**
- ❌ "Where do I find assertion helpers?" → Check 3 directories
- ❌ "What's the difference between common and utils?" → Unclear
- ❌ "Is parity for any equality test?" → No, kernel-specific
- ❌ Split assertions across 3 files (hard to navigate)

**Import complexity:**
```python
from tests.common.assertions import AssertionHelper
from tests.parity.assertions import assert_shapes_match
from tests.utils.assertions import TreeAssertions
```

### After Refactor

**Directory structure:**
```
tests/
  support/         # "Test support code" (clear, standard)
```

**Developer experience:**
- ✅ "Where do I find assertion helpers?" → tests/support/assertions.py
- ✅ "Where are executors?" → tests/support/executors.py
- ✅ "Where's the validator?" → tests/support/validator.py
- ✅ All assertions in one file (easy navigation)

**Import simplicity:**
```python
from tests.support import ParityAssertion, TreeAssertions, assert_shapes_match
```

### Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Utility directories | 3 | 1 | -67% |
| Assertion files | 3 | 1 | -67% |
| Lines of code | 2,418 | 2,418 | 0 (just reorganization) |
| Import locations | 3 | 1 | -67% |
| Files to check for utilities | 9 across 3 dirs | 7 in 1 dir | Clearer |
| Time to find utilities | ~30s (check 3 places) | ~5s (one place) | -83% |

---

## Alternative Approaches Considered

### Alternative 1: Consolidate by Domain

```
tests/
  kernel_support/    # Everything for kernel testing
    assertions.py
    context.py
    executors.py
  dse_support/      # Everything for DSE testing
    assertions.py
  shared/           # Truly shared (constants, base classes)
    assertions.py
    constants.py
```

**Pros**: Clear domain boundaries
**Cons**:
- Still 3 directories (similar to current)
- Duplication potential (executors in kernel_support, but DSE also uses them?)
- Doesn't solve "where do I look?" problem

**Verdict**: ❌ Doesn't improve discoverability enough

### Alternative 2: Minimal Changes (Just Rename)

```
tests/
  kernel/           # Rename parity/ → kernel/
  dse/             # Rename utils/ → dse/
  shared/          # Rename common/ → shared/
```

**Pros**: Minimal changes, low risk
**Cons**:
- Still 3 directories
- Still have "where do I look?" problem
- Doesn't match industry standards

**Verdict**: ⚠️ Better names, but doesn't solve organizational issues

### Alternative 3: Leave As-Is

**Pros**: No work required
**Cons**: All problems remain

**Verdict**: ❌ Technical debt accumulates

---

## Risks and Mitigations

### Risk 1: Import errors after refactor
**Severity**: Medium
**Mitigation**:
- Grep for all old imports before deleting directories
- Run full test suite after each phase
- Use automated find/replace for consistency

### Risk 2: Breaking changes for external users
**Severity**: Low (internal test utilities, not public API)
**Mitigation**:
- This is internal test infrastructure
- Not imported by brainsmith/ production code
- Only affects test files

### Risk 3: Git history obscured
**Severity**: Low
**Mitigation**:
- Use `git mv` for file moves (preserves history)
- Document refactor in commit message
- Old files still accessible in git history

### Risk 4: Merge conflicts with open PRs
**Severity**: Low
**Mitigation**:
- Coordinate with team before executing
- Automated find/replace is easy to reapply
- Consider timing (after releases, before major work)

---

## Success Criteria

✅ **Complete when:**
1. Single tests/support/ directory contains all test utilities
2. No references to tests/common, tests/parity, tests/utils in code
3. All tests pass with same pass/fail/skip counts
4. Imports simplified (can use `from tests.support import ...`)
5. Documentation updated to reflect new structure

✅ **Quality metrics:**
- Zero breaking changes (same test results)
- Clearer code organization (single support/ directory)
- Easier discoverability (one place to look)
- Matches industry standards (support/ pattern)

---

## Recommendation

**Execute this refactor** with the following priorities:

**Priority 1 (High Value, Low Risk)**:
- Create tests/support/ directory
- Move tests/common/* → tests/support/*
- Update imports (straightforward find/replace)

**Priority 2 (High Value, Medium Effort)**:
- Merge assertions.py (3 files → 1)
- Rename test_fixtures.py → context.py
- Update imports in frameworks/

**Priority 3 (Polish)**:
- Create convenience __init__.py for clean imports
- Update all documentation
- Clean up empty directories

**Timeline**: 1-2 hours for Priority 1-2, 30 minutes for Priority 3

**Best time to execute**:
- After current work completes
- Before starting new major features
- When test suite is stable

---

## Appendix: Import Migration Guide

### For Test Framework Developers

**Old:**
```python
from tests.common.assertions import AssertionHelper
from tests.parity.assertions import ParityAssertion, assert_shapes_match
from tests.common.executors import CppSimExecutor
from tests.common.validator import GoldenValidator
from tests.parity.test_fixtures import make_execution_context
```

**New (explicit):**
```python
from tests.support.assertions import AssertionHelper, ParityAssertion, assert_shapes_match
from tests.support.executors import CppSimExecutor
from tests.support.validator import GoldenValidator
from tests.support.context import make_execution_context
```

**New (convenient):**
```python
from tests.support import (
    ParityAssertion, assert_shapes_match,
    CppSimExecutor, GoldenValidator, make_execution_context
)
```

### For Test Writers

**Old:**
```python
from tests.utils.assertions import TreeAssertions, ExecutionAssertions
from tests.common.constants import DSE_DEFAULT_CLOCK_PERIOD_NS
```

**New:**
```python
from tests.support.assertions import TreeAssertions, ExecutionAssertions
from tests.support.constants import DSE_DEFAULT_CLOCK_PERIOD_NS
```

**OR:**
```python
from tests.support import TreeAssertions, ExecutionAssertions, DSE_DEFAULT_CLOCK_PERIOD_NS
```

---

**Status**: ⏸️ **AWAITING APPROVAL** - Ready for execution after review
