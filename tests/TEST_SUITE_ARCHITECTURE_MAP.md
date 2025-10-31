# Test Suite Architecture Map

**Created**: 2025-10-30
**Purpose**: Comprehensive analysis of test directory structure, redundancy identification, and cleanup recommendations

---

## Executive Summary

The test suite contains **15,584 lines** across **59 Python files** organized into 8 major categories. After Phase 3 completion, we have **both OLD and NEW** test frameworks coexisting, with significant redundancy (estimated **2,000+ lines** of obsolete code ready for deletion).

**Key Findings**:
- ✅ New frameworks work perfectly (SingleKernelTest, DualKernelTest)
- ⚠️ Only 2 tests migrated to new frameworks (AddStreams pilot)
- ❌ Old frameworks still in use (4 base classes, 1,787 lines)
- 🗑️ Ready for deletion: ~2,300 lines once migration complete

---

## Directory Structure & Purpose

```
tests/                          (15,584 total lines)
├── frameworks/        [NEW]    (1,598 lines) - Composition-based test frameworks ✅
├── common/            [SHARED] (1,436 lines) - Reusable utilities (pipelines, validators, executors)
├── parity/            [OLD]    (2,639 lines) - Inheritance-based frameworks ⚠️ LEGACY
├── pipeline/          [MIXED]  (1,365 lines) - Single kernel integration tests
├── dual_pipeline/     [MIXED]  (578 lines)   - Dual kernel parity tests
├── fixtures/                   (2,869 lines) - Test data factories & helpers
├── integration/               (1,161 lines) - DSE/FINN/Hardware integration tests
├── unit/                      (808 lines)   - Registry and component tests
├── utils/                     (597 lines)   - DSE-specific assertion helpers
└── golden/                    (0 lines)     - Empty directory (outputs only) 🗑️
```

---

## Component Inventory & Analysis

### 1. NEW Frameworks (Phase 2) - Keep All ✅

**Directory**: `tests/frameworks/` (1,598 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `kernel_test_base.py` | 205 | ✅ Keep | Minimal config interface (3 abstract methods) |
| `single_kernel_test.py` | 399 | ✅ Keep | Test one kernel vs golden (6 inherited tests) |
| `dual_kernel_test.py` | 726 | ✅ Keep | Test manual vs auto parity (20 inherited tests) |
| `test_addstreams_validation.py` | 268 | ✅ Keep | Meta-tests validating frameworks |
| `__init__.py` | - | ✅ Keep | Package exports |

**Dependencies**:
```
frameworks/
├─→ tests.common.pipeline (PipelineRunner)
├─→ tests.common.validator (GoldenValidator)
├─→ tests.common.executors (PythonExecutor, CppSimExecutor, RTLSimExecutor)
└─→ tests.parity.test_fixtures (make_execution_context) ← NEEDS MIGRATION
```

**Action**: Keep all. These are the target architecture.

---

### 2. Phase 1 Utilities - Keep All ✅

**Directory**: `tests/common/` (1,436 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `executors.py` | 455 | ✅ Keep | Clean executor protocol (Python/cppsim/rtlsim) |
| `pipeline.py` | 201 | ✅ Keep | Unified pipeline runner (single source of truth) |
| `validator.py` | 216 | ✅ Keep | Golden reference validation |
| `tensor_mapping.py` | 204 | ✅ Keep | Tensor name mapping utilities |
| `golden_reference_mixin.py` | 276 | ⚠️ Review | OLD mixin pattern (replaced by GoldenValidator) |
| `assertions.py` | 158 | ✅ Keep | Base AssertionHelper class |
| `constants.py` | 102 | ✅ Keep | Shared test constants |
| `__init__.py` | - | ✅ Keep | Package exports |

**Dependencies**:
```
common/
├─→ tests.parity.assertions (assert_arrays_close) ← CAN BE INLINED
└─→ brainsmith.* (production code)
```

**Redundancy Analysis**:

**`golden_reference_mixin.py` (276 lines)** - ⚠️ **CANDIDATE FOR DELETION**

```python
# OLD pattern (mixin):
class GoldenReferenceMixin:
    def validate_against_golden(self, actual, expected, backend, rtol, atol):
        # Duplicated validation logic
        pass

# NEW pattern (composition):
from tests.common.validator import GoldenValidator
validator = GoldenValidator()
validator.validate(actual, expected, backend, rtol, atol)
```

**Used by**:
- 0 files in new frameworks ✅
- Unknown usage in old tests

**Recommendation**: Delete after verifying no active usage.

---

### 3. OLD Parity Frameworks - Delete After Migration ❌

**Directory**: `tests/parity/` (2,639 lines)

| File | Lines | Status | Purpose | Action |
|------|-------|--------|---------|--------|
| `base_parity_test.py` | 1,204 | ❌ Delete | Massive abstract base (25 test methods) | Delete after migration |
| `executors.py` | 497 | ❌ Delete | OLD BackendExecutor (SRP violation) | Replaced by common/executors.py |
| `computational_parity_test.py` | 418 | ❌ Delete | Execution parity tests | Merged into DualKernelTest |
| `core_parity_test.py` | 410 | ❌ Delete | 7 structural parity tests | Merged into DualKernelTest |
| `hls_codegen_parity.py` | 396 | ⚠️ Review | HLS codegen-specific tests | May have unique value |
| `hw_estimation_parity_test.py` | 332 | ❌ Delete | 5 HW estimation tests | Merged into DualKernelTest |
| `assertions.py` | 346 | ⚠️ Keep | ParityAssertion helper | Used by new frameworks |
| `backend_helpers.py` | 221 | ⚠️ Review | HLS backend setup utilities | May be needed |
| `test_fixtures.py` | 136 | ✅ Keep | make_execution_context | Used by new frameworks |
| `__init__.py` | - | ⚠️ Review | Package exports | Update after cleanup |

**Total for Deletion**: 1,204 + 497 + 418 + 410 + 332 = **2,861 lines**

**Dependencies (Who uses OLD frameworks)**:
```
OLD parity frameworks used by:
├─→ tests/dual_pipeline/dual_pipeline_parity_test_v2.py (320 lines) ← OLD framework
├─→ tests/pipeline/base_integration_test.py (721 lines) ← OLD framework
└─→ Unknown kernel tests (not yet inventoried)
```

**Recommendation**:
1. Migrate remaining kernel tests to new frameworks
2. Delete old base classes: `base_parity_test.py`, `core_parity_test.py`, `hw_estimation_parity_test.py`, `computational_parity_test.py`
3. Delete old executors: `parity/executors.py` (replaced by `common/executors.py`)
4. **Keep**: `test_fixtures.py`, `assertions.py`, `backend_helpers.py` (used by new frameworks)
5. Review `hls_codegen_parity.py` for unique test coverage

---

### 4. Pipeline Tests - Migrate ⚠️

**Directory**: `tests/pipeline/` (1,365 lines)

| File | Lines | Status | Purpose | Action |
|------|-------|--------|---------|--------|
| `base_integration_test.py` | 721 | ❌ Delete | OLD IntegratedPipelineTest | Replaced by SingleKernelTest |
| `test_addstreams_integration.py` | 644 | ✅ Migrated | AddStreams tests (4 test classes) | Using SingleKernelTest ✅ |
| `conftest.py` | - | ✅ Keep | Pipeline-specific fixtures | - |
| `__init__.py` | - | ✅ Keep | Package exports | - |

**Migration Status**:
- ✅ AddStreams: Migrated to SingleKernelTest
- ❌ Other kernels: Not yet migrated

**Action**:
1. Migrate remaining kernel tests (if any exist)
2. Delete `base_integration_test.py` (721 lines)

---

### 5. Dual Pipeline Tests - Migrate ⚠️

**Directory**: `tests/dual_pipeline/` (578 lines)

| File | Lines | Status | Purpose | Action |
|------|-------|--------|---------|--------|
| `dual_pipeline_parity_test_v2.py` | 320 | ❌ Delete | Diamond inheritance framework | Replaced by DualKernelTest |
| `test_addstreams_v2.py` | 258 | ✅ Migrated | AddStreams dual tests | Using DualKernelTest ✅ |
| `__init__.py` | - | ✅ Keep | Package exports | - |

**Migration Status**:
- ✅ AddStreams: Migrated to DualKernelTest
- ❌ Other kernels: Unknown if any exist

**Action**:
1. Search for other dual pipeline tests
2. Delete `dual_pipeline_parity_test_v2.py` (320 lines)

---

### 6. Fixtures - Keep All ✅

**Directory**: `tests/fixtures/` (2,869 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `kernel_test_helpers.py` | 1,041 | ✅ Keep | Kernel test utilities |
| `blueprints.py` | 824 | ✅ Keep | Blueprint fixtures for DSE tests |
| `test_kernel_test_helpers.py` | 413 | ✅ Keep | Tests for helpers |
| `models.py` | 344 | ✅ Keep | ONNX model factories |
| `components/steps.py` | 166 | ✅ Keep | DSE step fixtures |
| `design_spaces.py` | 124 | ✅ Keep | Design space fixtures |
| Others | - | ✅ Keep | Various fixtures |

**Action**: Keep all. These are data factories and test utilities, not frameworks.

---

### 7. Integration Tests - Keep All ✅

**Directory**: `tests/integration/` (1,161 lines)

| Subdirectory | Lines | Purpose | Status |
|-------------|-------|---------|--------|
| `fast/` | 683 | Blueprint parsing, design space, tree construction | ✅ Keep |
| `finn/` | 836 | Pipeline, segment execution, cache behavior | ✅ Keep |
| `hardware/` | - | Bitfile generation (slow) | ✅ Keep |
| `rtl/` | - | RTL generation tests | ✅ Keep |

**Action**: Keep all. These test the DSE system, not kernel frameworks.

---

### 8. Unit Tests - Keep All ✅

**Directory**: `tests/unit/` (808 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `test_registry_edge_cases.py` | 808 | Registry system tests | ✅ Keep |

**Action**: Keep all. Unit tests for registry system.

---

### 9. Utils - Keep All ✅

**Directory**: `tests/utils/` (597 lines)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `assertions.py` | 597 | DSE-specific assertions (TreeAssertions, etc.) | ✅ Keep |

**Action**: Keep all. DSE test utilities, not duplication of parity assertions.

---

### 10. Golden - Delete 🗑️

**Directory**: `tests/golden/` (0 lines)

**Contents**: Empty directory with only `outputs/` subdirectory

**Action**: Delete empty directory. Golden reference computation is now in test methods, not files.

---

## Dependency Graph

### Current Dependencies (Mixed Old/New)

```
NEW Framework Stack:
frameworks/single_kernel_test.py
frameworks/dual_kernel_test.py
├─→ frameworks/kernel_test_base.py
├─→ common/pipeline.py (PipelineRunner)
├─→ common/validator.py (GoldenValidator)
├─→ common/executors.py (PythonExecutor, CppSimExecutor, RTLSimExecutor)
├─→ parity/test_fixtures.py (make_execution_context) ← CROSS-BOUNDARY
└─→ parity/assertions.py (assert_shapes_match, etc.) ← CROSS-BOUNDARY

OLD Framework Stack:
parity/base_parity_test.py
├─→ parity/core_parity_test.py
├─→ parity/hw_estimation_parity_test.py
├─→ parity/computational_parity_test.py
├─→ parity/executors.py (OLD BackendExecutor)
├─→ parity/assertions.py
├─→ parity/test_fixtures.py
└─→ parity/backend_helpers.py

pipeline/base_integration_test.py
├─→ parity/executors.py (OLD)
├─→ parity/test_fixtures.py
├─→ parity/assertions.py
└─→ parity/backend_helpers.py

dual_pipeline/dual_pipeline_parity_test_v2.py
├─→ parity/core_parity_test.py
├─→ parity/hw_estimation_parity_test.py
├─→ parity/executors.py (OLD)
├─→ parity/test_fixtures.py
└─→ parity/backend_helpers.py
```

### Cross-Boundary Dependencies (Need Resolution)

**New frameworks depend on old parity utilities**:
1. `tests.parity.test_fixtures.make_execution_context`
   - **Solution**: Move to `tests.common.test_fixtures`

2. `tests.parity.assertions` (ParityAssertion, assert_shapes_match, etc.)
   - **Solution**: Keep in parity (domain-specific) OR move to common if widely used

3. `tests.parity.backend_helpers.setup_hls_backend_via_specialize`
   - **Solution**: Keep in parity (HLS-specific) OR move to common.executors

---

## Redundancy Identification

### 1. Executor Duplication ✅ RESOLVED

**OLD**: `tests/parity/executors.py` (497 lines)
- BackendExecutor with `execute_and_compare()` (SRP violation)
- Mixed execution + comparison logic

**NEW**: `tests/common/executors.py` (455 lines)
- Clean Executor protocol (execute only)
- Single Responsibility Principle

**Status**: NEW replaces OLD entirely. OLD can be deleted after migration.

---

### 2. Pipeline Duplication ✅ RESOLVED

**Before**: Duplicated 5 times across test files (90% similarity)
- `CoreParityTest.run_manual_pipeline()` (50 lines)
- `CoreParityTest.run_auto_pipeline()` (50 lines)
- `HWEstimationParityTest.run_manual_pipeline()` (50 lines)
- `HWEstimationParityTest.run_auto_pipeline()` (50 lines)
- `IntegratedPipelineTest.run_inference_pipeline()` (55 lines)
- **Total**: ~255 lines of near-duplicate code

**After**: Single source of truth
- `tests.common.pipeline.PipelineRunner` (201 lines)

**Status**: RESOLVED. Old duplicates can be deleted after migration.

---

### 3. Validation Duplication ✅ RESOLVED

**Before**: Duplicated 3 times
- `GoldenReferenceMixin.validate_against_golden()` (in multiple classes)
- `IntegratedPipelineTest.validate_against_golden()`
- Inline validation in test methods

**After**: Single validator
- `tests.common.validator.GoldenValidator` (216 lines)

**Status**: RESOLVED. `common/golden_reference_mixin.py` (276 lines) is now obsolete.

---

### 4. Framework Test Duplication ✅ RESOLVED

**Before**: 4 separate base classes with overlapping tests
- `base_parity_test.py` (1,204 lines) - 25 test methods
- `core_parity_test.py` (410 lines) - 7 parity tests
- `hw_estimation_parity_test.py` (332 lines) - 5 estimation tests
- `computational_parity_test.py` (418 lines) - 8 execution tests
- **Total**: 2,364 lines with ~40% overlap

**After**: 2 focused frameworks
- `single_kernel_test.py` (399 lines) - 6 tests
- `dual_kernel_test.py` (726 lines) - 20 tests
- **Total**: 1,125 lines (52% reduction)

**Status**: RESOLVED. Old frameworks can be deleted after migration.

---

### 5. Assertion Helper Organization ✅ WELL-ORGANIZED

**NO DUPLICATION** - Well-layered architecture:

```
common/assertions.py (158 lines)
└─ AssertionHelper (base class)
   ├─→ parity/assertions.py (346 lines)
   │   └─ ParityAssertion (manual vs auto comparisons)
   └─→ utils/assertions.py (597 lines)
       ├─ TreeAssertions (DSE tree validation)
       ├─ ExecutionAssertions (DSE execution validation)
       └─ BlueprintAssertions (blueprint parsing validation)
```

**Status**: KEEP ALL. Clear separation of concerns.

---

## Deletion Candidates (After Migration Complete)

### High Priority (Core Frameworks) - 2,861 lines

| File | Lines | Why Delete | Blocks |
|------|-------|-----------|--------|
| `parity/base_parity_test.py` | 1,204 | Massive monolithic base class | Old test migrations |
| `parity/executors.py` | 497 | Replaced by `common/executors.py` | Old test migrations |
| `parity/computational_parity_test.py` | 418 | Merged into DualKernelTest | Old test migrations |
| `parity/core_parity_test.py` | 410 | Merged into DualKernelTest | Old test migrations |
| `parity/hw_estimation_parity_test.py` | 332 | Merged into DualKernelTest | Old test migrations |

### Medium Priority (Test Base Classes) - 1,041 lines

| File | Lines | Why Delete | Blocks |
|------|-------|-----------|--------|
| `pipeline/base_integration_test.py` | 721 | Replaced by SingleKernelTest | Pipeline test migrations |
| `dual_pipeline/dual_pipeline_parity_test_v2.py` | 320 | Replaced by DualKernelTest | Dual pipeline test migrations |

### Low Priority (Unused Utilities) - ~300 lines

| File | Lines | Why Delete | Blocks |
|------|-------|-----------|--------|
| `common/golden_reference_mixin.py` | 276 | Replaced by GoldenValidator | Verify zero usage |
| `golden/` directory | 0 | Empty, unused | None |

### Review Before Deletion

| File | Lines | Action | Reason |
|------|-------|--------|--------|
| `parity/hls_codegen_parity.py` | 396 | Review | May contain unique HLS-specific tests |
| `parity/backend_helpers.py` | 221 | Review | May be needed by new frameworks |

---

## Total Impact Summary

### Code Volume Analysis

**Before Refactor** (estimated active test framework code):
```
Old Frameworks:     2,861 lines (parity base classes)
Old Integration:    1,041 lines (pipeline/dual_pipeline bases)
Old Utilities:       276 lines (golden_reference_mixin)
Total OLD:          4,178 lines
```

**After Refactor**:
```
New Frameworks:     1,125 lines (single + dual kernel tests)
New Utilities:       872 lines (pipeline + validator + executors)
Total NEW:          1,997 lines
```

**Reduction**: 4,178 → 1,997 lines = **52% code reduction** (2,181 lines deleted)

**Additional Deletions** (after migration):
```
Empty directories:      (golden/)
Obsolete mixins:    276 lines (golden_reference_mixin.py)
Total deletable:    2,457 lines
```

---

## Migration Roadmap

### Phase 3 Status: AddStreams Complete ✅

**Migrated**:
- ✅ `tests/pipeline/test_addstreams_integration.py` → SingleKernelTest
- ✅ `tests/dual_pipeline/test_addstreams_v2.py` → DualKernelTest

**Impact**: 2 files migrated, 902 lines of test code now using new frameworks

---

### Phase 4: Migrate Remaining Kernels

**Unknown Kernel Tests** (need inventory):
```bash
# Find all test files using OLD frameworks
grep -r "IntegratedPipelineTest\|CoreParityTest\|HWEstimationParityTest\|base_parity_test" tests/ --include="test_*.py"
```

**Estimated Scope**:
- ElementwiseBinary (Add, Mul, Sub) - 3 kernels
- StreamingFCLayer - 1 kernel
- VectorVectorActivation - 1 kernel
- Thresholding - 1 kernel
- Unknown additional kernels

**Estimated Duration**: 1-2 weeks (based on AddStreams taking 2 hours)

---

### Phase 5: Delete Old Frameworks

**Only after ALL tests migrated**:

1. **Delete old base classes** (2,861 lines):
   - `parity/base_parity_test.py`
   - `parity/core_parity_test.py`
   - `parity/hw_estimation_parity_test.py`
   - `parity/computational_parity_test.py`
   - `parity/executors.py`

2. **Delete old test base classes** (1,041 lines):
   - `pipeline/base_integration_test.py`
   - `dual_pipeline/dual_pipeline_parity_test_v2.py`

3. **Delete obsolete utilities** (276 lines):
   - `common/golden_reference_mixin.py`

4. **Delete empty directories**:
   - `tests/golden/`

5. **Review and potentially delete** (617 lines):
   - `parity/hls_codegen_parity.py` (if merged into new frameworks)
   - `parity/backend_helpers.py` (if functionality moved to common)

**Total Potential Deletion**: 4,795 lines (31% of test code)

---

## Cleanup Actions by Priority

### Immediate (Can Do Now)

1. ✅ Delete `tests/golden/` directory (empty)
2. ⚠️ Verify `common/golden_reference_mixin.py` has zero usage
3. ⚠️ Document cross-boundary dependencies (parity → common)

### Short-term (After Next 2-3 Kernel Migrations)

1. Move `parity/test_fixtures.py` → `common/test_fixtures.py`
2. Evaluate `parity/backend_helpers.py` usage
3. Update imports in new frameworks

### Long-term (After All Migrations Complete)

1. Delete all OLD framework base classes (2,861 lines)
2. Delete old pipeline/dual_pipeline bases (1,041 lines)
3. Delete obsolete utilities (276 lines)
4. Archive or delete `parity/hls_codegen_parity.py` if redundant
5. Update documentation and remove legacy references

---

## Recommendations

### High Priority

1. **Inventory remaining kernel tests**
   - Search for all tests using OLD frameworks
   - Create migration plan for each kernel

2. **Resolve cross-boundary dependencies**
   - Move `test_fixtures.py` to common
   - Clarify `backend_helpers.py` ownership

3. **Delete empty directories immediately**
   - Remove `tests/golden/` (no contents)

### Medium Priority

4. **Continue kernel migrations**
   - Use AddStreams as template (2 hours per kernel)
   - Maintain 100% test pass rate

5. **Document migration patterns**
   - Create "How to Migrate" guide
   - Include common pitfalls (exec_mode, golden reference)

### Low Priority

6. **Final cleanup phase**
   - Delete old frameworks (4,178 lines)
   - Update all documentation
   - Run full test suite validation

---

## Risk Assessment

### Low Risk (Safe to Delete Now)

- ✅ `tests/golden/` - Empty directory
- ⚠️ `common/golden_reference_mixin.py` - If zero usage confirmed

### Medium Risk (Delete After Migration)

- ⚠️ All OLD framework base classes - After all tests migrated
- ⚠️ Old pipeline bases - After pipeline tests migrated

### High Risk (Requires Careful Review)

- 🔴 `parity/hls_codegen_parity.py` - May have unique test coverage
- 🔴 `parity/backend_helpers.py` - May be needed by new frameworks
- 🔴 `parity/test_fixtures.py` - Currently used by new frameworks

---

## Next Steps

1. **Immediate**: Delete `tests/golden/` directory ✅
2. **This week**: Inventory all tests using OLD frameworks
3. **Next sprint**: Migrate 2-3 more kernels to new frameworks
4. **After all migrations**: Execute Phase 5 cleanup (delete 4,795 lines)

---

**Conclusion**: Test suite is in transition from OLD (inheritance-based) to NEW (composition-based) architecture. With AddStreams successfully migrated, we have a clear template for remaining kernels. Once all migrations complete, we can safely delete **~4,800 lines** of obsolete code (31% reduction).
