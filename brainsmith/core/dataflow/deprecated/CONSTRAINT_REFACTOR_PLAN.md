# Dimension Constraints Refactoring Plan

## Goal
Reduce `dimension_constraints.py` from 750 lines to ~200 lines by eliminating DRY violations and achieving Arete.

**Target: 77% code reduction** while maintaining full functionality and test coverage.

---

## Phase 1: Extract Helper Functions (Day 1, 2-3 hours)

### Objective
Eliminate ~400 lines of duplicated helper logic across all constraint classes.

### Steps

#### 1.1 Create Helper Functions (30 min)
Add to top of `dimension_constraints.py` after imports:

```python
# ===========================================================================
# Helper Functions (DRY)
# ===========================================================================

def _get_interface(
    context: Dict[str, Any],
    name: str
) -> Tuple[Any, Optional[ConstraintViolation]]:
    """Get interface from context with error handling.

    Returns:
        (interface, None) on success
        (None, violation) on error
    """
    if name not in context:
        violation = ConstraintViolation(
            constraint_type="constraint_evaluation",
            message=f"Interface '{name}' not found in context",
            severity="error"
        )
        return None, violation
    return context[name], None


def _get_dimension_value(
    interface: Any,
    dim_index: Optional[int],
    interface_name: str
) -> Tuple[Optional[int], Optional[str], Optional[ConstraintViolation]]:
    """Extract dimension value and description.

    Returns:
        (value, description, None) on success
        (None, None, violation) on error
    """
    if dim_index is None:
        from .types import prod
        return prod(interface.tensor_shape), "total", None

    if dim_index >= len(interface.tensor_shape):
        violation = ConstraintViolation(
            constraint_type="constraint_evaluation",
            message=f"Dimension index {dim_index} out of range for {interface_name}",
            severity="error"
        )
        return None, None, violation

    return interface.tensor_shape[dim_index], f"dim[{dim_index}]", None


def _resolve_parameter(
    param: Union[int, str, float],
    context: Dict[str, Any],
    param_type: str = "parameter"
) -> Tuple[Optional[Any], Optional[ConstraintViolation]]:
    """Resolve parameter from context or return literal.

    Returns:
        (value, None) on success
        (None, violation) on error
    """
    if isinstance(param, str):
        if param not in context:
            violation = ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"{param_type} '{param}' not found in context",
                severity="error"
            )
            return None, violation
        return context[param], None
    return param, None
```

#### 1.2 Refactor DivisibleConstraint (15 min)

**Before (lines 76-133):**
```python
def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
    violations = []

    # Get interface model
    if self.interface_name not in context:
        violations.append(ConstraintViolation(...))
        return ValidationResult(violations=violations)
    interface = context[self.interface_name]

    # Get dimension value
    if self.dim_index is None:
        from .types import prod
        dim_value = prod(interface.tensor_shape)
        dim_desc = "total_size"
    else:
        if self.dim_index >= len(interface.tensor_shape):
            violations.append(ConstraintViolation(...))
            return ValidationResult(violations=violations)
        dim_value = interface.tensor_shape[self.dim_index]
        dim_desc = f"dim[{self.dim_index}]"

    # Get divisor value
    if isinstance(self.divisor, str):
        if self.divisor not in context:
            violations.append(ConstraintViolation(...))
            return ValidationResult(violations=violations)
        divisor_value = context[self.divisor]
    else:
        divisor_value = self.divisor

    # Validate divisibility
    if dim_value % divisor_value != 0:
        violations.append(ConstraintViolation(...))

    return ValidationResult(violations=violations)
```

**After (~15 lines):**
```python
def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
    violations = []

    # Get interface
    interface, err = _get_interface(context, self.interface_name)
    if err:
        return ValidationResult(violations=[err])

    # Get dimension value
    dim_value, dim_desc, err = _get_dimension_value(interface, self.dim_index, self.interface_name)
    if err:
        return ValidationResult(violations=[err])

    # Get divisor
    divisor_value, err = _resolve_parameter(self.divisor, context, "Divisor")
    if err:
        return ValidationResult(violations=[err])

    # Validate
    if dim_value % divisor_value != 0:
        violations.append(ConstraintViolation(
            constraint_type="divisibility",
            message=f"{self.interface_name}.{dim_desc} must be divisible by {divisor_value}",
            expected=f"multiple of {divisor_value}",
            actual=dim_value,
            severity="error",
            details={"remainder": dim_value % divisor_value}
        ))

    return ValidationResult(violations=violations)
```

#### 1.3 Refactor Remaining Atomic Constraints (45 min)

Apply same pattern to:
- MinValueConstraint (lines 154-208)
- MaxValueConstraint (lines 229-283)
- RangeConstraint (lines 305-371)
- PowerOfTwoConstraint (lines 391-432)

Each becomes ~15-20 lines (from ~50 lines).

#### 1.4 Refactor Cross-Interface Constraints (45 min)

Add helper for two-interface extraction:

```python
def _get_two_dimension_values(
    context: Dict[str, Any],
    source_interface: str,
    source_dim: Optional[int],
    target_interface: str,
    target_dim: Optional[int]
) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], List[ConstraintViolation]]:
    """Get values from two interfaces."""
    violations = []

    # Get source
    source_intf, err = _get_interface(context, source_interface)
    if err:
        return None, None, None, None, [err]

    source_value, source_desc, err = _get_dimension_value(source_intf, source_dim, source_interface)
    if err:
        return None, None, None, None, [err]

    # Get target
    target_intf, err = _get_interface(context, target_interface)
    if err:
        return None, None, None, None, [err]

    target_value, target_desc, err = _get_dimension_value(target_intf, target_dim, target_interface)
    if err:
        return None, None, None, None, [err]

    return source_value, source_desc, target_value, target_desc, []
```

Refactor:
- EqualityConstraint (lines 459-526)
- DivisibleByDimensionConstraint (lines 549-623)
- ScaledEqualityConstraint (lines 647-728)

Each becomes ~20-25 lines (from ~70 lines).

#### 1.5 Test (15 min)

```bash
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v
```

**Expected Result:** All tests pass, ~400 lines deleted.

---

## Phase 2: Delete PowerOfTwoConstraint (Day 1, 30 min)

### Objective
Remove rarely-used constraint class (~60 lines).

### Steps

#### 2.1 Check Usage (5 min)
```bash
grep -r "PowerOfTwoConstraint" brainsmith/ --include="*.py"
```

If only in tests/dimension_constraints.py and __init__.py → safe to delete.

#### 2.2 Delete Class (5 min)
- Delete `PowerOfTwoConstraint` class (lines 379-437)
- Remove from `__all__` export list
- Remove from `__init__.py` imports

#### 2.3 Remove Test (5 min)
- Delete `test_power_of_two_constraint()` from test file

#### 2.4 Add Note to Docs (10 min)
Document alternative in docstring:

```python
"""
For power-of-2 validation, use:

    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    # In custom validation:
    if not is_power_of_two(dimension):
        raise ValueError(...)
"""
```

#### 2.5 Test (5 min)
```bash
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v
```

**Expected Result:** 9 tests pass (down from 10), ~60 lines deleted.

---

## Phase 3: Delete RangeConstraint (Day 1, 30 min)

### Objective
Remove composite constraint (~80 lines). Force composition over monolithic classes.

### Steps

#### 3.1 Update Test to Use Composition (10 min)

**Before:**
```python
def test_range_constraint():
    constraint = RangeConstraint("input", 0, 100, 200)
    result = constraint.validate_with_context(context)
    assert result.is_valid
```

**After:**
```python
def test_range_constraint_via_composition():
    """Range constraints via Min + Max composition."""
    # 100 <= input[0] <= 200
    min_constraint = MinValueConstraint("input", 0, 100)
    max_constraint = MaxValueConstraint("input", 0, 200)

    min_result = min_constraint.validate_with_context(context)
    max_result = max_constraint.validate_with_context(context)

    assert min_result.is_valid
    assert max_result.is_valid
```

#### 3.2 Delete Class (5 min)
- Delete `RangeConstraint` class (lines 291-376)
- Remove from `__all__`
- Remove from `__init__.py`

#### 3.3 Test (5 min)
```bash
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v
```

#### 3.4 Update Documentation (10 min)
Add to module docstring:

```python
"""
Range Constraints via Composition:

    Instead of a single RangeConstraint, compose Min + Max:

    # Range: 8 <= input[0] <= 1024
    schema.add_dimension_constraint(MinValueConstraint("input", 0, 8))
    schema.add_dimension_constraint(MaxValueConstraint("input", 0, 1024))

    This is more flexible and follows Unix philosophy.
"""
```

**Expected Result:** 8 tests pass, ~80 lines deleted.

---

## Phase 4: Unify Min/Max Constraints (Day 2, 1-2 hours) [OPTIONAL]

### Objective
Reduce Min/Max to single `ComparisonConstraint` (~130 lines saved).

### Decision Point
**Ask:** Do we have multiple comparison operators needed?
- If **YES** → Proceed with unification
- If **NO** → Skip this phase, keep Min/Max separate

### Steps (if proceeding)

#### 4.1 Create ComparisonConstraint (20 min)

```python
@dataclass(frozen=True)
class ComparisonConstraint(DimensionConstraint):
    """Compare dimension to value with operator.

    Examples:
        ComparisonConstraint("input", 0, ">=", 100)  # input[0] >= 100
        ComparisonConstraint("input", 1, "<=", 1024)  # input[1] <= 1024
        ComparisonConstraint("output", 0, "==", "SIZE")  # output[0] == SIZE
    """

    interface_name: str
    dim_index: Optional[int]
    operator: str  # ">=", "<=", "==", "!=", ">", "<"
    value: Union[int, str]

    _OPERATORS = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
    }

    _DESCRIPTIONS = {
        ">=": "must be >=",
        "<=": "must be <=",
        "==": "must equal",
        "!=": "must not equal",
        ">": "must be >",
        "<": "must be <",
    }

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        violations = []

        # Get interface
        interface, err = _get_interface(context, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Get dimension
        dim_value, dim_desc, err = _get_dimension_value(interface, self.dim_index, self.interface_name)
        if err:
            return ValidationResult(violations=[err])

        # Resolve value
        compare_value, err = _resolve_parameter(self.value, context, "Comparison value")
        if err:
            return ValidationResult(violations=[err])

        # Validate
        if self.operator not in self._OPERATORS:
            violations.append(ConstraintViolation(
                constraint_type="invalid_operator",
                message=f"Invalid operator '{self.operator}'",
                severity="error"
            ))
        elif not self._OPERATORS[self.operator](dim_value, compare_value):
            violations.append(ConstraintViolation(
                constraint_type="comparison",
                message=f"{self.interface_name}.{dim_desc} {self._DESCRIPTIONS[self.operator]} {compare_value}",
                expected=f"{self.operator} {compare_value}",
                actual=dim_value,
                severity="error"
            ))

        return ValidationResult(violations=violations)

    def describe(self) -> str:
        dim_str = "total" if self.dim_index is None else f"dim[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} {self.operator} {self.value}"
```

#### 4.2 Create Compatibility Aliases (10 min)

```python
# Backward compatibility
def MinValueConstraint(interface_name: str, dim_index: Optional[int], min_value: Union[int, str]):
    """Alias for ComparisonConstraint with >= operator."""
    return ComparisonConstraint(interface_name, dim_index, ">=", min_value)

def MaxValueConstraint(interface_name: str, dim_index: Optional[int], max_value: Union[int, str]):
    """Alias for ComparisonConstraint with <= operator."""
    return ComparisonConstraint(interface_name, dim_index, "<=", max_value)
```

#### 4.3 Update Tests (20 min)
Tests continue to work via aliases, or update to use new form:

```python
def test_comparison_constraint():
    # Old style (via alias)
    constraint = MinValueConstraint("input", 0, 100)

    # New style (direct)
    constraint = ComparisonConstraint("input", 0, ">=", 100)
```

#### 4.4 Update Exports (5 min)
```python
__all__ = [
    "DimensionConstraint",
    "DivisibleConstraint",
    "ComparisonConstraint",
    # Aliases for compatibility
    "MinValueConstraint",
    "MaxValueConstraint",
    # ...
]
```

#### 4.5 Test (5 min)
```bash
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v
```

**Expected Result:** All tests pass, ~100 lines deleted (after removing old classes).

---

## Phase 5: Unify Cross-Interface Constraints (Day 2, 2-3 hours)

### Objective
Extract shared logic from cross-interface constraints (~200 lines saved).

### Steps

#### 5.1 Create Base Class (30 min)

```python
@dataclass(frozen=True)
class BinaryDimensionConstraint(DimensionConstraint, ABC):
    """Base for constraints comparing two dimensions.

    Subclasses only need to implement _validate_relationship().
    """

    source_interface: str
    source_dim: Optional[int]
    target_interface: str
    target_dim: Optional[int]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Extract both values and delegate to subclass."""
        # Get both dimension values
        source_value, source_desc, target_value, target_desc, errors = \
            _get_two_dimension_values(
                context,
                self.source_interface,
                self.source_dim,
                self.target_interface,
                self.target_dim
            )

        if errors:
            return ValidationResult(violations=errors)

        # Delegate to subclass for specific validation
        violation = self._validate_relationship(
            source_value, source_desc,
            target_value, target_desc
        )

        if violation:
            return ValidationResult(violations=[violation])
        return ValidationResult()

    @abstractmethod
    def _validate_relationship(
        self,
        source_value: int,
        source_desc: str,
        target_value: int,
        target_desc: str
    ) -> Optional[ConstraintViolation]:
        """Validate the specific relationship.

        Returns:
            ConstraintViolation if invalid, None if valid
        """
        pass
```

#### 5.2 Refactor EqualityConstraint (15 min)

```python
@dataclass(frozen=True)
class EqualityConstraint(BinaryDimensionConstraint):
    """Two dimensions must be equal."""

    def _validate_relationship(
        self, source_value, source_desc, target_value, target_desc
    ) -> Optional[ConstraintViolation]:
        if source_value != target_value:
            return ConstraintViolation(
                constraint_type="dimension_equality",
                message=f"{self.source_interface}.{source_desc} must equal {self.target_interface}.{target_desc}",
                expected=source_value,
                actual=target_value,
                severity="error"
            )
        return None

    def describe(self) -> str:
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.source_interface}.{source_desc} == {self.target_interface}.{target_desc}"
```

#### 5.3 Refactor DivisibleByDimensionConstraint (15 min)

```python
@dataclass(frozen=True)
class DivisibleByDimensionConstraint(BinaryDimensionConstraint):
    """Dimension must be divisible by another dimension.

    Note: source is divisor, target is dividend (target % source == 0)
    """

    def _validate_relationship(
        self, source_value, source_desc, target_value, target_desc
    ) -> Optional[ConstraintViolation]:
        # Note: Using source as divisor, target as dividend
        divisor = source_value
        dividend = target_value

        if divisor == 0:
            return ConstraintViolation(
                constraint_type="constraint_evaluation",
                message=f"Divisor {self.source_interface}.{source_desc} cannot be zero",
                severity="error"
            )

        if dividend % divisor != 0:
            return ConstraintViolation(
                constraint_type="divisibility",
                message=f"{self.target_interface}.{target_desc} must be divisible by {self.source_interface}.{source_desc}",
                expected=f"multiple of {divisor}",
                actual=dividend,
                severity="error",
                details={"remainder": dividend % divisor}
            )
        return None

    def describe(self) -> str:
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.target_interface}.{target_desc} % {self.source_interface}.{source_desc} == 0"
```

#### 5.4 Refactor ScaledEqualityConstraint (20 min)

Add scale_factor to base:

```python
@dataclass(frozen=True)
class ScaledEqualityConstraint(BinaryDimensionConstraint):
    """Dimension must equal another dimension * scale_factor."""

    scale_factor: Union[int, float, str]

    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        # Get dimension values via parent
        source_value, source_desc, target_value, target_desc, errors = \
            _get_two_dimension_values(
                context,
                self.source_interface,
                self.source_dim,
                self.target_interface,
                self.target_dim
            )

        if errors:
            return ValidationResult(violations=errors)

        # Resolve scale factor
        scale, err = _resolve_parameter(self.scale_factor, context, "Scale factor")
        if err:
            return ValidationResult(violations=[err])

        # Validate
        violation = self._validate_with_scale(
            source_value, source_desc,
            target_value, target_desc,
            scale
        )

        if violation:
            return ValidationResult(violations=[violation])
        return ValidationResult()

    def _validate_with_scale(
        self, source_value, source_desc, target_value, target_desc, scale
    ) -> Optional[ConstraintViolation]:
        expected = source_value * scale
        if target_value != expected:
            return ConstraintViolation(
                constraint_type="scaled_equality",
                message=f"{self.target_interface}.{target_desc} must equal {self.source_interface}.{source_desc} * {scale}",
                expected=expected,
                actual=target_value,
                severity="error"
            )
        return None

    def describe(self) -> str:
        source_desc = "total" if self.source_dim is None else f"dim[{self.source_dim}]"
        target_desc = "total" if self.target_dim is None else f"dim[{self.target_dim}]"
        return f"{self.target_interface}.{target_desc} == {self.source_interface}.{source_desc} * {self.scale_factor}"
```

#### 5.5 Test (10 min)
```bash
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v
```

**Expected Result:** All tests pass, ~150 lines deleted.

---

## Phase 6: Final Cleanup & Documentation (Day 2, 1 hour)

### Steps

#### 6.1 Reorganize File Structure (15 min)

```python
# Final structure:
# 1. Imports
# 2. Helper functions (~80 lines)
# 3. Base class (DimensionConstraint)
# 4. Atomic constraints:
#    - DivisibleConstraint
#    - ComparisonConstraint (or Min/Max if keeping separate)
# 5. Cross-interface base (BinaryDimensionConstraint)
# 6. Cross-interface constraints:
#    - EqualityConstraint
#    - DivisibleByDimensionConstraint
#    - ScaledEqualityConstraint
# 7. __all__ export list
```

#### 6.2 Add Module Docstring (15 min)

```python
"""
Atomic dimension constraints for dataflow modeling.

This module provides constraint types that validate individual dimensions
or relationships between dimensions across interfaces.

## Constraint Categories

### Atomic Constraints (Single Dimension)
- DivisibleConstraint: Dimension divisible by value
- ComparisonConstraint: Dimension compared to value (>=, <=, ==, etc.)

### Cross-Interface Constraints
- EqualityConstraint: Two dimensions equal
- DivisibleByDimensionConstraint: Dimension divisible by another dimension
- ScaledEqualityConstraint: Dimension equals another * scale factor

## Design Principles

**Composition over Monoliths:**
    Range constraints via Min + Max composition:

    schema.add_dimension_constraint(MinValueConstraint("input", 0, 8))
    schema.add_dimension_constraint(MaxValueConstraint("input", 0, 1024))

**DRY:**
    All constraints use shared helper functions for:
    - Interface lookup
    - Dimension extraction
    - Parameter resolution

**Simplicity:**
    Each constraint class focuses on ONE validation rule.
    Complex validations are composed from simple constraints.
"""
```

#### 6.3 Update Main Package Docs (15 min)

Update `brainsmith/core/dataflow/README.md` or docstring with examples:

```markdown
## Dimension Constraints

### Atomic Constraints

```python
from brainsmith.core.dataflow import DivisibleConstraint, MinValueConstraint

# SIMD must divide input dimension
input_schema.add_dimension_constraint(
    DivisibleConstraint("input", 0, "SIMD")
)

# Minimum dimension size
input_schema.add_dimension_constraint(
    MinValueConstraint("input", 1, 64)
)
```

### Cross-Interface Constraints

```python
from brainsmith.core.dataflow import equal_dimension, divisible_dimension

# Matrix-vector multiplication: columns == vector length
schema.relationships.append(
    equal_dimension("matrix", "vector", 1, 0)
)

# Tiling: tensor divisible by block
schema.relationships.append(
    divisible_dimension("tensor", "block", 0, 0)
)
```
```

#### 6.4 Run Full Test Suite (10 min)

```bash
# Test constraints
./smithy pytest brainsmith/core/dataflow/tests/test_dimension_constraints.py -v

# Test integration with validation system
./smithy pytest brainsmith/core/dataflow/tests/test_validation.py -v

# Test imports
./smithy "python -c 'from brainsmith.core.dataflow import *; print(\"All imports OK\")'"
```

#### 6.5 Final Line Count (5 min)

```bash
wc -l brainsmith/core/dataflow/dimension_constraints.py
```

**Expected:** ~150-200 lines (down from 750)

---

## Success Criteria

### Metrics
- [ ] **Line count:** 750 → ~200 lines (73% reduction)
- [ ] **Test coverage:** All existing tests pass
- [ ] **No breaking changes:** Backward compatible exports
- [ ] **DRY compliance:** No duplicated validation logic
- [ ] **Readability:** Each constraint < 30 lines

### Validation Checklist
- [ ] Phase 1: Extract helpers → All tests pass
- [ ] Phase 2: Delete PowerOfTwo → 9 tests pass
- [ ] Phase 3: Delete Range → 8 tests pass
- [ ] Phase 4 (optional): Unify Min/Max → All tests pass
- [ ] Phase 5: Unify cross-interface → All tests pass
- [ ] Phase 6: Documentation complete

### Quality Gates
- [ ] No pylint/mypy errors
- [ ] All imports working
- [ ] Documentation updated
- [ ] Code review approved

---

## Rollback Plan

If any phase fails:

1. **Revert last commit:**
   ```bash
   git revert HEAD
   ```

2. **Run tests:**
   ```bash
   ./smithy pytest brainsmith/core/dataflow/tests/ -v
   ```

3. **Document issue:**
   - What failed
   - Why it failed
   - Alternative approach

4. **Proceed to next phase or skip problematic phase**

---

## Timeline

| Phase | Duration | Lines Saved | Cumulative |
|-------|----------|-------------|------------|
| Phase 1: Helpers | 2-3 hours | ~400 | ~400 |
| Phase 2: Delete PowerOfTwo | 30 min | ~60 | ~460 |
| Phase 3: Delete Range | 30 min | ~80 | ~540 |
| Phase 4: Unify Min/Max (optional) | 1-2 hours | ~100 | ~640 |
| Phase 5: Unify Cross-Interface | 2-3 hours | ~150 | ~790 |
| Phase 6: Cleanup & Docs | 1 hour | - | ~790 |
| **Total** | **1-2 days** | **~550-790 lines** | **73-77% reduction** |

---

## Notes

- **Phase 4 is optional** - Decision point based on whether unified comparison is cleaner
- **All phases maintain backward compatibility** via aliases or composition
- **Tests guide the refactoring** - if tests fail, approach needs adjustment
- **Arete achieved** when code is obvious in retrospect and contains only essential complexity
