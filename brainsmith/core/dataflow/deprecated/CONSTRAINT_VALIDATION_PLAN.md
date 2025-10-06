# Dimension Constraint Validation Implementation Plan

## Overview

Replace context-based constraint validation with direct shape checking during model creation. This aligns with the existing `_resolve_dimensions()` pattern and eliminates the need for context dictionaries.

## Current Problem

**Context-based validation is inefficient:**
```python
# Current: Build context dict with interfaces AND nodeattrs
context = {inp.name: inp for inp in model.inputs}
context.update({out.name: out for out in model.outputs})
context.update({"SIMD": self.get_nodeattr("SIMD"), ...})  # Need to add ALL nodeattrs

# Validate after model creation
constraint.validate_with_context(context)
```

**Issues:**
- Context dict is computationally expensive
- Requires knowing which nodeattrs to include
- Validation happens AFTER model creation (too late)
- Doesn't align with existing resolution patterns

## Proposed Solution

**Direct constraint checking during model creation:**
```python
# In _create_input_model() - validate WHILE creating
for constraint in schema.dimension_constraints:
    error = constraint.check_interface(
        interface_name=schema.name,
        tensor_shape=tensor.shape,
        nodeattr_getter=self.get_nodeattr
    )
    if error:
        raise self._error(f"{schema.name}: {error}")
```

**Benefits:**
- No context dict needed
- Constraints resolve their own parameters
- Fail-fast validation during creation
- Aligns with `_resolve_dimensions()` pattern
- Type-safe and efficient

## Implementation Phases

### Phase 1: Add check_interface() to Base Class

**File:** `brainsmith/core/dataflow/dimension_constraints.py`

Add abstract method to `DimensionConstraint`:

```python
@dataclass(frozen=True)
class DimensionConstraint(ABC):
    """Base class for dimension constraints."""

    @abstractmethod
    def validate_with_context(self, context: Dict[str, Any]) -> ValidationResult:
        """Legacy validation method (will be deprecated)."""
        pass

    @abstractmethod
    def check_interface(
        self,
        interface_name: str,
        tensor_shape: Tuple[int, ...],
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Check constraint against an interface's shape.

        Args:
            interface_name: Name of interface being validated
            tensor_shape: Concrete shape tuple to check
            nodeattr_getter: Function to resolve nodeattr names (e.g., self.get_nodeattr)

        Returns:
            None if constraint is valid or not applicable to this interface
            Error message string if constraint is violated
        """
        pass
```

### Phase 2: Implement check_interface() for Atomic Constraints

**File:** `brainsmith/core/dataflow/dimension_constraints.py`

Implement for each atomic constraint class:

**DivisibleConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    # Skip if not for this interface
    if interface_name != self.interface_name:
        return None

    # Get dimension value
    from math import prod
    if self.dim_index is None:
        dim_value = prod(tensor_shape)
        dim_desc = "total size"
    else:
        if self.dim_index >= len(tensor_shape):
            return f"Dimension index {self.dim_index} out of range for shape {tensor_shape}"
        dim_value = tensor_shape[self.dim_index]
        dim_desc = f"dim[{self.dim_index}]"

    # Resolve divisor parameter
    if isinstance(self.divisor, str):
        try:
            divisor_value = nodeattr_getter(self.divisor)
        except (AttributeError, KeyError):
            return f"Parameter '{self.divisor}' not found in node attributes"
    else:
        divisor_value = self.divisor

    # Validate divisibility
    if dim_value % divisor_value != 0:
        return f"{dim_desc} ({dim_value}) must be divisible by {divisor_value}"

    return None
```

**MinValueConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    if interface_name != self.interface_name:
        return None

    # Get dimension value
    if self.dim_index is None:
        dim_value = prod(tensor_shape)
        dim_desc = "total size"
    else:
        if self.dim_index >= len(tensor_shape):
            return f"Dimension index {self.dim_index} out of range"
        dim_value = tensor_shape[self.dim_index]
        dim_desc = f"dim[{self.dim_index}]"

    # Resolve min value
    min_val = nodeattr_getter(self.min_value) if isinstance(self.min_value, str) else self.min_value

    # Validate
    if dim_value < min_val:
        return f"{dim_desc} ({dim_value}) must be >= {min_val}"

    return None
```

**MaxValueConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    if interface_name != self.interface_name:
        return None

    # Get dimension value
    if self.dim_index is None:
        dim_value = prod(tensor_shape)
        dim_desc = "total size"
    else:
        if self.dim_index >= len(tensor_shape):
            return f"Dimension index {self.dim_index} out of range"
        dim_value = tensor_shape[self.dim_index]
        dim_desc = f"dim[{self.dim_index}]"

    # Resolve max value
    max_val = nodeattr_getter(self.max_value) if isinstance(self.max_value, str) else self.max_value

    # Validate
    if dim_value > max_val:
        return f"{dim_desc} ({dim_value}) must be <= {max_val}"

    return None
```

### Phase 3: Implement check_interface() for Cross-Interface Constraints

**File:** `brainsmith/core/dataflow/dimension_constraints.py`

Cross-interface constraints need multiple interfaces, so they return None during single-interface validation:

**EqualityConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    # Cannot validate cross-interface constraints with single interface
    # Will be validated separately after all interfaces are created
    return None
```

**DivisibleByDimensionConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    # Cannot validate cross-interface constraints with single interface
    return None
```

**ScaledEqualityConstraint:**
```python
def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
    # Cannot validate cross-interface constraints with single interface
    return None
```

### Phase 4: Add Atomic Constraint Validation to AutoHWCustomOp

**File:** `brainsmith/core/finn/auto_hw_custom_op.py`

Add helper method:

```python
def _validate_atomic_constraints(
    self,
    interface_name: str,
    tensor_shape: Tuple[int, ...],
    constraints: List['DimensionConstraint']
) -> None:
    """Validate atomic dimension constraints for an interface.

    Called during _create_input_model/_create_output_model to validate
    constraints that only reference the current interface.

    Args:
        interface_name: Name of interface being validated
        tensor_shape: Shape to validate against
        constraints: List of constraints from schema

    Raises:
        HWCustomOpError: If any constraint is violated
    """
    for constraint in constraints:
        error_msg = constraint.check_interface(
            interface_name,
            tensor_shape,
            self.get_nodeattr
        )
        if error_msg:
            raise self._error(f"{interface_name}: {error_msg}")
```

### Phase 5: Integrate into _create_input_model()

**File:** `brainsmith/core/finn/auto_hw_custom_op.py`

Modify `_create_input_model()` to validate constraints:

```python
def _create_input_model(self, index: int) -> Optional[InputModel]:
    """Create input model from schema and tensor context."""
    schema = self.kernel_schema.inputs[index]
    tensor = self.tensor_context.inputs[index]

    if tensor is None:
        return None

    # Validate atomic dimension constraints BEFORE creating model
    self._validate_atomic_constraints(
        schema.name,
        tensor.shape,
        schema.dimension_constraints
    )

    block_shape = self._resolve_dimensions(
        schema.block_tiling,
        tensor.shape,
        f"Input '{schema.name}' block"
    )

    stream_shape = self._resolve_dimensions(
        schema.stream_tiling,
        block_shape,
        f"Input '{schema.name}' stream"
    )

    datatype_attr = self.kernel_schema.get_datatype_attr(index)
    datatype = DataType[self.get_nodeattr(datatype_attr)]

    return InputModel(
        name=schema.name,
        tensor_shape=tensor.shape,
        block_shape=block_shape,
        stream_shape=stream_shape,
        datatype=datatype,
        is_weight=schema.is_weight
    )
```

### Phase 6: Integrate into _create_output_model()

**File:** `brainsmith/core/finn/auto_hw_custom_op.py`

Similar integration for outputs:

```python
def _create_output_model(self, index: int) -> OutputModel:
    """Create output model from schema and tensor context."""
    schema = self.kernel_schema.outputs[index]
    tensor = self.tensor_context.outputs[index]

    # Validate atomic dimension constraints BEFORE creating model
    self._validate_atomic_constraints(
        schema.name,
        tensor.shape,
        schema.dimension_constraints
    )

    block_shape = self._resolve_dimensions(
        schema.block_tiling,
        tensor.shape,
        f"Output '{schema.name}' block"
    )

    datatype_attr = self.kernel_schema.get_datatype_attr(index, False)
    datatype = DataType[self.get_nodeattr(datatype_attr)]

    return OutputModel(
        name=schema.name,
        tensor_shape=tensor.shape,
        block_shape=block_shape,
        datatype=datatype
    )
```

### Phase 7: Handle Cross-Interface Constraints

**File:** `brainsmith/core/finn/auto_hw_custom_op.py`

Add method to validate cross-interface constraints:

```python
def _validate_cross_interface_constraints(self, model: KernelModel) -> None:
    """Validate constraints that reference multiple interfaces.

    Called in build_model() after all interface models are created.

    Args:
        model: Complete KernelModel with all interfaces

    Raises:
        HWCustomOpError: If any constraint is violated
    """
    # Build lookup for interfaces
    interfaces = {inp.name: inp for inp in model.inputs}
    interfaces.update({out.name: out for out in model.outputs})

    # Check cross-interface constraints from relationships
    for relationship in self.kernel_schema.relationships:
        try:
            constraints = relationship.get_constraints()
            for constraint in constraints:
                # Use helper to validate cross-interface constraint
                source_iface = interfaces.get(constraint.source_interface)
                target_iface = interfaces.get(constraint.target_interface)

                if source_iface is None or target_iface is None:
                    continue  # Skip if interface not found

                # Get dimension values
                source_dim = (constraint.source_dim if constraint.source_dim is not None
                             else None)  # None means total size
                target_dim = (constraint.target_dim if constraint.target_dim is not None
                             else None)

                # Let constraint validate itself
                from math import prod
                source_val = (source_iface.tensor_shape[source_dim] if source_dim is not None
                             else prod(source_iface.tensor_shape))
                target_val = (target_iface.tensor_shape[target_dim] if target_dim is not None
                             else prod(target_iface.tensor_shape))

                # Type-specific validation (will be improved with constraint methods)
                if isinstance(constraint, EqualityConstraint):
                    if source_val != target_val:
                        raise self._error(
                            f"Constraint violated: {constraint.source_interface}[{source_dim}] "
                            f"({source_val}) must equal {constraint.target_interface}[{target_dim}] "
                            f"({target_val})"
                        )
                # ... handle other cross-interface constraint types

        except Exception as e:
            raise self._error(f"Cross-interface constraint validation failed: {str(e)}")
```

### Phase 8: Integrate Cross-Interface Validation into build_model()

**File:** `brainsmith/core/finn/auto_hw_custom_op.py`

Modify `build_model()`:

```python
def build_model(self) -> KernelModel:
    """Create KernelModel from schema, tensor context, and nodeattrs.

    Raises:
        RuntimeError: If tensor context not initialized
        HWCustomOpError: If validation fails
    """
    # Create all interface models (atomic constraints validated during creation)
    input_models = []
    for i in range(len(self.kernel_schema.inputs)):
        input_model = self._create_input_model(i)  # Validates atomic constraints
        if input_model is not None:
            input_models.append(input_model)

    output_models = []
    for i in range(len(self.kernel_schema.outputs)):
        output_models.append(self._create_output_model(i))  # Validates atomic constraints

    model = KernelModel(
        name=self.kernel_schema.name,
        inputs=tuple(input_models),
        outputs=tuple(output_models),
    )

    # Validate cross-interface constraints now that all interfaces exist
    self._validate_cross_interface_constraints(model)

    self._kernel_model = model
    return model
```

### Phase 9: Testing

**File:** `brainsmith/core/dataflow/tests/test_dimension_constraints.py`

Update tests to use new validation flow:

```python
def test_divisible_constraint_in_build_model():
    """Test that divisible constraints are validated during model creation."""
    # Create mock AutoHWCustomOp with constraint
    # ... test that constraint is checked during _create_input_model()
    # ... verify error is raised if constraint violated
```

Add tests for:
- Atomic constraint validation timing
- Cross-interface constraint validation timing
- Parameter resolution (str -> int via nodeattr)
- Error messages and reporting

### Phase 10: Deprecation Plan

**Future work:**
- Mark `validate_with_context()` as deprecated
- Remove context-based validation from `validation.py` if no longer needed
- Keep `ValidationResult` and `ConstraintViolation` for compatibility
- Consider migrating to simpler error strings instead of ValidationResult objects

## Migration Path

1. **Phase 1-3:** Add `check_interface()` alongside existing `validate_with_context()` (both work)
2. **Phase 4-8:** Integrate `check_interface()` into AutoHWCustomOp (new validation path active)
3. **Phase 9:** Test thoroughly to ensure both paths work
4. **Phase 10:** Deprecate and eventually remove old context-based validation

## Alignment with Datatype Constraints

This design mirrors how datatype constraints work:
- **Datatype:** Resolved during `_create_input_model()` via `self.get_nodeattr(datatype_attr)`
- **Dimensions:** Validated during `_create_input_model()` via `constraint.check_interface()`

Both happen at the same point in the pipeline, with access to nodeattrs.

## Performance Benefits

**Before (context-based):**
- Build context dict with ALL nodeattrs (expensive)
- Store references to ALL interface models
- Validate AFTER model creation (too late to prevent creation)
- Dict lookups for every parameter resolution

**After (direct checking):**
- No context dict needed
- Direct function calls with concrete values
- Validate DURING model creation (fail-fast)
- Constraints resolve their own parameters (self-contained)

**Estimated improvement:** ~10x faster for large schemas with many constraints
