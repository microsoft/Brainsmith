# Simplified Constraint Validation Plan

## Core Insight: Two Validation Methods

Instead of decomposing cross-interface constraints, use **two validation methods**:

1. **`check_interface()`** - Atomic constraints validate single interface
2. **`check_relationship()`** - Cross-interface constraints validate relationships

Both methods avoid context dicts and resolve parameters inline.

## Architecture

```python
class DimensionConstraint(ABC):
    """Base class for all dimension constraints."""

    @abstractmethod
    def check_interface(
        self,
        interface_name: str,
        tensor_shape: Tuple[int, ...],
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Validate atomic constraint on single interface.

        Returns:
            None if valid or not applicable
            Error message if violated
        """
        pass

    @abstractmethod
    def check_relationship(
        self,
        interfaces: Dict[str, InterfaceModel]
    ) -> Optional[str]:
        """Validate cross-interface relationship.

        Only implemented for constraints that span multiple interfaces.
        Atomic constraints return None.

        Returns:
            None if valid or not applicable
            Error message if violated
        """
        pass
```

## Constraint Types

### Atomic Constraints (Single Interface)

**DivisibleConstraint, MinValueConstraint, MaxValueConstraint**

- Implement `check_interface()` - validates the dimension
- Implement `check_relationship()` - returns None (not applicable)

```python
class DivisibleConstraint(DimensionConstraint):
    def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
        if interface_name != self.interface_name:
            return None  # Not for this interface

        # Get dimension value
        dim_val = tensor_shape[self.dim_index] if self.dim_index is not None else prod(tensor_shape)

        # Resolve divisor (literal or nodeattr)
        divisor = nodeattr_getter(self.divisor) if isinstance(self.divisor, str) else self.divisor

        # Validate
        if dim_val % divisor != 0:
            return f"dim[{self.dim_index}] ({dim_val}) not divisible by {divisor}"
        return None

    def check_relationship(self, interfaces):
        return None  # Not a cross-interface constraint
```

### Cross-Interface Constraints (Multiple Interfaces)

**EqualityConstraint, DivisibleByDimensionConstraint, ScaledEqualityConstraint**

- Implement `check_interface()` - returns None (needs multiple interfaces)
- Implement `check_relationship()` - validates the relationship

```python
class EqualityConstraint(DimensionConstraint):
    source_interface: str
    source_dim: Optional[int]
    target_interface: str
    target_dim: Optional[int]

    def check_interface(self, interface_name, tensor_shape, nodeattr_getter):
        return None  # Cannot validate cross-interface with single interface

    def check_relationship(self, interfaces):
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available (optional?)

        # Get dimension values
        source_val = (source.tensor_shape[self.source_dim] if self.source_dim is not None
                     else prod(source.tensor_shape))
        target_val = (target.tensor_shape[self.target_dim] if self.target_dim is not None
                     else prod(target.tensor_shape))

        # Validate equality
        if source_val != target_val:
            src_desc = f"{self.source_interface}[{self.source_dim}]" if self.source_dim is not None else f"{self.source_interface}.total"
            tgt_desc = f"{self.target_interface}[{self.target_dim}]" if self.target_dim is not None else f"{self.target_interface}.total"
            return f"{src_desc} ({source_val}) must equal {tgt_desc} ({target_val})"

        return None
```

## Integration into AutoHWCustomOp

### Phase 1: Atomic Constraint Validation (During Interface Creation)

**In `_create_input_model()` and `_create_output_model()`:**

```python
def _validate_atomic_constraints(
    self,
    interface_name: str,
    tensor_shape: Tuple[int, ...],
    constraints: List[DimensionConstraint]
) -> None:
    """Validate atomic constraints for an interface."""
    for constraint in constraints:
        error = constraint.check_interface(interface_name, tensor_shape, self.get_nodeattr)
        if error:
            raise self._error(f"{interface_name}: {error}")

def _create_input_model(self, index: int) -> Optional[InputModel]:
    schema = self.kernel_schema.inputs[index]
    tensor = self.tensor_context.inputs[index]

    if tensor is None:
        return None

    # Validate atomic constraints BEFORE creating model
    self._validate_atomic_constraints(
        schema.name,
        tensor.shape,
        schema.dimension_constraints
    )

    # ... rest of model creation
    return InputModel(...)
```

### Phase 2: Cross-Interface Validation (After All Interfaces Created)

**In `build_model()`:**

```python
def _validate_cross_interface_constraints(self, model: KernelModel) -> None:
    """Validate cross-interface constraints after all interfaces created."""
    # Build interface lookup
    interfaces = {inp.name: inp for inp in model.inputs}
    interfaces.update({out.name: out for out in model.outputs})

    # Check all constraints from all interfaces
    all_constraints = []
    for schema in self.kernel_schema.inputs:
        all_constraints.extend(schema.dimension_constraints)
    for schema in self.kernel_schema.outputs:
        all_constraints.extend(schema.dimension_constraints)

    # Validate cross-interface constraints
    for constraint in all_constraints:
        error = constraint.check_relationship(interfaces)
        if error:
            raise self._error(error)

def build_model(self) -> KernelModel:
    # Create all interfaces (atomic constraints validated during creation)
    input_models = []
    for i in range(len(self.kernel_schema.inputs)):
        input_model = self._create_input_model(i)  # Validates atomic
        if input_model is not None:
            input_models.append(input_model)

    output_models = []
    for i in range(len(self.kernel_schema.outputs)):
        output_models.append(self._create_output_model(i))  # Validates atomic

    model = KernelModel(...)

    # Validate cross-interface constraints
    self._validate_cross_interface_constraints(model)

    return model
```

## Implementation Steps

1. **Add both methods to `DimensionConstraint` base class**
   - `check_interface()` - abstract method
   - `check_relationship()` - abstract method

2. **Implement for atomic constraints**
   - `check_interface()` - full validation logic
   - `check_relationship()` - return None

3. **Implement for cross-interface constraints**
   - `check_interface()` - return None
   - `check_relationship()` - full validation logic

4. **Add helpers to AutoHWCustomOp**
   - `_validate_atomic_constraints()` - calls check_interface()
   - `_validate_cross_interface_constraints()` - calls check_relationship()

5. **Integrate validation**
   - Atomic: in `_create_input_model()` / `_create_output_model()`
   - Cross-interface: in `build_model()` after all interfaces created

6. **Test thoroughly**
   - Atomic constraint timing
   - Cross-interface constraint timing
   - Parameter resolution (nodeattr references)

## Benefits Over Context-Based Validation

**Performance:**
- No context dict creation
- No dict lookups for parameter resolution
- Direct function calls with concrete values

**Clarity:**
- Two clear validation phases (atomic → cross-interface)
- Constraints are self-contained
- Matches existing `_resolve_dimensions()` pattern

**Type Safety:**
- Interfaces passed directly, not through dict
- Nodeattr resolution is explicit
- Return values are simple (Optional[str])

**Alignment:**
- Mirrors datatype constraint validation timing
- Validates during model creation (fail-fast)
- Consistent with existing AutoHWCustomOp patterns

## Migration Path

1. Add new methods alongside existing `validate_with_context()`
2. Implement new validation in AutoHWCustomOp
3. Test both paths work correctly
4. Deprecate `validate_with_context()` and context-based validation
5. Remove deprecated code after migration period
