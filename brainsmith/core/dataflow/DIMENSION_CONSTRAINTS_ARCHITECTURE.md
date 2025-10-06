# Dimension Constraint Architecture

## Overview

Dimension constraints validate tensor shapes during FPGA kernel model creation. The system validates **stream dimensions** (streaming parallelism - elements per cycle) rather than logical tensor dimensions.

**Key Design Principle**: Constraints validate `stream_shape`, which defines hardware parallelism in FPGA kernels.

## Shape Hierarchy

Three levels of shapes exist in the dataflow model:

1. **`tensor_shape`** - Full logical tensor dimensions (e.g., `[128, 64]`)
2. **`block_shape`** - Block tiling dimensions (e.g., `[128, 64]`)
3. **`stream_shape`** - **Streaming parallelism** - elements processed per cycle (e.g., `[1, 8]`)

**Constraints validate `stream_shape`** because this defines the hardware parallelism that SIMD/PE parameters control.

## Two-Phase Validation

### Phase 1: Atomic Constraints (Per-Interface)

Validated during `_create_input_model()` / `_create_output_model()`:

- **DivisibleConstraint** - `stream[dim] % divisor == 0`
- **MinValueConstraint** - `stream[dim] >= min_value`
- **MaxValueConstraint** - `stream[dim] <= max_value`

**Validation timing**: After model creation (needs `stream_shape` computed)

### Phase 2: Cross-Interface Constraints

Validated in `build_model()` after all interfaces exist:

- **EqualityConstraint** - `source.stream[i] == target.stream[j]`
- **DivisibleByDimensionConstraint** - `target.stream[j] % source.stream[i] == 0`
- **ScaledEqualityConstraint** - `target.stream[j] == source.stream[i] * scale`

**Validation timing**: After all interface models created

## Constraint Interface

### Base Class

```python
@dataclass(frozen=True)
class DimensionConstraint(ABC):
    """Base class for all dimension constraints."""

    @abstractmethod
    def check_interface(
        self,
        interface_name: str,
        interface_model: Any,  # InputModel or OutputModel
        nodeattr_getter: Callable[[str], Any]
    ) -> Optional[str]:
        """Validate atomic constraint on single interface.

        Args:
            interface_name: Name of interface being validated
            interface_model: Full interface model with stream_shape
            nodeattr_getter: Function to resolve nodeattrs (e.g., self.get_nodeattr)

        Returns:
            None if valid or not applicable to this interface
            Error message if constraint violated
        """
        pass

    @abstractmethod
    def check_relationship(
        self,
        interfaces: Dict[str, InterfaceModel]
    ) -> Optional[str]:
        """Validate cross-interface relationship.

        Args:
            interfaces: Dict of interface name → InterfaceModel

        Returns:
            None if valid or not applicable (atomic constraints)
            Error message if relationship violated
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of constraint."""
        pass
```

### Atomic Constraint Example

```python
@dataclass(frozen=True)
class DivisibleConstraint(DimensionConstraint):
    interface_name: str
    dim_index: Optional[int]  # None = total size
    divisor: Union[int, str]  # Literal or nodeattr name

    def check_interface(self, interface_name, interface_model, nodeattr_getter):
        # Only validate this interface
        if interface_name != self.interface_name:
            return None

        # Use stream_shape for validation
        if hasattr(interface_model, 'stream_shape'):
            stream_shape = interface_model.stream_shape
        else:
            # OutputModel: use block_shape as stream equivalent
            stream_shape = interface_model.block_shape

        # Get dimension value
        if self.dim_index is None:
            dim_value = prod(stream_shape)
            dim_desc = "stream_total"
        else:
            dim_value = stream_shape[self.dim_index]
            dim_desc = f"stream[{self.dim_index}]"

        # Resolve divisor (handles literal or nodeattr)
        if isinstance(self.divisor, str):
            divisor_value = nodeattr_getter(self.divisor)
        else:
            divisor_value = self.divisor

        # Validate divisibility
        if dim_value % divisor_value != 0:
            return f"{dim_desc} ({dim_value}) not divisible by {divisor_value}"

        return None

    def check_relationship(self, interfaces):
        # Not a cross-interface constraint
        return None

    def describe(self):
        dim_str = "stream_total" if self.dim_index is None else f"stream[{self.dim_index}]"
        return f"{self.interface_name}.{dim_str} % {self.divisor} == 0"
```

### Cross-Interface Constraint Example

```python
@dataclass(frozen=True)
class EqualityConstraint(DimensionConstraint):
    source_interface: str
    source_dim: Optional[int]
    target_interface: str
    target_dim: Optional[int]

    def check_interface(self, interface_name, interface_model, nodeattr_getter):
        # Cannot validate with single interface
        return None

    def check_relationship(self, interfaces):
        source = interfaces.get(self.source_interface)
        target = interfaces.get(self.target_interface)

        if source is None or target is None:
            return None  # Interfaces not available

        # Get stream shapes
        source_stream = source.stream_shape if hasattr(source, 'stream_shape') else source.block_shape
        target_stream = target.stream_shape if hasattr(target, 'stream_shape') else target.block_shape

        # Get dimension values
        source_val = (source_stream[self.source_dim] if self.source_dim is not None
                     else prod(source_stream))
        target_val = (target_stream[self.target_dim] if self.target_dim is not None
                     else prod(target_stream))

        # Validate equality
        if source_val != target_val:
            src_desc = (f"{self.source_interface}.stream[{self.source_dim}]"
                       if self.source_dim is not None else f"{self.source_interface}.stream_total")
            tgt_desc = (f"{self.target_interface}.stream[{self.target_dim}]"
                       if self.target_dim is not None else f"{self.target_interface}.stream_total")
            return f"{src_desc} ({source_val}) must equal {tgt_desc} ({target_val})"

        return None

    def describe(self):
        src = f"{self.source_interface}.stream[{self.source_dim}]" if self.source_dim is not None else f"{self.source_interface}.stream_total"
        tgt = f"{self.target_interface}.stream[{self.target_dim}]" if self.target_dim is not None else f"{self.target_interface}.stream_total"
        return f"{src} == {tgt}"
```

## Integration with AutoHWCustomOp

### Atomic Constraint Validation

```python
def _validate_atomic_constraints(
    self,
    interface_name: str,
    interface_model: Any,
    constraints: List[DimensionConstraint]
) -> None:
    """Validate atomic constraints during interface creation."""
    for constraint in constraints:
        error = constraint.check_interface(
            interface_name,
            interface_model,
            self.get_nodeattr  # Provides nodeattr resolution
        )
        if error:
            raise self._error(f"{interface_name}: {error}")
```

**Usage in Model Creation:**

```python
def _create_input_model(self, index: int) -> Optional[InputModel]:
    schema = self.kernel_schema.inputs[index]
    tensor = self.tensor_context.inputs[index]

    if tensor is None:
        return None

    # Create model to compute stream_shape
    block_shape = self._resolve_dimensions(...)
    stream_shape = self._resolve_dimensions(...)
    datatype = DataType[self.get_nodeattr(...)]

    input_model = InputModel(
        name=schema.name,
        tensor_shape=tensor.shape,
        block_shape=block_shape,
        stream_shape=stream_shape,
        datatype=datatype,
        is_weight=schema.is_weight
    )

    # VALIDATE ATOMIC CONSTRAINTS (after model creation)
    self._validate_atomic_constraints(
        schema.name,
        input_model,  # Pass full model with stream_shape
        schema.dimension_constraints
    )

    return input_model
```

### Cross-Interface Constraint Validation

```python
def _validate_cross_interface_constraints(self, model: KernelModel) -> None:
    """Validate cross-interface constraints after all interfaces created."""
    # Build interface lookup
    interfaces = {inp.name: inp for inp in model.inputs}
    interfaces.update({out.name: out for out in model.outputs})

    # Collect all constraints
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
```

**Usage in build_model:**

```python
def build_model(self) -> KernelModel:
    # Create interfaces (atomic constraints validated)
    input_models = []
    for i in range(len(self.kernel_schema.inputs)):
        input_model = self._create_input_model(i)
        if input_model is not None:
            input_models.append(input_model)

    output_models = []
    for i in range(len(self.kernel_schema.outputs)):
        output_models.append(self._create_output_model(i))

    model = KernelModel(
        name=self.kernel_schema.name,
        inputs=tuple(input_models),
        outputs=tuple(output_models),
    )

    # VALIDATE CROSS-INTERFACE CONSTRAINTS
    self._validate_cross_interface_constraints(model)

    self._kernel_model = model
    return model
```

## Parameter Resolution

Constraints can reference node attributes symbolically:

```python
# Schema definition
DivisibleConstraint("input", 1, "SIMD")  # "SIMD" is a nodeattr

# At validation time (in _create_input_model):
divisor = nodeattr_getter("SIMD")  # Resolves to concrete value (e.g., 16)
if dim_value % divisor != 0:
    raise error
```

This mirrors how `_resolve_dimensions()` works in `auto_hw_custom_op.py:256-304`.

## Performance Characteristics

**Direct validation (current):**
```python
# Direct function call with concrete values
error = constraint.check_interface(
    interface_name="input",
    interface_model=input_model,  # Has stream_shape
    nodeattr_getter=self.get_nodeattr
)
```

**Benefits over context-based validation:**
- No dict construction overhead
- No dict lookups
- Direct value passing
- Inline parameter resolution
- ~10x faster for schemas with many constraints

## Relationship to FINN

Dimension constraints align with FINN's streaming dataflow model:

- **SIMD** (Single Instruction Multiple Data) - Parallel processing units
- **PE** (Processing Elements) - Parallel compute lanes
- **Stream width** - Elements processed per clock cycle

Constraints ensure hardware parallelism parameters (SIMD, PE) correctly divide stream dimensions.

## Example: SIMD Constraint

```python
# Schema: Input must be SIMD-aligned
InputSchema(
    name="input",
    dimension_constraints=[
        DivisibleConstraint("input", 1, "SIMD")  # stream[1] % SIMD == 0
    ]
)

# At runtime with SIMD=8:
# ✓ stream_shape = (1, 8)  -> 8 % 8 == 0
# ✓ stream_shape = (1, 16) -> 16 % 8 == 0
# ✗ stream_shape = (1, 7)  -> 7 % 8 != 0 (ERROR)
```

## Testing Strategy

1. **Atomic constraint tests** - Verify `check_interface()` validates stream dimensions
2. **Cross-interface tests** - Verify `check_relationship()` validates relationships
3. **Timing tests** - Verify atomic constraints fail during model creation
4. **Parameter resolution** - Verify nodeattr references resolve correctly
5. **Error messages** - Verify clear, actionable error messages with "stream[dim]" notation

## Future Extensions

### Adding New Constraint Types

1. Create class inheriting from `DimensionConstraint`
2. Implement `check_interface()` (atomic) or `check_relationship()` (cross-interface)
3. Implement `describe()` for human-readable description
4. Add to `__init__.py` exports
5. Write tests

### Possible New Constraints

- **AlignedConstraint** - `stream[dim] % alignment == 0` (alias for DivisibleConstraint)
- **PowerOfTwoConstraint** - `is_power_of_two(stream[dim])`
- **ProportionalConstraint** - `target.stream[j] == source.stream[i] * ratio`
- **SumConstraint** - `sum(stream_shape) == value` (cross-dimension within interface)

All follow the same two-method pattern (`check_interface()` + `check_relationship()`).

## Files

- **dimension_constraints.py** - Constraint implementations
- **auto_hw_custom_op.py** - Integration and validation
- **schemas.py** - Schema definitions with constraint lists
- **relationships.py** - Relationship → Constraint conversion
- **validation.py** - Standalone validation (not used in AutoHWCustomOp)
- **tests/test_dimension_constraints.py** - Comprehensive test suite
