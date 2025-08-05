# Kernel Integrator Type System Migration Guide

This guide helps you migrate code from the old type system (v3.x) to the new modular type system (v4.0).

## Overview of Changes

The v4.0 refactoring introduces a modular type system that:
- Eliminates circular dependencies between modules
- Provides clear separation of concerns
- Improves maintainability and testability
- Adds an integration layer for dataflow compatibility

## Breaking Changes

### 1. Import Path Changes

All types have been reorganized into dedicated modules under `brainsmith.tools.kernel_integrator.types/`.

#### Core Type Imports

**Before (v3.x):**
```python
from brainsmith.tools.kernel_integrator.data import (
    InterfaceType,
    PortDirection,
    DatatypeSpec,
    Port,
    Parameter
)
```

**After (v4.0):**
```python
# InterfaceType now lives in dataflow
from brainsmith.core.dataflow.types import InterfaceType

# Other types in their respective modules
from brainsmith.tools.kernel_integrator.types.core import (
    PortDirection,
    DatatypeSpec,
    DimensionSpec
)
from brainsmith.tools.kernel_integrator.types.rtl import (
    Port,
    Parameter,
    ParsedModule
)
```

#### Metadata Type Imports

**Before (v3.x):**
```python
from brainsmith.tools.kernel_integrator.metadata import (
    InterfaceMetadata,
    KernelMetadata,
    build_kernel_metadata
)
```

**After (v4.0):**
```python
from brainsmith.tools.kernel_integrator.types.metadata import (
    InterfaceMetadata,
    KernelMetadata
)
# build_kernel_metadata remains in the old location for compatibility
from brainsmith.tools.kernel_integrator.metadata import build_kernel_metadata
```

#### Generation Type Imports

**Before (v3.x):**
```python
from brainsmith.tools.kernel_integrator.data import (
    GenerationResult,
    GeneratedFile
)
```

**After (v4.0):**
```python
from brainsmith.tools.kernel_integrator.types.generation import (
    GenerationResult,
    GeneratedFile,
    GenerationContext
)
```

### 2. API Changes

#### InterfaceMetadata

The `InterfaceMetadata` class has been updated to match the old API more closely:

**Before (v3.x):**
```python
interface = InterfaceMetadata(
    compiler_name="input0",
    type=InterfaceType.INPUT,  # Note: 'type' parameter
    # ... other fields
)
```

**After (v4.0):**
```python
interface = InterfaceMetadata(
    compiler_name="input0",
    interface_type=InterfaceType.INPUT,  # Now: 'interface_type'
    # ... other fields
)
```

#### KernelMetadata.interfaces

The interfaces are now stored as a list instead of a dictionary:

**Before (v3.x):**
```python
# Interfaces as dict
kernel.interfaces["input0"].datatype_width
```

**After (v4.0):**
```python
# Interfaces as list - use helper methods
input_interface = kernel.get_interface("input0")
if input_interface:
    width = input_interface.datatype_width

# Or get all of a type
input_interfaces = kernel.get_input_interfaces()
```

#### Shape Specifications

Shape types have changed from custom Shape class to standard tuples:

**Before (v3.x):**
```python
from brainsmith.core.dataflow.shapes import Shape
shape = Shape([1, 784])
```

**After (v4.0):**
```python
# Shapes are now tuples
shape = (1, 784)

# For variable shapes, use ShapeSpec (list)
from brainsmith.core.dataflow.types import ShapeSpec
variable_shape: ShapeSpec = ["N", 768]
```

### 3. New Required Imports

#### Using the Integration Layer

To convert between kernel integrator and dataflow types:

```python
from brainsmith.tools.kernel_integrator.converters import (
    metadata_to_kernel_definition,
    kernel_definition_to_metadata
)

# Convert to dataflow
kernel_def = metadata_to_kernel_definition(kernel_metadata)

# Convert back
metadata = kernel_definition_to_metadata(kernel_def, source_file)
```

#### Building Constraints

```python
from brainsmith.tools.kernel_integrator.constraint_builder import (
    build_dimension_constraints,
    build_parameter_constraints
)
```

## Migration Strategies

### Strategy 1: Minimal Changes (Recommended for Small Projects)

Update imports only, keeping the same logic:

```python
# Update imports to new paths
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.rtl import Port, Parameter

# Change interface access pattern
# Old: kernel.interfaces["input0"]
# New: kernel.get_interface("input0")
```

### Strategy 2: Full Migration (Recommended for Active Projects)

1. Update all imports to new paths
2. Replace dictionary access with helper methods
3. Use the integration layer for dataflow compatibility
4. Update shape handling to use tuples

## Common Migration Patterns

### Pattern 1: Accessing Interfaces

**Before:**
```python
def process_kernel(kernel: KernelMetadata):
    # Direct dictionary access
    for name, interface in kernel.interfaces.items():
        if interface.type == InterfaceType.INPUT:
            print(f"Input {name}: width={interface.datatype_width}")
```

**After:**
```python
def process_kernel(kernel: KernelMetadata):
    # Use helper methods
    for interface in kernel.get_input_interfaces():
        print(f"Input {interface.compiler_name}: width={interface.datatype_width}")
```

### Pattern 2: Creating Metadata

**Before:**
```python
interface = InterfaceMetadata(
    compiler_name="weight0",
    type=InterfaceType.WEIGHT,
    datatype=DatatypeSpec(name="INT8", width=8),
    block_dimensions=Shape([784, 128])
)
```

**After:**
```python
interface = InterfaceMetadata(
    compiler_name="weight0",
    interface_type=InterfaceType.WEIGHT,  # Changed parameter name
    datatype_name="INT8",  # Flattened structure
    datatype_width=8,
    block_dimensions=[784, 128]  # Now a list
)
```

### Pattern 3: Working with Shapes

**Before:**
```python
from brainsmith.core.dataflow.shapes import Shape

def get_shape_product(shape: Shape) -> int:
    return shape.product()
```

**After:**
```python
from typing import Tuple
import math

def get_shape_product(shape: Tuple[int, ...]) -> int:
    return math.prod(shape)
```

## Deprecated Features

The following features are deprecated and should be migrated:

1. **Direct dictionary access to interfaces** - Use `get_interface()` method
2. **Shape class** - Use tuples for concrete shapes, lists for symbolic
3. **Nested datatype objects in metadata** - Use flattened fields

## Troubleshooting

### Import Errors

**Error:** `ImportError: cannot import name 'InterfaceType' from 'brainsmith.tools.kernel_integrator.data'`

**Solution:** Import from dataflow instead:
```python
from brainsmith.core.dataflow.types import InterfaceType
```

### Attribute Errors

**Error:** `AttributeError: 'KernelMetadata' object has no attribute 'interfaces' (dict access)`

**Solution:** Use the list interface and helper methods:
```python
# Instead of: kernel.interfaces["input0"]
interface = kernel.get_interface("input0")
```

### Type Errors

**Error:** `TypeError: 'type' is not a valid parameter for InterfaceMetadata`

**Solution:** Use `interface_type` instead:
```python
InterfaceMetadata(interface_type=InterfaceType.INPUT, ...)
```

## Complete Migration Example

Here's a complete example showing a typical migration:

### Before (v3.x)

```python
from brainsmith.tools.kernel_integrator.data import (
    InterfaceType, KernelMetadata, GenerationResult
)
from brainsmith.tools.kernel_integrator.metadata import build_kernel_metadata
from brainsmith.core.dataflow.shapes import Shape

def process_kernel(rtl_file: Path) -> GenerationResult:
    # Parse and build metadata
    kernel = build_kernel_metadata(rtl_file)
    
    # Access interfaces by dict
    for name, interface in kernel.interfaces.items():
        if interface.type == InterfaceType.INPUT:
            shape = interface.block_dimensions
            size = shape.product()
            print(f"Input {name} size: {size}")
    
    # Generate code
    result = generate_code(kernel)
    return result
```

### After (v4.0)

```python
from pathlib import Path
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.generation import GenerationResult
from brainsmith.tools.kernel_integrator.metadata import build_kernel_metadata
from brainsmith.tools.kernel_integrator.converters import metadata_to_kernel_definition
import math

def process_kernel(rtl_file: Path) -> GenerationResult:
    # Parse and build metadata (same function)
    kernel = build_kernel_metadata(rtl_file)
    
    # Access interfaces using helper methods
    for interface in kernel.get_input_interfaces():
        if interface.block_dimensions:
            # Convert symbolic dimensions if needed
            shape = interface.block_dimensions
            if all(isinstance(d, int) for d in shape):
                size = math.prod(shape)
                print(f"Input {interface.compiler_name} size: {size}")
    
    # Optional: Convert to dataflow types
    kernel_def = metadata_to_kernel_definition(kernel)
    
    # Generate code (same function)
    result = generate_code(kernel)
    return result
```

## Getting Help

If you encounter issues during migration:

1. Check the [API Reference](API_REFERENCE.md) for detailed type documentation
2. Review the [Architecture Documentation](ARCHITECTURE.md) for system overview
3. Look at the test files for usage examples
4. File an issue if you find a bug or missing feature

## Version Compatibility

- The old import paths are completely removed in v4.0
- Builder functions like `build_kernel_metadata` remain for compatibility
- The integration layer ensures full compatibility with dataflow types

Arete.