# Unified Type System Refactoring: Dataflow + Kernel Integrator

## Overview

This document extends the kernel integrator type refactoring to include integration with the dataflow modeling system, ensuring proper dependency direction where `kernel_integrator` depends on `dataflow` but not vice versa.

## Current Analysis

### Dependency Direction (Correct)
- ✅ kernel_integrator → dataflow (imports DatatypeConstraintGroup, DimensionRelationship, RelationType)
- ✅ No reverse dependencies found
- ✅ Shared types already exist in dataflow

### Type Overlaps Identified

1. **InterfaceType Enum**
   - Defined in `kernel_integrator/data.py`
   - Represents fundamental interface categories (INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL)
   - Should be in dataflow as it's a core architectural concept

2. **Shape/Dimension Representations**
   - dataflow: `Shape = Tuple[int, ...]`, `RaggedShape`
   - kernel_integrator: `List[str]`, `List[Union[str, int]]` for pragmas
   - Opportunity for unified shape expression types

3. **Parameter Systems**
   - dataflow: Constraint-based system for kernel parameters
   - kernel_integrator: RTL parameter parsing and metadata
   - Different purposes, should remain separate

## Proposed Unified Architecture

### Layer 0: Dataflow Core Types (Extended)

Move fundamental types to `dataflow/types.py`:

```python
# brainsmith/core/dataflow/types.py (extended)

from enum import Enum
from typing import Union, List, Tuple

# Move from kernel_integrator
class InterfaceType(Enum):
    """Fundamental interface types for all kernels."""
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    CONFIG = "config"
    CONTROL = "control"

# New unified shape expression types
ShapeExpr = Union[int, str]  # Single dimension: 784 or "N"
ShapeSpec = List[ShapeExpr]  # Complete shape: [1, 784] or ["N", 768]

# Existing types remain
Shape = Tuple[int, ...]  # Concrete shapes only
RaggedShape = List[Shape]  # For irregular shapes
```

### Layer 1: Kernel Integrator Core Types (Updated)

Import from dataflow instead of defining locally:

```python
# brainsmith/tools/kernel_integrator/types/core.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# Import from dataflow
from brainsmith.core.dataflow.types import InterfaceType, ShapeSpec

# RTL-specific enums
class PortDirection(Enum):
    """Direction of RTL ports."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

@dataclass(frozen=True)
class DimensionSpec:
    """Dimension specification using dataflow shape types."""
    bdim: ShapeSpec  # Block dimensions using unified type
    sdim: ShapeSpec  # Stream dimensions using unified type
    
    def to_concrete_shape(self, params: Dict[str, int]) -> Tuple[List[int], List[int]]:
        """Resolve symbolic dimensions to concrete values."""
        bdim_concrete = [params.get(d, d) if isinstance(d, str) else d for d in self.bdim]
        sdim_concrete = [params.get(d, d) if isinstance(d, str) else d for d in self.sdim]
        return bdim_concrete, sdim_concrete
```

### Unified Type Mappings

#### Types to Move to Dataflow

1. **InterfaceType** - Fundamental enum used by both systems
2. **ShapeExpr/ShapeSpec** - Unified shape expression types

#### Types to Keep Separate

1. **RTL-Specific (kernel_integrator)**
   - Port, PortGroup, PortDirection
   - Parameter (RTL parameters)
   - ParsedModule, ValidationResult
   - All pragma types

2. **High-Level Modeling (dataflow)**
   - KernelDefinition, KernelModel
   - Interface definitions (InputDefinition, OutputDefinition)
   - Constraint and relationship types
   - Tiling specifications

3. **Already Shared (correct location)**
   - DatatypeConstraintGroup (in dataflow)
   - DimensionRelationship, RelationType (in dataflow)
   - QONNX datatype definitions (in dataflow)

### Integration Points

#### 1. Metadata Conversion

```python
# brainsmith/tools/kernel_integrator/converters.py

from brainsmith.core.dataflow.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.input_definition import InputDefinition
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata

def metadata_to_kernel_definition(metadata: KernelMetadata) -> KernelDefinition:
    """Convert parsed RTL metadata to high-level kernel definition."""
    inputs = []
    outputs = []
    
    for name, interface in metadata.interfaces.items():
        if interface.type == InterfaceType.INPUT:
            inputs.append(InputDefinition(
                name=name,
                shape=interface.dimensions.to_shape_spec(),
                dtype=interface.datatype.to_qonnx_type()
            ))
        # ... similar for outputs, weights, etc.
    
    return KernelDefinition(
        name=metadata.module_name,
        inputs=inputs,
        outputs=outputs,
        # ... other fields
    )
```

#### 2. Constraint Application

```python
# brainsmith/tools/kernel_integrator/constraint_builder.py

from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup
from brainsmith.tools.kernel_integrator.types.metadata import InterfaceMetadata

def build_datatype_constraints(interface: InterfaceMetadata) -> DatatypeConstraintGroup:
    """Build dataflow constraints from RTL interface metadata."""
    # Use existing DatatypeConstraintGroup from dataflow
    return DatatypeConstraintGroup(
        signed=interface.datatype.signed,
        allowed_widths={interface.datatype.bit_width},
        # ... other constraints
    )
```

## Benefits of Unified Approach

1. **Single Source of Truth**: Core concepts like InterfaceType defined once
2. **Proper Layering**: RTL parsing → metadata → dataflow models
3. **No Duplication**: Reuse constraint and relationship types
4. **Clear Boundaries**: RTL-specific vs. high-level modeling types
5. **Future Extensibility**: Easy to add new interface types or constraints

## Migration Strategy

### Phase 1: Move Core Types
1. Move InterfaceType to dataflow/types.py
2. Add ShapeExpr/ShapeSpec to dataflow/types.py
3. Update imports in kernel_integrator

### Phase 2: Refactor Kernel Integrator
1. Implement proposed type structure from previous design
2. Use dataflow types where appropriate
3. Keep RTL-specific types isolated

### Phase 3: Create Converters
1. Build metadata → kernel definition converters
2. Create constraint builders
3. Add validation for conversions

### Phase 4: Update Documentation
1. Document type relationships
2. Create dependency diagrams
3. Update API documentation

## Example: Complete Type Flow

```python
# 1. RTL Parsing (kernel_integrator domain)
from brainsmith.tools.kernel_integrator.rtl_parser import parse_rtl
parsed_module = parse_rtl("matmul.sv")

# 2. Build Metadata (kernel_integrator domain)
from brainsmith.tools.kernel_integrator.metadata_builder import build_metadata
metadata = build_metadata(parsed_module)

# 3. Convert to Dataflow Model (bridge)
from brainsmith.tools.kernel_integrator.converters import metadata_to_kernel_definition
kernel_def = metadata_to_kernel_definition(metadata)

# 4. Use in Dataflow System (dataflow domain)
from brainsmith.core.dataflow.kernel_model import KernelModel
model = KernelModel(kernel_def)
```

## Summary

This unified approach:
- Maintains proper dependency direction (kernel_integrator → dataflow)
- Eliminates type duplication
- Creates clear architectural boundaries
- Provides clean integration points
- Supports future extensibility

The key insight is that dataflow provides the fundamental types and constraints, while kernel_integrator handles RTL-specific parsing and metadata extraction, with clean converters bridging the two domains.

Arete.