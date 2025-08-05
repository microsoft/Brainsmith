# Kernel Integrator Type System Refactoring Design

## Overview

This document proposes a comprehensive refactoring of the type system in `brainsmith/tools/kernel_integrator/` to eliminate circular dependencies, improve maintainability, and establish clear architectural boundaries.

## Current Problems

1. **Circular Dependencies**
   - `data.py` ↔ `metadata.py` create import cycles
   - Heavy reliance on `TYPE_CHECKING` guards
   - Types scattered across modules without clear organization

2. **Kitchen Sink Classes**
   - `TemplateContext`: 30+ fields mixing concerns
   - `KernelMetadata`: Attempts to hold all kernel information
   - `GenerationResult`: Mixes result tracking with file operations

3. **Unclear Boundaries**
   - Types spread across modules without logical grouping
   - No clear separation between parsing, generation, and runtime concerns
   - External dependencies (e.g., `DatatypeConstraintGroup`) not properly isolated

## Proposed Architecture

### Layer 1: Core Types (`types/core.py`)

Base types with no dependencies on other kernel integrator modules:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

class InterfaceType(Enum):
    """Types of interfaces in RTL modules."""
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    CONFIG = "config"
    CONTROL = "control"

class PortDirection(Enum):
    """Direction of RTL ports."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

@dataclass(frozen=True)
class DatatypeSpec:
    """Immutable datatype specification."""
    type_name: str
    template_params: Dict[str, int]
    bit_width: int
    
    @classmethod
    def from_string(cls, type_str: str) -> "DatatypeSpec":
        """Parse from string like 'ap_fixed<16,6>'."""
        # Implementation here
        pass

@dataclass(frozen=True)
class DimensionSpec:
    """Immutable dimension specification."""
    bdim: List[int]  # Block dimensions
    sdim: List[int]  # Stream dimensions
    
    @property
    def total_elements(self) -> int:
        """Calculate total number of elements."""
        # Implementation here
        pass
```

### Layer 2: RTL Types (`types/rtl.py`)

Types for RTL parsing and representation:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from .core import PortDirection, DatatypeSpec

@dataclass
class Port:
    """Single RTL port definition."""
    name: str
    direction: PortDirection
    width: int
    array_bounds: Optional[List[int]] = None

@dataclass
class Parameter:
    """RTL parameter definition."""
    name: str
    value: str
    is_local: bool = False

@dataclass
class ParsedModule:
    """Parsed RTL module representation."""
    name: str
    ports: List[Port]
    parameters: List[Parameter]
    file_path: str
    line_number: int

@dataclass
class ValidationError:
    """Single validation error."""
    severity: str  # "error" or "warning"
    message: str
    location: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
```

### Layer 3: Metadata Types (`types/metadata.py`)

Higher-level metadata built from RTL types:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from .core import InterfaceType, DatatypeSpec, DimensionSpec
from .rtl import Port, Parameter

@dataclass
class InterfaceMetadata:
    """Metadata for a single interface."""
    type: InterfaceType
    name: str
    datatype: DatatypeSpec
    dimensions: DimensionSpec
    ports: List[Port]
    parameters: Dict[str, Parameter]
    
    # Computed properties
    @property
    def width(self) -> int:
        """Total interface width in bits."""
        return self.datatype.bit_width * self.dimensions.total_elements

@dataclass
class KernelMetadata:
    """Complete kernel metadata."""
    module_name: str
    interfaces: Dict[str, InterfaceMetadata]
    global_parameters: Dict[str, Parameter]
    
    # Methods for specific queries
    def get_interface(self, interface_type: InterfaceType) -> Optional[InterfaceMetadata]:
        """Get first interface of given type."""
        for interface in self.interfaces.values():
            if interface.type == interface_type:
                return interface
        return None
```

### Layer 4: Generation Types (`types/generation.py`)

Types for code generation process:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from .metadata import KernelMetadata

@dataclass
class GeneratedFile:
    """Single generated file."""
    path: Path
    content: str
    
    def write(self) -> None:
        """Write content to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.content)

@dataclass
class GenerationContext:
    """Context for template generation."""
    kernel_metadata: KernelMetadata
    output_dir: Path
    class_name: str
    
    # Template-specific data
    template_data: Dict[str, any] = field(default_factory=dict)

@dataclass
class GenerationResult:
    """Result of generation process."""
    generated_files: List[GeneratedFile]
    validation_result: ValidationResult
    
    @property
    def is_success(self) -> bool:
        return self.validation_result.is_valid
    
    def write_all(self) -> None:
        """Write all generated files."""
        for file in self.generated_files:
            file.write()
```

### Layer 5: Binding Types (`types/binding.py`)

Types for code generation bindings:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from .core import InterfaceType

@dataclass
class IOSpec:
    """Specification for operator I/O."""
    python_name: str
    cpp_name: str
    interface_type: InterfaceType
    index: int = 0

@dataclass
class AttributeBinding:
    """Binding for operator attribute."""
    python_name: str
    cpp_name: str
    param_name: str
    python_type: str
    cpp_type: str
    default_value: Optional[str] = None

@dataclass
class CodegenBinding:
    """Complete binding specification."""
    class_name: str
    finn_op_type: str
    io_specs: List[IOSpec]
    attributes: List[AttributeBinding]
    
    # Helper methods
    def get_io_spec(self, interface_type: InterfaceType) -> Optional[IOSpec]:
        """Get IO spec for interface type."""
        for spec in self.io_specs:
            if spec.interface_type == interface_type:
                return spec
        return None
```

### Layer 6: Config Types (`types/config.py`)

Configuration types:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    """Kernel integrator configuration."""
    rtl_file: Path
    output_dir: Path
    module_name: Optional[str] = None
    class_name: Optional[str] = None
    template_dir: Optional[Path] = None
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.rtl_file.exists():
            raise ValueError(f"RTL file not found: {self.rtl_file}")
        
        # Set defaults
        if self.module_name is None:
            self.module_name = self.rtl_file.stem
        if self.class_name is None:
            self.class_name = self._to_camel_case(self.module_name)
    
    def _to_camel_case(self, name: str) -> str:
        """Convert module_name to ClassName."""
        parts = name.split('_')
        return ''.join(part.capitalize() for part in parts)
```

## Implementation Strategy

### Phase 1: Create New Type Structure
1. Create `types/` directory with all new modules
2. Implement core types with no dependencies
3. Build up layers incrementally

### Phase 2: Update Imports
1. Update all imports to use new type modules
2. Remove circular dependencies
3. Clean up TYPE_CHECKING guards

### Phase 3: Refactor Kitchen Sink Classes
1. Break up `TemplateContext` into focused contexts
2. Split `KernelMetadata` into smaller, focused classes
3. Simplify `GenerationResult` to just track results

### Phase 4: Standardize Patterns
1. Use `@dataclass(frozen=True)` for immutable types
2. Standardize validation in `__post_init__`
3. Use properties for computed values

## Benefits

1. **No Circular Dependencies**: Clear layered architecture
2. **Better Organization**: Types grouped by purpose
3. **Easier Testing**: Each layer can be tested independently
4. **Clear Contracts**: Required vs optional fields explicit
5. **Maintainability**: Easy to find and modify types

## Migration Path

1. **Parallel Implementation**: Build new types alongside old
2. **Adapter Layer**: Create adapters between old/new types
3. **Incremental Migration**: Update one module at a time
4. **Deprecation**: Mark old types as deprecated
5. **Cleanup**: Remove old types after full migration

## Example Usage

```python
# Clean imports with no cycles
from brainsmith.tools.kernel_integrator.types.core import InterfaceType
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata
from brainsmith.tools.kernel_integrator.types.generation import GenerationContext

# Clear type usage
def generate_hw_custom_op(metadata: KernelMetadata, context: GenerationContext) -> GenerationResult:
    """Generate HWCustomOp with clean type boundaries."""
    # Implementation here
    pass
```

This refactoring will significantly improve the maintainability and clarity of the kernel integrator codebase.

Arete.