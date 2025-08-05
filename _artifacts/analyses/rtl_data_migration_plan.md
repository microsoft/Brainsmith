# RTL Data Migration Plan

## Current State

The `rtl_parser/rtl_data.py` module contains types that overlap with the new type system:
- `PortDirection` - Duplicates `types/core.py`
- `Port` - Duplicates `types/rtl.py`
- `Parameter` - Duplicates `types/rtl.py`
- `PortGroup` - Already in `types/rtl.py`
- `PragmaType` - Already in `types/rtl.py`
- `ProtocolValidationResult` - NOT in new types (specific to protocol validation)

## Migration Strategy

### Phase 1: Add Missing Types
1. Add `ProtocolValidationResult` to `types/rtl.py`
2. Ensure all functionality is preserved

### Phase 2: Update Imports
1. Update rtl_parser modules to import from new types
2. Keep rtl_data.py as a compatibility shim temporarily

### Phase 3: Clean Up
1. Remove duplicate definitions from rtl_data.py
2. Keep only rtl_parser-specific types if any

## Import Mapping

| Old Import | New Import |
|------------|------------|
| `from .rtl_data import Port` | `from brainsmith.tools.kernel_integrator.types.rtl import Port` |
| `from .rtl_data import Parameter` | `from brainsmith.tools.kernel_integrator.types.rtl import Parameter` |
| `from .rtl_data import PortDirection` | `from brainsmith.tools.kernel_integrator.types.core import PortDirection` |
| `from .rtl_data import PortGroup` | `from brainsmith.tools.kernel_integrator.types.rtl import PortGroup` |
| `from .rtl_data import PragmaType` | `from brainsmith.tools.kernel_integrator.types.rtl import PragmaType` |
| `from .rtl_data import ProtocolValidationResult` | `from brainsmith.tools.kernel_integrator.types.rtl import ProtocolValidationResult` |

## Files to Update

1. `rtl_parser/parser.py` - Uses Port, Parameter, PragmaType
2. `rtl_parser/protocol_validator.py` - Uses Port, PortGroup, ProtocolValidationResult, PortDirection
3. `rtl_parser/interface_builder.py` - Uses Port, ProtocolValidationResult, PortGroup
4. `rtl_parser/module_extractor.py` - Uses Port, Parameter, PragmaType, PortDirection
5. `rtl_parser/interface_scanner.py` - Uses Port, PortGroup
6. `rtl_parser/parameter_linker.py` - Uses Parameter
7. `rtl_parser/pragma.py` - Uses PragmaType
8. All pragma modules - Use PragmaType

## Compatibility Approach

To avoid breaking everything at once:

1. First add `ProtocolValidationResult` to new types
2. Make rtl_data.py re-export from new types:
   ```python
   from brainsmith.tools.kernel_integrator.types.core import PortDirection
   from brainsmith.tools.kernel_integrator.types.rtl import (
       Port, Parameter, PortGroup, PragmaType, ProtocolValidationResult
   )
   ```
3. Then gradually update imports in other modules
4. Finally remove rtl_data.py when all imports updated

This approach ensures no breaking changes during migration.