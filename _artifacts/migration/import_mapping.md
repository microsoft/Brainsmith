# Import Mapping for Type Migration

## RTL Parser Imports

### From rtl_data
```python
# OLD
from .rtl_data import Port, Parameter, PortDirection, PortGroup, PragmaType, ProtocolValidationResult

# NEW
from brainsmith.tools.kernel_integrator.types.core import PortDirection
from brainsmith.tools.kernel_integrator.types.rtl import (
    Port, Parameter, PortGroup, PragmaType, ProtocolValidationResult
)
```

### From data
```python
# OLD
from .data import InterfaceType, GenerationResult, PerformanceMetrics

# NEW
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.types.generation import (
    GenerationResult, PerformanceMetrics
)
```

### From metadata
```python
# OLD
from .metadata import KernelMetadata, InterfaceMetadata

# NEW
from brainsmith.tools.kernel_integrator.types.metadata import (
    KernelMetadata, InterfaceMetadata
)
```

### From config
```python
# OLD
from .config import Config

# NEW
from brainsmith.tools.kernel_integrator.types.config import Config
```

## Files to Update

1. **RTL Parser modules** - Heavy users of rtl_data types
2. **Generator modules** - Use GenerationResult
3. **Template modules** - Use various types
4. **Main modules** - cli.py, kernel_integrator.py

Arete.