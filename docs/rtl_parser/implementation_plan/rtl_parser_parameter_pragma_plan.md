# RTL Parser Implementation Plan

## Overview
This plan outlines the implementation of parameter processing and pragma handling for the RTL Parser component of the Hardware Kernel Generator (HKG).

## Current State
The RTL Parser has:
- Complete interface analysis pipeline with scanning and validation
- Basic parameter extraction from module definitions
- Generic pragma parsing framework

## Implementation Requirements

### 1. Parameter Processing

#### A. Enhanced Parameter Model (`data.py`)
```python
@dataclass
class KernelParameter:
    """Parameter in generated hardware kernel"""
    name: str
    param_type: str
    default_value: Optional[str] = None
    derived_function: Optional[str] = None    # Python function name if derived
    dependent_params: List[str] = field(default_factory=list)  # Parameters this depends on
```

#### B. Parameter Processor
- Create `parameter_processor.py` to:
  1. Convert RTL parameters to HW Kernel parameters
  2. Handle derived parameter relationships
  3. Validate parameter types and values

Code structure:
```python
class ParameterProcessor:
    def __init__(self):
        self.derived_functions = {}  # name -> function mapping
        
    def process_parameters(self, 
                         rtl_params: List[Parameter],
                         pragmas: List[Pragma]) -> List[KernelParameter]:
        """Convert RTL parameters to Kernel parameters"""
        
    def register_derived_function(self, name: str, func: Callable):
        """Register Python function for derived parameters"""
        
    def validate_parameters(self, params: List[KernelParameter]) -> List[str]:
        """Validate parameter relationships and types"""
```

### 2. Pragma Processing

#### A. Update Pragma Types (`pragma.py`)
Replace existing pragma types with:
```python
class PragmaType(Enum):
    TOP = "top"                      # Select top module
    SUPPORTED_DTYPE = "supported_dtype"  # Data type restrictions
    DERIVED_PARAM = "derived_param"   # Parameter relationships
```

#### B. Pragma Handlers
Implement handlers for each pragma type:

1. Top Module:
```python
def _handle_top(inputs: List[str]) -> Dict:
    """Handle top module pragma
    Format: @brainsmith top <module_name>
    """
```

2. Supported Dtype:
```python
def _handle_supported_dtype(inputs: List[str]) -> Dict:
    """Handle datatype support pragma
    Format: @brainsmith supported_dtype <signal> <type> <min> [max]
    """
```

3. Derived Parameter:
```python
def _handle_derived_param(inputs: List[str]) -> Dict:
    """Handle derived parameter pragma
    Format: @brainsmith derived_param <function> <param1> [param2 ...]
    """
```

### 3. Integration Points

#### A. Parser Updates (`parser.py`)
1. Extend RTLParser to collect pragmas:
```python
def parse_file(self, filepath: str) -> HWKernel:
    # Existing parsing logic...
    
    # Extract pragmas
    pragmas = extract_pragmas(tree.root_node)
    
    # Process parameters with pragmas
    processed_params = self.param_processor.process_parameters(
        parameters, pragmas
    )
    
    # Create kernel with processed params
    kernel = HWKernel(
        name=name,
        parameters=processed_params,
        ports=ports,
        interfaces=interfaces,
        pragmas=pragmas
    )
```

2. Add param_processor initialization:
```python
def __init__(self):
    self.interface_builder = InterfaceBuilder()
    self.param_processor = ParameterProcessor()
```

#### B. Data Type Integration
1. Interface Model Updates:
- Add datatype support tracking to Interface class
- Validate datatype restrictions during interface building

2. Parameter Type Validation:
- Define supported parameter types
- Add type checking to parameter processing

## Implementation Order

1. Parameter Processing
   - Update data models
   - Implement basic parameter processor
   - Add parameter validation

2. Pragma Updates  
   - Replace pragma types
   - Implement new handlers
   - Add pragma validation

3. Integration
   - Update parser
   - Connect parameter processing
   - Add datatype support

4. Testing
   - Unit tests for new components
   - Integration tests with example RTL
   - Validation tests for error cases

## Testing Strategy

### 1. Unit Tests
- Parameter conversion
- Pragma parsing
- Validation logic

### 2. Integration Tests
- Full parser pipeline
- Example RTL files
- Error handling

### 3. Validation Tests
- Invalid pragmas
- Parameter conflicts
- Type mismatches