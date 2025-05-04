# RTL Parser Implementation Analysis

## Component Overview

### 1. Data Structures (`data.py`)
Core data model definitions that represent SystemVerilog constructs.

**Key Classes:**
- `Direction` enum: INPUT/OUTPUT/INOUT
- `Parameter`: Module parameters with validation
- `Port`: Module ports with direction and width
- `Pragma`: Special @brainsmith comments
- `HWKernel`: Top-level container for parsed module

**Design Choices:**
- Uses `@dataclass` for clean initialization and representation
- Strong validation in `__post_init__` methods
- Immutable data structures with explicit modification methods
- Type hints throughout for IDE support

### 2. Pragma Processing (`pragma.py`)
Handles special @brainsmith comments that guide code generation.

**Features:**
- Custom pragma types (INTERFACE, PARAMETER, RESOURCE, etc.)
- Type-specific validation and processing
- Structured data extraction
- Line number tracking for error reporting

**Handler Architecture:**
```python
handlers = {
    'interface': _handle_interface,
    'parameter': _handle_parameter,
    'resource': _handle_resource,
    'timing': _handle_timing,
    'feature': _handle_feature
}
```

### 3. Main Parser (`parser.py`)
Core parsing logic using tree-sitter for SystemVerilog analysis.

**Key Features:**
- Grammar loading and initialization
- File-level parsing
- Module discovery
- Error handling and reporting

**Processing Flow:**
1. Initialize tree-sitter with SystemVerilog grammar
2. Parse file to AST
3. Find module declaration
4. Extract components (parameters, ports, pragmas)
5. Construct HWKernel instance

### 4. Interface Analysis (`interface.py`)
Specialized parsing for module interface components.

**Main Functions:**
- `parse_port_declaration()`: Complex port parsing
- `parse_parameter_declaration()`: Parameter extraction
- `extract_module_header()`: Module interface analysis

**Helper Utilities:**
- `_find_child()`: AST node search
- `_has_text()`: Text content search
- `_debug_node()`: Debug visualization

## Inter-component Relationships

### Data Flow
```
Source File
    │
    ▼
RTLParser (parser.py)
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
Interface         Parameter          Pragma
Analysis          Parsing           Processing
(interface.py)    (interface.py)    (pragma.py)
    │                 │                 │
    └─────────────────┴─────────────────┘
                      │
                      ▼
                  HWKernel
                  (data.py)
```

### Key Interactions

1. **Parser → Interface**
   - Parser calls interface functions to process declarations
   - Interface returns structured data objects
   - Strong separation of concerns

2. **Interface → Data**
   - Interface creates Parameter/Port instances
   - Validates data during construction
   - Maintains immutability

3. **Parser → Pragma**
   - Parser finds comments in AST
   - Pragma processor validates and extracts data
   - Results stored in HWKernel

4. **All → Data**
   - Central data models used throughout
   - Consistent validation
   - Type safety

## Implementation Highlights

### 1. Robust Port Parsing
```python
def parse_port_declaration(node):
    # Multiple strategies for finding direction
    direction = (
        _find_first_word() or
        _find_direction_node() or
        _search_text_content()
    )
    
    # Width extraction preserves expressions
    width = extract_width_expression()
    
    # Name finding with context
    name = find_port_name(excluding_width=width)
    
    return Port(name, direction, width)
```

### 2. Flexible Pragma System
```python
class PragmaParser:
    def parse_comment(self, node, line):
        # Format: @brainsmith <type> <inputs...>
        if not is_brainsmith_pragma(node):
            return None
            
        type, *inputs = parse_pragma_parts(node)
        return process_with_handler(type, inputs)
```

### 3. Smart AST Navigation
```python
def _find_child(node, type_names, recursive=True):
    # Direct children first
    for child in node.children:
        if matches_type(child, type_names):
            return child
            
    # Then recursive if needed
    if recursive:
        return search_recursive(node, type_names)
```

## Error Handling Strategy

1. **Validation Layers**
   - Data class validation
   - Parser syntax checking
   - Pragma format verification
   - Interface component validation

2. **Error Types**
   - `ParserError`: Base class
   - `SyntaxError`: Invalid SystemVerilog
   - `PragmaError`: Invalid pragma format
   - `ValueError`: Invalid data values

3. **Debug Support**
   - Detailed logging
   - AST visualization
   - Line number tracking
   - Error context

## Extensibility Points

1. **New Pragma Types**
   - Add to PragmaType enum
   - Implement handler function
   - Register in handlers dict

2. **Additional Port Types**
   - Extend Direction enum
   - Add parsing logic to interface.py
   - Update validation as needed

3. **Custom Parameters**
   - Add parameter types
   - Extend parameter parsing
   - Update validation rules

## Testing Strategy

1. **Unit Tests**
   - Parser initialization
   - Individual component parsing
   - Error cases
   - Edge cases

2. **Integration Tests**
   - Full file parsing
   - Complex modules
   - Real-world examples

3. **Validation Tests**
   - Syntax variations
   - Error recovery
   - Data consistency