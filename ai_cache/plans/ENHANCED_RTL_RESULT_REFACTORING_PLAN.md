# EnhancedRTLParsingResult Refactoring Plan

## Executive Summary
The current EnhancedRTLParsingResult is a 500+ line monstrosity doing work that either:
1. The RTL parser already does (interface categorization)
2. Should be done by the template renderer (context building)
3. Shouldn't exist at all (complexity estimation)

This plan refactors it to a ~50 line data container and pushes logic to the right places.

## Current Problems

### 1. Redundant Interface Categorization
```python
# Current: EnhancedRTLParsingResult re-implements what RTL parser already does
def _map_rtl_interface_to_category(self, interface) -> str:
    if interface.type == InterfaceType.AXI_LITE:
        return "config"
    # ... 50+ lines of logic RTL parser already handles
```

**Reality**: RTL parser already sets `interface.type` and `interface.metadata['direction']`

### 2. Dynamic Class Creation Inside Methods
```python
# Current: Creating classes inside methods (WTF)
class InterfaceTypeObj:
    def __init__(self, value):
        self.value = value
        
class TemplateInterface:
    def __init__(self, name, iface, category, enhanced_result):
        # ... 20+ attributes
```

**Reality**: Should be simple data dictionaries or proper dataclasses

### 3. Business Logic in Data Container
```python
# Current: Complex calculations that don't belong here
def _estimate_kernel_complexity(self) -> str:
    interface_count = len(self.interfaces)
    param_count = len(self.parameters)
    if interface_count <= 2 and param_count <= 3:
        return 'low'
    # ...
```

**Reality**: Templates don't even use this. Delete it.

### 4. Template Context Building in Wrong Place
```python
# Current: 100+ lines of template context generation
def get_template_context(self) -> Dict[str, Any]:
    self._template_context = {
        # 30+ different keys with complex transformations
    }
```

**Reality**: This belongs in the template renderer

## Refactoring Strategy

### Phase 1: Enhance RTL Parser (Minimal Changes)

#### 1.1 Add Interface Category to Metadata
```python
# In interface_builder.py, enhance _create_interface():
def _create_interface(self, name: str, type: InterfaceType, 
                     ports: Dict[str, Port], metadata: Dict) -> Interface:
    # Existing code...
    
    # ADD: Explicit category for template use
    if type == InterfaceType.AXI_STREAM:
        if metadata.get('direction') == Direction.INPUT:
            metadata['category'] = 'input'
        elif metadata.get('direction') == Direction.OUTPUT:
            metadata['category'] = 'output'
    elif type == InterfaceType.AXI_LITE:
        metadata['category'] = 'config'
    elif type == InterfaceType.GLOBAL_CONTROL:
        metadata['category'] = 'control'
    
    # ADD: Template-friendly wrapper name
    metadata['template_name'] = self._get_template_name(name, metadata)
    
    return Interface(name, type, ports, validation_result, metadata)
```

#### 1.2 Extract Data Width as Integer
```python
# In protocol_validator.py, enhance _extract_metadata():
def _extract_metadata(self, ports: Dict[str, Port]) -> Dict:
    metadata = {}
    
    # Existing: data_width_expr extraction
    
    # ADD: Parse width to integer for templates
    if 'data_width_expr' in metadata:
        metadata['data_width'] = self._parse_width_to_int(metadata['data_width_expr'])
    
    return metadata

def _parse_width_to_int(self, width_expr: str) -> Optional[int]:
    """Parse width expressions like '[31:0]' to 32."""
    if width_expr.isdigit():
        return int(width_expr)
    
    # Handle [N:0] format
    import re
    match = re.match(r'\[(\d+):0\]', width_expr)
    if match:
        return int(match.group(1)) + 1
    
    # Can't parse - return None (template will use default)
    return None
```

#### 1.3 Create Simple Factory Function
```python
# In rtl_parser/__init__.py:
def create_template_ready_result(rtl_result: RTLParsingResult) -> 'TemplateReadyResult':
    """
    Convert RTLParsingResult to template-ready format.
    
    This is a simple data transformation, not complex processing.
    """
    from .template_ready import TemplateReadyResult
    
    # Categorize interfaces using metadata
    interfaces_by_category = {
        'input': [],
        'output': [],
        'weight': [],
        'config': [],
        'control': []
    }
    
    for name, iface in rtl_result.interfaces.items():
        category = iface.metadata.get('category', 'unknown')
        
        # Check for weight pragma
        for pragma in rtl_result.pragmas:
            if (pragma.type == PragmaType.WEIGHT and 
                pragma.parsed_data.get('interface_name') == name):
                category = 'weight'
                break
        
        if category in interfaces_by_category:
            interfaces_by_category[category].append(iface)
    
    return TemplateReadyResult(
        name=rtl_result.name,
        class_name=_to_class_name(rtl_result.name),
        source_file=rtl_result.source_file,
        interfaces=rtl_result.interfaces,
        interfaces_by_category=interfaces_by_category,
        parameters=rtl_result.parameters,
        pragmas=rtl_result.pragmas,
        metadata={
            'has_weights': len(interfaces_by_category['weight']) > 0,
            'has_inputs': len(interfaces_by_category['input']) > 0,
            'has_outputs': len(interfaces_by_category['output']) > 0,
            'interface_count': len(rtl_result.interfaces),
            'parameter_count': len(rtl_result.parameters)
        }
    )

def _to_class_name(module_name: str) -> str:
    """Convert module_name to ClassName."""
    parts = module_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts)
```

### Phase 2: Create Simple TemplateReadyResult

#### 2.1 New Data Container
```python
# New file: rtl_parser/template_ready.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class TemplateReadyResult:
    """
    Simple data container with pre-processed template data.
    
    NO METHODS. NO LOGIC. JUST DATA.
    """
    # Core identification
    name: str
    class_name: str
    source_file: Optional[Path]
    
    # Raw data from parser
    interfaces: Dict[str, 'Interface']
    parameters: List['Parameter']
    pragmas: List['Pragma']
    
    # Pre-categorized for templates
    interfaces_by_category: Dict[str, List['Interface']]
    
    # Simple metadata
    metadata: Dict[str, any] = field(default_factory=dict)
```

#### 2.2 Replace EnhancedRTLParsingResult
```python
# In rtl_parser/__init__.py:
def parse_rtl_file_enhanced(rtl_file, advanced_pragmas: bool = False) -> TemplateReadyResult:
    """Parse RTL file and return template-ready result."""
    # Get standard result
    rtl_result = parse_rtl_file(rtl_file, advanced_pragmas)
    
    # Convert to template-ready format
    return create_template_ready_result(rtl_result)

# DELETE the 500+ line EnhancedRTLParsingResult class
```

### Phase 3: Move Logic to DirectTemplateRenderer

#### 3.1 Template Context Building
```python
# In direct_renderer.py:
class DirectTemplateRenderer:
    """Renders templates directly from RTL parsing results."""
    
    def build_context(self, template_ready: TemplateReadyResult, 
                     compiler_data: Dict) -> Dict[str, Any]:
        """Build complete template context."""
        return {
            # Basic info
            'kernel_name': template_ready.name,
            'class_name': template_ready.class_name,
            'source_file': str(template_ready.source_file),
            
            # Interfaces with template compatibility
            'interfaces': self._build_interface_context(template_ready),
            'input_interfaces': self._wrap_interfaces(template_ready.interfaces_by_category['input']),
            'output_interfaces': self._wrap_interfaces(template_ready.interfaces_by_category['output']),
            'weight_interfaces': self._wrap_interfaces(template_ready.interfaces_by_category['weight']),
            'config_interfaces': self._wrap_interfaces(template_ready.interfaces_by_category['config']),
            
            # Parameters
            'rtl_parameters': self._build_parameter_context(template_ready.parameters),
            
            # Simple counts and flags
            'has_weights': template_ready.metadata['has_weights'],
            'has_inputs': template_ready.metadata['has_inputs'],
            'has_outputs': template_ready.metadata['has_outputs'],
            
            # Compiler data
            'compiler_data': compiler_data,
            
            # For RTL wrapper compatibility
            'kernel': {
                'name': template_ready.name,
                'parameters': template_ready.parameters
            },
            'interfaces_list': list(template_ready.interfaces.values()),
            'InterfaceType': InterfaceType,
        }
    
    def _wrap_interfaces(self, interfaces: List[Interface]) -> List[Dict]:
        """Wrap Interface objects for template compatibility."""
        wrapped = []
        for idx, iface in enumerate(interfaces):
            wrapped.append({
                'name': iface.name,
                'wrapper_name': iface.metadata.get('template_name', f"{iface.metadata['category']}{idx}"),
                'type': iface,  # Original object
                'dtype': self._get_interface_dtype(iface),
                'protocol': iface.type.value.lower(),
                'data_width': iface.metadata.get('data_width', 8),
                'data_width_expr': iface.metadata.get('data_width_expr', ''),
                # Default dimensions
                'tensor_dims': [128],
                'block_dims': [128],
                'stream_dims': [1],
            })
        return wrapped
    
    def _get_interface_dtype(self, interface: Interface) -> Dict:
        """Create simple dtype info for templates."""
        width = interface.metadata.get('data_width', 8)
        return {
            'name': 'UINT',
            'finn_type': f'UINT{width}',
            'bitwidth': width,
            'signed': False,
        }
```

### Phase 4: Delete Unnecessary Code

#### 4.1 Remove from EnhancedRTLParsingResult
- `_estimate_kernel_complexity()` - Templates don't use this
- `_has_resource_estimation_pragmas()` - Not needed
- `_has_verification_pragmas()` - Not needed
- `_infer_kernel_type()` - Not used
- `_extract_datatype_constraints()` - Move to renderer if needed
- All the dynamic class creation nonsense

#### 4.2 Remove from CLI
- Complex fallback logic if we're confident in the new approach
- Legacy compatibility shims after testing

### Phase 5: Update Generator Pipeline

#### 5.1 Simplified Enhanced Generator
```python
# In unified_hwkg/generator.py:
def generate_from_rtl(self, rtl_file: Path, compiler_data: Dict, 
                     output_dir: Path, **options) -> GenerationResult:
    """Generate from RTL using direct pipeline."""
    
    # Parse to template-ready format
    template_ready = parse_rtl_file_enhanced(rtl_file)
    
    # Create renderer
    renderer = DirectTemplateRenderer()
    
    # Build context
    context = renderer.build_context(template_ready, compiler_data)
    
    # Render templates
    files = renderer.render_all(context, output_dir)
    
    return GenerationResult(
        generated_files=files,
        success=True
    )
```

## Implementation Order

1. **Day 1**: Enhance RTL parser (1.1-1.3)
2. **Day 2**: Create TemplateReadyResult and factory
3. **Day 3**: Update DirectTemplateRenderer with context building
4. **Day 4**: Update generator pipeline and test
5. **Day 5**: Delete old code and clean up

## Benefits

1. **Code Reduction**: ~450 lines deleted, ~100 lines added
2. **Performance**: No redundant processing
3. **Clarity**: Each component has one clear job
4. **Maintainability**: Simple data flow, no hidden complexity
5. **Testability**: Pure functions, simple data structures

## Success Metrics

- [ ] EnhancedRTLParsingResult reduced to <50 lines
- [ ] No dynamic class creation
- [ ] No business logic in data containers
- [ ] All tests pass
- [ ] Template generation 20%+ faster

## What We're NOT Doing

1. **NOT** adding complex new features to RTL parser
2. **NOT** breaking existing APIs unnecessarily  
3. **NOT** over-engineering the solution
4. **NOT** adding abstraction layers we don't need

## Summary

This refactoring:
- Uses RTL parser's existing capabilities instead of reimplementing them
- Moves template context building to the renderer where it belongs
- Deletes complexity estimation and other unused "features"
- Results in cleaner, faster, more maintainable code

Total effort: ~5 days
Total code reduction: ~350 lines
Total sanity preserved: 100%