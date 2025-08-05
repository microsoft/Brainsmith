# Code Generation System - Recommendations

## Context

The Code Generation System successfully produces FINN-compatible Python operators from parsed RTL metadata. It uses a generator pattern with Jinja2 templates to create multiple output files. While the current implementation works well, there are opportunities to improve extensibility, reduce template complexity, and enhance maintainability.

## Current Strengths

- **Generator Pattern**: Clean abstraction with `GeneratorBase` and auto-discovery
- **Template-Driven**: Flexible Jinja2 templates allow easy customization
- **CodegenBinding System**: Excellent parameter resolution abstraction
- **Clear Output**: Generated code is readable and well-documented
- **Separation of Concerns**: Templates separate from generation logic

## Recommendations

### 1. Implement True Plugin Architecture

**Current Issue**: Generators must be in the generators/ directory and are imported by name.

**Solution**:
```python
# generators/plugin_registry.py
from importlib.metadata import entry_points
from typing import Dict, Type
from .base import GeneratorBase

class GeneratorRegistry:
    """Plugin-based generator registry using entry points"""
    
    def __init__(self):
        self._generators: Dict[str, Type[GeneratorBase]] = {}
        self._load_builtin_generators()
        self._load_plugin_generators()
    
    def _load_plugin_generators(self):
        """Load generators from installed packages"""
        # Entry point: brainsmith.generators
        for entry_point in entry_points(group='brainsmith.generators'):
            try:
                generator_class = entry_point.load()
                self.register(entry_point.name, generator_class)
            except Exception as e:
                logger.warning(f"Failed to load generator {entry_point.name}: {e}")
    
    def register(self, name: str, generator_class: Type[GeneratorBase]):
        """Register a generator"""
        if not issubclass(generator_class, GeneratorBase):
            raise ValueError(f"{generator_class} must inherit from GeneratorBase")
        self._generators[name] = generator_class
```

**Plugin Setup (setup.py)**:
```python
setup(
    name="my-brainsmith-plugin",
    entry_points={
        'brainsmith.generators': [
            'my_custom_gen = my_plugin.generators:MyCustomGenerator',
        ],
    },
)
```

**Benefit**: External packages can provide custom generators without modifying core code.

### 2. Simplify Template Complexity

**Current Issue**: Templates contain complex logic that makes them hard to maintain.

**Solution**: Move logic to template helpers and use template inheritance.

```python
# templates/helpers.py
class TemplateHelpers:
    """Helper functions for templates"""
    
    @staticmethod
    def format_parameter_value(param: Parameter) -> str:
        """Format parameter value for Python code"""
        if param.type == "string":
            return f'"{param.value}"'
        elif param.type == "list":
            return str(param.value)
        else:
            return str(param.value)
    
    @staticmethod
    def generate_interface_properties(interface: InterfaceMetadata) -> Dict[str, Any]:
        """Generate interface property dictionary"""
        return {
            "data_type": interface.data_type.to_finn_string(),
            "data_layout": interface.get_layout_string(),
            "stream_width": interface.get_stream_width(),
        }
```

**Base Template (templates/base_hw_custom_op.j2)**:
```jinja
{# Base template for HWCustomOp classes #}
{% block imports %}
from finn.custom_op.fpgadataflow.hwcustomop_codegen import HWCustomOp
{% endblock %}

{% block class_definition %}
class {{ class_name }}(HWCustomOp):
    {% block class_body %}{% endblock %}
{% endblock %}
```

**Derived Template (templates/hw_custom_op.py.j2)**:
```jinja
{% extends "base_hw_custom_op.j2" %}

{% block class_body %}
    def __init__(self):
        super().__init__()
        {% for param in helpers.get_sorted_parameters(parameters) %}
        self.{{ param.name }} = {{ helpers.format_parameter_value(param) }}
        {% endfor %}
{% endblock %}
```

**Benefit**: Cleaner templates, reusable components, easier maintenance.

### 3. Add Template Validation and Testing

**Current Issue**: No validation that templates produce valid Python code.

**Solution**:
```python
# generators/template_validator.py
import ast
import black
from typing import List, Tuple

class TemplateValidator:
    """Validate generated code from templates"""
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Check if generated code is valid Python"""
        errors = []
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
    
    def validate_code_style(self, code: str) -> Tuple[bool, str]:
        """Format code with black and check style"""
        try:
            formatted = black.format_str(code, mode=black.Mode())
            return True, formatted
        except Exception as e:
            return False, str(e)
    
    def validate_finn_compatibility(self, code: str) -> Tuple[bool, List[str]]:
        """Check FINN-specific requirements"""
        errors = []
        tree = ast.parse(code)
        
        # Check for required methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("HWCustomOp"):
                required_methods = ["get_nodeattr_types", "execute_node", "compile"]
                class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                for method in required_methods:
                    if method not in class_methods:
                        errors.append(f"Missing required method: {method}")
        
        return len(errors) == 0, errors
```

**Test Framework**:
```python
# tests/generators/test_templates.py
class TestTemplateGeneration:
    """Test template generation with various contexts"""
    
    @pytest.fixture
    def template_contexts(self):
        """Various template contexts for testing"""
        return [
            # Minimal context
            TemplateContext(kernel_name="simple", ...),
            # Complex context with all features
            TemplateContext(kernel_name="complex", ...),
            # Edge cases
            TemplateContext(kernel_name="edge_case", ...)
        ]
    
    def test_template_generates_valid_python(self, template_contexts):
        """Ensure all templates generate valid Python"""
        generator = HWCustomOpGenerator()
        validator = TemplateValidator()
        
        for context in template_contexts:
            code = generator.render_template(context)
            valid, errors = validator.validate_python_syntax(code)
            assert valid, f"Invalid Python for {context.kernel_name}: {errors}"
```

**Benefit**: Catch template errors early, ensure generated code quality.

### 4. Enhance CodegenBinding with Debug Support

**Current Issue**: Complex parameter resolution is hard to debug.

**Solution**:
```python
# codegen_binding.py
@dataclass
class BindingDebugInfo:
    """Debug information for parameter binding"""
    parameter_name: str
    source_type: SourceType
    resolution_path: List[str]
    resolved_value: Any
    resolution_time_ms: float

class CodegenBinding:
    """Enhanced with debug support"""
    
    def __init__(self, enable_debug: bool = False):
        self._bindings = {}
        self._enable_debug = enable_debug
        self._debug_info: List[BindingDebugInfo] = []
    
    def resolve_parameter(self, name: str, context: Dict) -> Any:
        """Resolve parameter with debug tracking"""
        start_time = time.time()
        resolution_path = []
        
        if self._enable_debug:
            resolution_path.append(f"Resolving parameter: {name}")
        
        # Resolution logic with path tracking
        value = self._resolve_with_tracking(name, context, resolution_path)
        
        if self._enable_debug:
            self._debug_info.append(BindingDebugInfo(
                parameter_name=name,
                source_type=self._bindings[name].source.type,
                resolution_path=resolution_path,
                resolved_value=value,
                resolution_time_ms=(time.time() - start_time) * 1000
            ))
        
        return value
    
    def get_debug_report(self) -> str:
        """Generate human-readable debug report"""
        report = ["Parameter Resolution Debug Report", "=" * 40]
        
        for info in self._debug_info:
            report.extend([
                f"\nParameter: {info.parameter_name}",
                f"Source Type: {info.source_type.value}",
                f"Resolution Time: {info.resolution_time_ms:.2f}ms",
                f"Resolved Value: {info.resolved_value}",
                "Resolution Path:"
            ])
            for step in info.resolution_path:
                report.append(f"  - {step}")
        
        return "\n".join(report)
```

**Benefit**: Easier debugging of parameter resolution issues.

### 5. Implement Generator Composition

**Current Issue**: Each generator is independent, limiting code reuse.

**Solution**:
```python
# generators/composition.py
class ComposableGenerator(GeneratorBase):
    """Base class for composable generators"""
    
    def __init__(self):
        super().__init__()
        self._components: List[GeneratorComponent] = []
    
    def add_component(self, component: GeneratorComponent):
        """Add a reusable component"""
        self._components.append(component)
    
    def generate(self, context: TemplateContext) -> str:
        """Generate using all components"""
        result = self.render_template(context)
        
        for component in self._components:
            result = component.process(result, context)
        
        return result

class ImportOptimizer(GeneratorComponent):
    """Component to optimize Python imports"""
    
    def process(self, code: str, context: TemplateContext) -> str:
        """Remove unused imports and sort them"""
        # Implementation
        pass

class DocstringGenerator(GeneratorComponent):
    """Component to enhance docstrings"""
    
    def process(self, code: str, context: TemplateContext) -> str:
        """Add comprehensive docstrings"""
        # Implementation
        pass
```

**Benefit**: Reusable components across generators, cleaner code.

### 6. Add Template Context Validation

**Current Issue**: Invalid contexts can cause cryptic template errors.

**Solution**:
```python
# templates/context_validator.py
from typing import Dict, List, Any
import jsonschema

class ContextValidator:
    """Validate template contexts before rendering"""
    
    SCHEMAS = {
        "hw_custom_op": {
            "type": "object",
            "required": ["kernel_name", "class_name", "parameters"],
            "properties": {
                "kernel_name": {"type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
                "class_name": {"type": "string", "pattern": "^[A-Z][a-zA-Z0-9]*$"},
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "value", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "value": {},
                            "type": {"type": "string", "enum": ["int", "float", "string", "list"]}
                        }
                    }
                }
            }
        }
    }
    
    def validate(self, template_name: str, context: Dict[str, Any]) -> List[str]:
        """Validate context against schema"""
        errors = []
        
        if template_name not in self.SCHEMAS:
            return [f"No schema defined for template: {template_name}"]
        
        try:
            jsonschema.validate(context, self.SCHEMAS[template_name])
        except jsonschema.ValidationError as e:
            errors.append(f"Context validation failed: {e.message}")
        
        return errors
```

**Benefit**: Clear error messages when context is invalid, preventing confusing template errors.

### 7. Implement Incremental Generation

**Current Issue**: Regenerating all files even when only one template changes.

**Solution**:
```python
# generators/incremental.py
class IncrementalGenerationManager:
    """Manage incremental code generation"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def should_regenerate(self, 
                         generator: GeneratorBase,
                         context: TemplateContext,
                         output_path: Path) -> bool:
        """Check if regeneration is needed"""
        
        # Check if output exists
        if not output_path.exists():
            return True
        
        # Check if template changed
        template_mtime = self._get_template_mtime(generator)
        output_mtime = output_path.stat().st_mtime
        
        if template_mtime > output_mtime:
            return True
        
        # Check if context changed
        context_hash = self._compute_context_hash(context)
        cached_hash = self._get_cached_context_hash(generator, output_path)
        
        return context_hash != cached_hash
    
    def mark_generated(self,
                      generator: GeneratorBase,
                      context: TemplateContext,
                      output_path: Path):
        """Mark file as generated with current context"""
        context_hash = self._compute_context_hash(context)
        self._save_context_hash(generator, output_path, context_hash)
```

**Benefit**: Faster regeneration during development, only updating changed files.

## Implementation Priority

1. **High Priority**:
   - Template validation and testing (ensures correctness)
   - Simplify template complexity (improves maintainability)
   - CodegenBinding debug support (helps users)

2. **Medium Priority**:
   - True plugin architecture (enables extensibility)
   - Template context validation (better error messages)
   - Generator composition (code reuse)

3. **Low Priority**:
   - Incremental generation (optimization)

## Expected Outcomes

- **Better Extensibility**: Plugin architecture allows custom generators
- **Improved Maintainability**: Simpler templates with validation
- **Enhanced Developer Experience**: Debug support and clear errors
- **Higher Code Quality**: Validated and tested output
- **Faster Development**: Incremental generation and component reuse