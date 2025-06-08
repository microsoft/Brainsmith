# HWKG Phase 2 Week 1: Implementation Examples

## Table of Contents
1. [Configuration Usage](#configuration-usage)
2. [Template Context Building](#template-context-building)
3. [Template Management](#template-management)
4. [Generator Implementation](#generator-implementation)
5. [Complete Workflow Example](#complete-workflow-example)
6. [Error Handling](#error-handling)
7. [Testing Patterns](#testing-patterns)

## Configuration Usage

### Creating Configuration from Command Line Arguments

```python
# In hkg.py or CLI handler
from brainsmith.tools.hw_kernel_gen.config import PipelineConfig, GeneratorType

# From argparse arguments
args = parser.parse_args()
config = PipelineConfig.from_args(args)

# Or create with defaults for HW Custom Op
config = PipelineConfig.from_defaults(GeneratorType.HW_CUSTOM_OP)
```

### Loading Configuration from File

```python
# Load from JSON configuration file
config = PipelineConfig.from_file("config/hw_custom_op_config.json")

# Example configuration file content:
{
    "generator_type": "hw_custom_op",
    "template_config": {
        "base_dirs": ["templates/hw_custom_op", "templates/common"],
        "cache_templates": true,
        "cache_size": 100,
        "cache_ttl": 3600
    },
    "generation_config": {
        "output_dir": "output/generated",
        "overwrite": false,
        "generate_tb": true,
        "target_language": "python"
    }
}
```

### Programmatic Configuration

```python
from pathlib import Path
from brainsmith.tools.hw_kernel_gen.config import (
    PipelineConfig, TemplateConfig, GenerationConfig,
    AnalysisConfig, ValidationConfig, GeneratorType
)

# Build configuration programmatically
template_config = TemplateConfig(
    base_dirs=[Path("templates/hw_custom_op")],
    custom_templates={
        "main": Path("custom/my_template.j2")
    },
    cache_templates=True,
    cache_size=50
)

generation_config = GenerationConfig(
    output_dir=Path("output"),
    overwrite=True,
    generate_tb=True,
    tb_type="pytest"
)

config = PipelineConfig(
    generator_type=GeneratorType.HW_CUSTOM_OP,
    template_config=template_config,
    generation_config=generation_config
)

# Validate before use
config.validate()

# Save for reuse
config.to_file("my_config.json")
```

## Template Context Building

### Building HW Custom Op Context

```python
from brainsmith.tools.hw_kernel_gen.template_context import (
    TemplateContextBuilder, HWCustomOpContext
)
from brainsmith.tools.hw_kernel_gen.data_structures import (
    ParsedRTLData, RTLModule, RTLInterface, RTLSignal
)

# Create context builder
builder = TemplateContextBuilder(config)

# Create parsed RTL data (normally from RTL parser)
module = RTLModule(
    name="MatMul_8x8",
    parameters={"WIDTH": 8, "DEPTH": 8},
    interfaces=[
        RTLInterface(
            name="s_axi",
            interface_type="axi",
            signals=[
                RTLSignal("s_axi_aclk", "input", 1, "clock"),
                RTLSignal("s_axi_aresetn", "input", 1, "reset"),
                RTLSignal("s_axi_awaddr", "input", 32),
                RTLSignal("s_axi_awvalid", "input", 1),
                RTLSignal("s_axi_awready", "output", 1)
            ]
        )
    ]
)

parsed_data = ParsedRTLData(
    modules=[module],
    top_module="MatMul_8x8"
)

# Optional FINN config
finn_config = {
    "function": "matmul",
    "domain": "finn"
}

# Build context
context = builder.build_hw_custom_op_context(
    parsed_rtl=parsed_data,
    finn_config=finn_config,
    config=config
)

# Access context data
print(f"Class name: {context.class_name}")  # MatMul8x8Op
print(f"Has AXI: {context.has_axi_interfaces}")  # True
print(f"AXI interfaces: {context.get_axi_interfaces()}")
```

### Context Caching

```python
# Context builder automatically caches results
context1 = builder.build_hw_custom_op_context(parsed_rtl, finn_config)
context2 = builder.build_hw_custom_op_context(parsed_rtl, finn_config)
# context2 is returned from cache

# Check cache statistics
stats = builder.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")

# Clear cache when needed
builder.clear_cache()
```

## Template Management

### Basic Template Rendering

```python
from brainsmith.tools.hw_kernel_gen.template_manager import create_template_manager

# Create template manager
template_manager = create_template_manager(config.template_config)

# Render a template
context = {"module_name": "MatMul", "width": 8}
output = template_manager.render_template("hw_custom_op.py.j2", context)

# Render from string
template_str = "class {{ class_name }}Op:\n    pass"
output = template_manager.render_string(template_str, {"class_name": "MatMul"})
```

### Using Custom Templates

```python
# Configure custom templates
template_config = TemplateConfig(
    base_dirs=[Path("templates")],
    custom_templates={
        "hw_custom_op.py.j2": Path("my_templates/custom_op.j2"),
        "testbench.py.j2": Path("my_templates/custom_tb.j2")
    }
)

template_manager = create_template_manager(template_config)

# Custom template takes precedence
output = template_manager.render_template("hw_custom_op.py.j2", context)
```

### Template Discovery and Management

```python
# List available templates
templates = template_manager.list_templates()
print(f"Available templates: {templates}")

# Check if template exists
if template_manager.template_exists("hw_custom_op.py.j2"):
    output = template_manager.render_template("hw_custom_op.py.j2", context)

# Clear template cache
template_manager.clear_cache()

# Reload templates (useful during development)
template_manager.reload_templates()

# Get cache statistics
stats = template_manager.get_stats()
print(f"Templates cached: {stats['templates_cached']}")
print(f"Cache size: {stats['cache_size_bytes']} bytes")
```

## Generator Implementation

### Creating a Custom Generator

```python
from brainsmith.tools.hw_kernel_gen.generator_base import (
    GeneratorBase, GenerationResult, create_generation_result
)
from brainsmith.tools.hw_kernel_gen.data_structures import PipelineInputs

class MyCustomGenerator(GeneratorBase):
    """Example custom generator implementation."""
    
    def generate(self, inputs: PipelineInputs) -> GenerationResult:
        """Generate code from inputs."""
        # Create result container
        result = create_generation_result()
        
        try:
            # Build context
            context = self._build_context(inputs.parsed_rtl)
            
            # Render main template
            main_content = self._render_template(
                "my_template.py.j2",
                context
            )
            
            # Create artifact
            artifact = self._create_artifact(
                file_name=f"{context.module_name}_generated.py",
                content=main_content,
                artifact_type="python"
            )
            
            # Validate and add to results
            if artifact.validate():
                result.add_artifact(artifact)
            else:
                result.add_error(f"Validation failed for {artifact.file_name}")
            
            # Add metrics
            result.metrics["templates_rendered"] = 1
            result.metrics["generation_time"] = self._elapsed_time
            
        except Exception as e:
            result.success = False
            result.add_error(f"Generation failed: {str(e)}")
        
        return result
```

### Using the Generator

```python
# Initialize generator with dependencies
generator = MyCustomGenerator(
    config=config,
    template_manager=template_manager,
    context_builder=builder
)

# Prepare inputs
inputs = PipelineInputs(
    parsed_rtl=parsed_data,
    finn_config=finn_config,
    config=config
)

# Generate code
result = generator.generate(inputs)

# Check results
if result.success:
    print(f"Generated {len(result.artifacts)} files")
    
    # Write all artifacts
    result.write_all_artifacts(base_dir=Path("output"))
    
    # Or handle individually
    for artifact in result.artifacts:
        print(f"Generated: {artifact.file_name}")
        artifact.write_to_file(Path("output"))
else:
    print(f"Generation failed: {result.errors}")
```

## Complete Workflow Example

Here's a complete example showing how all components work together:

```python
from pathlib import Path
from brainsmith.tools.hw_kernel_gen.config import (
    PipelineConfig, GeneratorType, create_default_config
)
from brainsmith.tools.hw_kernel_gen.template_manager import create_template_manager
from brainsmith.tools.hw_kernel_gen.template_context import TemplateContextBuilder
from brainsmith.tools.hw_kernel_gen.data_structures import (
    PipelineInputs, ParsedRTLData, RTLModule, RTLInterface, RTLSignal
)
# Assume we have a HWCustomOpGenerator implementation
from brainsmith.tools.hw_kernel_gen.generators import HWCustomOpGenerator

def generate_hw_custom_op(rtl_file: Path, output_dir: Path):
    """Complete workflow for generating HW Custom Op."""
    
    # 1. Create configuration
    config = create_default_config(GeneratorType.HW_CUSTOM_OP)
    config.generation_config.output_dir = output_dir
    config.generation_config.generate_tb = True
    
    # 2. Parse RTL (placeholder - would use actual parser)
    parsed_rtl = ParsedRTLData(
        modules=[
            RTLModule(
                name="MyAccelerator",
                parameters={"WIDTH": 32},
                interfaces=[
                    RTLInterface(
                        name="s_axi",
                        interface_type="axi",
                        signals=[
                            RTLSignal("clk", "input", 1, "clock"),
                            RTLSignal("rst", "input", 1, "reset"),
                            # ... more signals
                        ]
                    )
                ]
            )
        ],
        top_module="MyAccelerator"
    )
    
    # 3. Create components
    template_manager = create_template_manager(config.template_config)
    context_builder = TemplateContextBuilder(config)
    
    # 4. Initialize generator
    generator = HWCustomOpGenerator(
        config=config,
        template_manager=template_manager,
        context_builder=context_builder
    )
    
    # 5. Prepare inputs
    inputs = PipelineInputs(
        parsed_rtl=parsed_rtl,
        config=config,
        metadata={
            "rtl_file": str(rtl_file),
            "timestamp": "2024-01-01T00:00:00"
        }
    )
    
    # 6. Generate code
    result = generator.generate(inputs)
    
    # 7. Handle results
    if result.success:
        print(f"✅ Successfully generated {len(result.artifacts)} files:")
        
        # Validate all artifacts
        if result.validate_all_artifacts():
            # Write to disk
            result.write_all_artifacts(output_dir)
            
            for artifact in result.artifacts:
                print(f"  - {artifact.file_name} ({artifact.artifact_type})")
        else:
            print("❌ Artifact validation failed")
            
        # Display metrics
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  - {key}: {value}")
            
    else:
        print(f"❌ Generation failed:")
        for error in result.errors:
            print(f"  - {error}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
    
    return result

# Execute the workflow
if __name__ == "__main__":
    result = generate_hw_custom_op(
        rtl_file=Path("rtl/my_accelerator.v"),
        output_dir=Path("output/generated")
    )
```

## Error Handling

### Using the Error Framework

```python
from brainsmith.tools.hw_kernel_gen.errors import (
    BrainsmithError, ConfigurationError, TemplateError,
    CodeGenerationError
)

# Configuration validation
try:
    config = PipelineConfig.from_file("config.json")
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Context: {e.context}")
    if e.suggestions:
        print(f"Suggestions: {e.suggestions}")

# Template rendering with error handling
try:
    output = template_manager.render_template("template.j2", context)
except TemplateError as e:
    print(f"Template error: {e}")
    # Access original exception if needed
    if e.__cause__:
        print(f"Caused by: {e.__cause__}")

# Generator error handling
class SafeGenerator(GeneratorBase):
    def generate(self, inputs: PipelineInputs) -> GenerationResult:
        result = create_generation_result()
        
        try:
            # Generation logic
            context = self._build_context(inputs.parsed_rtl)
            content = self._render_template("template.j2", context)
            # ...
            
        except TemplateError as e:
            result.success = False
            result.add_error(f"Template error: {e}")
            
        except CodeGenerationError as e:
            result.success = False
            result.add_error(f"Generation error: {e}")
            
        except Exception as e:
            # Wrap unexpected errors
            error = CodeGenerationError(
                f"Unexpected error during generation: {e}",
                context={"inputs": inputs.metadata}
            )
            result.success = False
            result.add_error(str(error))
            
        return result
```

## Testing Patterns

### Testing Configuration

```python
import pytest
from brainsmith.tools.hw_kernel_gen.config import (
    PipelineConfig, TemplateConfig, GeneratorType
)

def test_config_validation():
    """Test configuration validation."""
    # Valid configuration
    config = PipelineConfig.from_defaults(GeneratorType.HW_CUSTOM_OP)
    config.validate()  # Should not raise
    
    # Invalid configuration
    config.generation_config.indent_size = -1
    with pytest.raises(ValueError, match="indent_size must be positive"):
        config.validate()

def test_config_serialization(tmp_path):
    """Test configuration save/load."""
    # Create config
    config = PipelineConfig.from_defaults(GeneratorType.RTL_BACKEND)
    
    # Save to file
    config_file = tmp_path / "config.json"
    config.to_file(config_file)
    
    # Load and compare
    loaded = PipelineConfig.from_file(config_file)
    assert loaded.generator_type == config.generator_type
    assert loaded.to_dict() == config.to_dict()
```

### Testing Template Context

```python
from brainsmith.tools.hw_kernel_gen.template_context import (
    TemplateContextBuilder, HWCustomOpContext
)

def test_context_building():
    """Test context building with caching."""
    builder = TemplateContextBuilder()
    
    # Build context
    context1 = builder.build_hw_custom_op_context(
        parsed_rtl=create_test_rtl(),
        finn_config={"function": "conv2d"}
    )
    
    # Verify context
    assert isinstance(context1, HWCustomOpContext)
    assert context1.module_name == "Conv2D"
    assert context1.class_name == "Conv2DCustomOp"
    
    # Test caching
    stats_before = builder.get_cache_stats()
    context2 = builder.build_hw_custom_op_context(
        parsed_rtl=create_test_rtl(),
        finn_config={"function": "conv2d"}
    )
    stats_after = builder.get_cache_stats()
    
    assert stats_after["hits"] == stats_before["hits"] + 1
    assert context1 is context2  # Same object from cache
```

### Testing Generators

```python
from brainsmith.tools.hw_kernel_gen.generator_base import GenerationResult

class TestGenerator:
    def test_generator_success(self, test_generator, test_inputs):
        """Test successful generation."""
        result = test_generator.generate(test_inputs)
        
        assert result.success
        assert len(result.artifacts) > 0
        assert len(result.errors) == 0
        
        # Verify artifacts
        for artifact in result.artifacts:
            assert artifact.validate()
            assert artifact.content
            assert artifact.file_name
    
    def test_generator_error_handling(self, test_generator):
        """Test generator error handling."""
        # Create invalid inputs
        invalid_inputs = PipelineInputs(
            parsed_rtl=None,  # Invalid
            config=create_default_config()
        )
        
        result = test_generator.generate(invalid_inputs)
        
        assert not result.success
        assert len(result.errors) > 0
        assert len(result.artifacts) == 0
```

### Integration Testing

```python
def test_end_to_end_workflow(tmp_path):
    """Test complete workflow integration."""
    # Setup
    config = create_default_config(GeneratorType.HW_CUSTOM_OP)
    config.generation_config.output_dir = tmp_path
    
    template_manager = create_template_manager(config.template_config)
    context_builder = TemplateContextBuilder(config)
    
    # Create generator
    generator = HWCustomOpGenerator(
        config=config,
        template_manager=template_manager,
        context_builder=context_builder
    )
    
    # Generate
    inputs = create_test_inputs()
    result = generator.generate(inputs)
    
    # Verify
    assert result.success
    assert result.validate_all_artifacts()
    
    # Write files
    result.write_all_artifacts(tmp_path)
    
    # Check files exist
    for artifact in result.artifacts:
        file_path = tmp_path / artifact.file_name
        assert file_path.exists()
        assert file_path.read_text() == artifact.content
```

## Summary

These implementation examples demonstrate:

1. **Configuration**: Multiple ways to create and manage configurations
2. **Context Building**: How to build template contexts with caching
3. **Template Management**: Template rendering, caching, and customization
4. **Generator Implementation**: Creating custom generators using the framework
5. **Complete Workflow**: End-to-end integration of all components
6. **Error Handling**: Proper error handling using the error framework
7. **Testing**: Comprehensive testing patterns for all components

The Week 1 architecture provides a solid foundation that is both powerful and easy to use, with clear patterns for extension and customization.