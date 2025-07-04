# Brainsmith Plugin System - Developer Guide

This guide provides comprehensive documentation for developing with the Brainsmith plugin system, including creating plugins, using the registry, and optimizing for production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Creating Plugins](#creating-plugins)
3. [Using Plugins](#using-plugins)
4. [Registry API](#registry-api)
5. [Backend System](#backend-system)
6. [Framework Integration](#framework-integration)
7. [Blueprint Optimization](#blueprint-optimization)
8. [Debugging](#debugging)
9. [Best Practices](#best-practices)
10. [Migration Guide](#migration-guide)

## Quick Start

### Basic Plugin Creation

```python
from brainsmith.core.plugins import transform, kernel, backend, step

# Create a transform
@transform(name="OptimizeModel", stage="topology_opt")
class OptimizeModel:
    def apply(self, model):
        # Your optimization logic
        return model, True  # (modified_model, was_changed)

# Create a kernel
@kernel(name="CustomOp", op_type="CustomOperation")
class CustomOp:
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

# Create a backend
@backend(name="CustomOpHLS", kernel="CustomOp", language="hls", default=True)
class CustomOpHLS(CustomOp):
    def generate_hls(self):
        return "// HLS implementation"

# Create a step
@step(name="ValidateModel", category="testing")
class ValidateModel:
    def __call__(self, build_context):
        # Validation logic
        return build_context
```

### Using Plugins - Multiple Access Patterns

```python
from brainsmith.plugins import transforms, kernels, backends, steps

# Direct attribute access (returns actual classes)
model = model.transform(transforms.OptimizeModel())
kernel_cls = kernels.CustomOp
backend_cls = backends.CustomOpHLS
step_result = steps.ValidateModel()(build_context)

# Dictionary access
model = model.transform(transforms["OptimizeModel"]())
kernel_cls = kernels["CustomOp"]

# Framework-qualified access
model = model.transform(transforms.qonnx.InferDataTypes())
model = model.transform(transforms["finn:Streamline"]())

# Step category access
validation_step = steps.testing.ValidateModel
build_step = steps.build.CompileToHardware
```

## Creating Plugins

### Transform Plugins

Transforms modify ONNX models during compilation.

```python
@transform(name="MyTransform", stage="cleanup", framework="myframework")
class MyTransform:
    """Transform that removes unused nodes."""
    
    def apply(self, model):
        """
        Apply the transform to the model.
        
        Args:
            model: ONNX model to transform
            
        Returns:
            tuple: (modified_model, was_changed)
        """
        # Transform logic here
        modified = False
        # ... transformation code ...
        return model, modified
```

**Available stages:**
- `pre_proc` - Pre-processing operations
- `cleanup` - Graph cleanup operations
- `topology_opt` - Topology optimizations
- `kernel_opt` - Kernel-specific optimizations
- `dataflow_opt` - Dataflow optimizations
- `post_proc` - Post-processing operations

### Kernel Plugins

Kernels define hardware acceleration interfaces.

```python
@kernel(name="MatMul", op_type="MatMul", framework="brainsmith")
class MatMul:
    """Matrix multiplication kernel."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.input_shapes = self.get_input_shapes()
        self.output_shapes = self.get_output_shapes()
    
    def estimate_resources(self):
        """Estimate hardware resource usage."""
        return {"dsps": 100, "brams": 10, "luts": 1000}
```

### Backend Plugins

Backends provide implementation-specific code generation.

```python
@backend(name="MatMulHLS", kernel="MatMul", language="hls", 
         optimization="throughput", resource_usage="high", default=True)
class MatMulHLS(MatMul):
    """High-throughput HLS implementation of MatMul."""
    
    def generate_hls(self):
        """Generate HLS C++ code."""
        return """
        void matmul(
            hls::stream<ap_fixed<16,6>>& in_a,
            hls::stream<ap_fixed<16,6>>& in_b,
            hls::stream<ap_fixed<16,6>>& out
        ) {
            // HLS implementation
        }
        """
    
    def get_resource_estimate(self):
        """Get detailed resource estimates."""
        return {
            "dsps": 120,
            "brams": 15,
            "luts": 1200,
            "ff": 2400
        }

@backend(name="MatMulRTL", kernel="MatMul", language="verilog",
         optimization="area", resource_usage="low")
class MatMulRTL(MatMul):
    """Area-optimized RTL implementation of MatMul."""
    
    def generate_rtl(self):
        """Generate Verilog code."""
        return """
        module matmul (
            input clk,
            input rst,
            // Port definitions
        );
        endmodule
        """
```

### Step Plugins

Steps are build system operations that execute during hardware compilation.

```python
@step(name="GenerateTestbench", category="testing", framework="brainsmith")
class GenerateTestbench:
    """Generate testbench for hardware validation."""
    
    def __call__(self, build_context):
        """
        Execute the step.
        
        Args:
            build_context: Current build context
            
        Returns:
            Updated build context
        """
        # Generate testbench files
        testbench_path = build_context.output_dir / "testbench.cpp"
        with open(testbench_path, 'w') as f:
            f.write(self.generate_testbench_code(build_context.model))
        
        build_context.files["testbench"] = testbench_path
        return build_context
    
    def generate_testbench_code(self, model):
        """Generate the actual testbench code."""
        return "// Testbench implementation"

@step(name="SynthesizeDesign", category="build", framework="xilinx")
class SynthesizeDesign:
    """Run Vivado synthesis."""
    
    def __call__(self, build_context):
        """Run synthesis and update build context with results."""
        # Synthesis logic
        return build_context
```

## Using Plugins

### Multiple Access Patterns

The plugin system supports multiple ways to access plugins:

```python
from brainsmith.plugins import transforms, kernels, backends, steps

# 1. Direct attribute access
transform_cls = transforms.MyTransform
kernel_cls = kernels.MatMul
backend_cls = backends.MatMulHLS
step_cls = steps.GenerateTestbench

# 2. Dictionary access (useful for dynamic lookup)
transform_name = "MyTransform"
transform_cls = transforms[transform_name]

# 3. Framework-qualified attribute access
qonnx_transform = transforms.qonnx.InferDataTypes
finn_transform = transforms.finn.Streamline
custom_kernel = kernels.myframework.CustomKernel

# 4. Framework-qualified dictionary access
transform_cls = transforms["qonnx:InferDataTypes"]
kernel_cls = kernels["finn:MatrixVectorUnit"]

# 5. Category/stage access
# Steps by category
test_step = steps.testing.GenerateTestbench
build_step = steps.build.SynthesizeDesign

# Transforms by stage
cleanup_transforms = transforms.get_by_stage("cleanup")
```

### Plugin Instantiation

All collections return actual classes, not wrapper objects:

```python
# Direct instantiation
transform_instance = transforms.MyTransform()
result = transform_instance.apply(model)

# With parameters
backend_instance = backends.MatMulHLS(config={"optimization": "speed"})
hls_code = backend_instance.generate_hls()

# Step execution
step_instance = steps.GenerateTestbench()
updated_context = step_instance(build_context)
```

## Registry API

### Basic Registry Operations

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# Get plugins by name
transform_cls = registry.get_transform("MyTransform")
kernel_cls = registry.get_kernel("MatMul")
backend_cls = registry.get_backend("MatMulHLS")
step_cls = registry.get_step("GenerateTestbench")

# Get with framework filter
qonnx_transform = registry.get_transform("InferDataTypes", framework="qonnx")
finn_kernel = registry.get_kernel("MatrixVectorUnit", framework="finn")

# List available plugins
transforms_list = registry.list_available_transforms()
kernels_list = registry.list_available_kernels()
steps_list = registry.list_available_steps()

# Get framework-specific plugins
qonnx_transforms = registry.get_framework_transforms("qonnx")
finn_backends = registry.get_framework_backends("finn")
```

### Metadata and Statistics

```python
# Get plugin metadata
metadata = registry.get_plugin_metadata("MatMulHLS")
print(f"Type: {metadata['type']}")
print(f"Kernel: {metadata['kernel']}")
print(f"Language: {metadata['language']}")
print(f"Framework: {metadata['framework']}")

# Get registry statistics
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Transforms: {stats['transforms']}")
print(f"Kernels: {stats['kernels']}")
print(f"Backends: {stats['backends']}")
print(f"Steps: {stats['steps']}")
print(f"Frameworks: {stats['frameworks']}")
```

## Backend System

### Backend Discovery and Selection

```python
from brainsmith.plugins import backends

# Find backends by criteria
hls_backends = backends.find(language="hls")
fast_backends = backends.find(optimization="throughput")
low_resource_backends = backends.find(resource_usage="low")

# Find backends for specific kernel
matmul_backends = backends.find(kernel="MatMul")
matmul_hls = backends.find(kernel="MatMul", language="hls")

# Get backends for kernel (name list)
backend_names = backends.list_for_kernel("MatMul")
print(f"MatMul backends: {backend_names}")

# Get first matching backend class
backend_cls = backends.get_for_kernel("MatMul", language="hls")
if backend_cls:
    backend = backend_cls()
```

### Multiple Backend Strategies

```python
# Register multiple backends with different trade-offs
@backend(name="MatMul_Fast", kernel="MatMul", language="hls",
         optimization="throughput", resource_usage="high", latency="low")
class MatMulFast(MatMul):
    pass

@backend(name="MatMul_Small", kernel="MatMul", language="hls", 
         optimization="area", resource_usage="low", latency="high")
class MatMulSmall(MatMul):
    pass

@backend(name="MatMul_Balanced", kernel="MatMul", language="hls",
         optimization="balanced", resource_usage="medium", default=True)
class MatMulBalanced(MatMul):
    pass

# Select backend based on requirements
def select_backend(kernel_name, requirements):
    if requirements.get("optimize_for") == "speed":
        return backends.get_for_kernel(kernel_name, optimization="throughput")
    elif requirements.get("optimize_for") == "area":
        return backends.get_for_kernel(kernel_name, optimization="area")
    else:
        return backends.get_for_kernel(kernel_name, optimization="balanced")
```

## Framework Integration

### Creating Framework-Specific Plugins

```python
# Register plugins for specific frameworks
@transform(name="CustomTransform", stage="cleanup", framework="myframework")
class CustomTransform:
    pass

@kernel(name="CustomKernel", framework="mycompany")
class CustomKernel:
    pass

@backend(name="CustomBackend", kernel="CustomKernel", 
         language="hls", framework="mycompany")
class CustomBackend(CustomKernel):
    pass

@step(name="CustomStep", category="analysis", framework="mytools")
class CustomStep:
    pass
```

### Accessing Framework-Specific Plugins

```python
# Access by framework
my_transform = transforms.myframework.CustomTransform
my_kernel = kernels.mycompany.CustomKernel
my_backend = backends.mycompany.CustomBackend
my_step = steps.mytools.CustomStep

# Dictionary access with framework qualification
my_transform = transforms["myframework:CustomTransform"]
my_kernel = kernels["mycompany:CustomKernel"]

# List plugins by framework
my_transforms = registry.get_framework_transforms("myframework")
my_kernels = registry.get_framework_kernels("mycompany")
```

## Blueprint Optimization

### Blueprint Definition

```yaml
# my_design.yaml
hw_compiler:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - GiveReadableTensorNames
    topology_opt:
      - OptimizeModel
    kernel_opt:
      - InferKernelShapes
  kernels:
    - MatMul
    - LayerNorm
  backends:
    - MatMulHLS
    - LayerNormHLS
  steps:
    - GenerateTestbench
    - SynthesizeDesign
```

### Loading Optimized Plugin Set

```python
from brainsmith.core.plugins.blueprint_loader import load_blueprint_plugins

# Load only plugins needed for this blueprint
collections = load_blueprint_plugins('my_design.yaml')

# Use the optimized collections (same API)
transforms = collections['transforms']
kernels = collections['kernels']
backends = collections['backends']
steps = collections['steps']

# Only blueprint-specified plugins are available
model = model.transform(transforms.OptimizeModel())
backend = backends.MatMulHLS()
result = steps.GenerateTestbench()(build_context)
```

## Debugging

### Plugin Registration Debugging

```python
from brainsmith.core.plugins import get_registry

registry = get_registry()

# Check if plugin is registered
if "MyTransform" in registry.transforms:
    print("MyTransform is registered")
else:
    print("MyTransform not found")

# List all plugins of a type
print("Registered transforms:")
for name in registry.list_available_transforms():
    metadata = registry.get_plugin_metadata(name)
    print(f"  {name}: stage={metadata.get('stage')}, framework={metadata.get('framework')}")

# Debug backend availability
kernel_name = "MatMul"
backends_list = registry.list_backends_by_kernel(kernel_name)
print(f"Backends for {kernel_name}: {backends_list}")

for backend_name in backends_list:
    metadata = registry.get_plugin_metadata(backend_name)
    print(f"  {backend_name}: language={metadata.get('language')}")
```

### Import and Discovery Issues

```python
# Ensure all plugin modules are imported
import myproject.transforms  # This will register transforms
import myproject.kernels     # This will register kernels
import myproject.backends    # This will register backends

# Check what got registered
stats = registry.get_stats()
print(f"After imports: {stats}")

# Debug missing plugins
try:
    plugin_cls = transforms.MyMissingTransform
except AttributeError as e:
    print(f"Plugin not found: {e}")
    print("Available transforms:", transforms.list())
```

## Best Practices

### Plugin Design

1. **Keep plugins focused**: Each plugin should have a single, clear responsibility
2. **Use descriptive names**: Plugin names should clearly indicate their purpose
3. **Provide good metadata**: Include framework, optimization strategy, resource usage
4. **Document interfaces**: Clear docstrings for apply(), generate_*() methods
5. **Handle errors gracefully**: Plugins should fail fast with clear error messages

### Framework Organization

```python
# Good: Organize by functionality
@transform(name="InferDataTypes", stage="topology_opt", framework="qonnx")
@transform(name="InferShapes", stage="topology_opt", framework="qonnx")

# Good: Clear backend differentiation
@backend(name="ConvHLS_Speed", kernel="Conv", language="hls", optimization="throughput")
@backend(name="ConvHLS_Area", kernel="Conv", language="hls", optimization="area")

# Avoid: Generic names
@backend(name="ConvImpl", kernel="Conv", language="hls")  # Not descriptive
```

### Performance Considerations

1. **Use registry queries efficiently**: Cache backend lookups if used repeatedly
2. **Leverage blueprint optimization**: Use subset registries for production
3. **Avoid dynamic plugin creation**: Register all plugins at import time
4. **Minimize validation overhead**: Plugin validation happens once at registration

### Testing

```python
import pytest
from brainsmith.core.plugins import get_registry, reset_plugin_system

def test_my_transform():
    # Reset for clean test environment
    reset_plugin_system()
    
    # Import your plugins
    from myproject.transforms import MyTransform
    
    # Test registration
    registry = get_registry()
    assert "MyTransform" in registry.transforms
    
    # Test functionality
    transform = MyTransform()
    result_model, changed = transform.apply(test_model)
    assert changed
    assert result_model is not None
```

## Migration Guide

### From Wrapper-Based Access

```python
# Old: Wrapper objects with convenience methods
kernel = kernels.MatMul
backend = kernel.hls()  # Convenience method on wrapper
instance = backend()

# New: Direct class access with explicit backend selection
kernel_cls = kernels.MatMul
backend_cls = backends.get_for_kernel("MatMul", language="hls")
kernel_instance = kernel_cls()
backend_instance = backend_cls()
```

### From Steps as Transforms

```python
# Old: Steps registered as transforms
@transform(name="MyStep", stage="build", plugin_type="step")
class MyStep:
    pass

# New: Steps have their own decorator and registry
@step(name="MyStep", category="build")
class MyStep:
    pass
```

### Framework Migration

```python
# Old: Only transforms had framework support
transforms.qonnx.BatchNormToAffine

# New: All plugin types support frameworks
transforms.qonnx.BatchNormToAffine
kernels.finn.MatrixVectorUnit
backends.finn.LayerNormHLS
steps.finn.CreateDataflowPartition
```

## Common Issues and Solutions

### Plugin Not Found

**Problem**: `AttributeError: Transform 'MyTransform' not found`

**Solutions**:
1. Ensure the module containing the plugin is imported
2. Check that the decorator is applied correctly
3. Verify the plugin name matches exactly

### Backend Not Available

**Problem**: No backend found for kernel

**Solutions**:
1. Check that backend is registered with correct kernel name
2. Verify the backend decorator specifies the right kernel
3. Use `backends.list_for_kernel(kernel_name)` to see available backends

### Framework Qualification Issues

**Problem**: `AttributeError: Transform 'MyTransform' not found in framework`

**Solutions**:
1. Verify the framework name in the decorator matches access pattern
2. Check that the plugin is registered with the expected framework
3. Use `registry.get_framework_transforms(framework)` to see available plugins

This developer guide provides comprehensive information for working with the Brainsmith plugin system. For additional examples and advanced usage patterns, refer to the examples in the repository.