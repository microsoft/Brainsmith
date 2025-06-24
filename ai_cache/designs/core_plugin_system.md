# Core Plugin System Design

**Version**: 1.0  
**Date**: December 2024  
**Purpose**: Simple, practical plugin system for transforms, kernels, and backends

## Design Principles

1. **Minimal Required Attributes**: Only `name` required, everything else optional
2. **Simple Registration**: Decorators for easy registration
3. **Clear Dependencies**: `requires` for specifying dependencies
4. **Consistent Pattern**: Same approach for transforms, kernels, backends, and hw_transforms

## Transform Registration

Based on the `ExpandNorms` example, transforms follow QONNX patterns:

```python
from brainsmith.plugin import transform
from qonnx.transformation.base import Transformation

@transform(
    name="ExpandNorms",
    stage="topology_optimization",
    description="Expand LayerNorms/RMSNorms into functional components",
    author="thomas-keller",
    version="1.0.0",
    requires=["qonnx>=0.1.0", "numpy>=1.20"]
)
class ExpandNorms(Transformation):
    """Expand any standard LayerNorms/RMSNorms into functional components."""
    
    def __init__(self):
        super().__init__()
    
    def apply(self, model):
        # Transform implementation (as shown in example)
        graph_modified = False
        # ... transformation logic ...
        return (model, graph_modified)
```

## Kernel Registration (Stub)

Simple stub registration for kernels:

```python
from brainsmith.plugin import kernel

@kernel(
    name="MatMul",
    description="Matrix multiplication kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class MatMulKernel:
    """Stub for MatMul kernel - implementation by kernel team."""
    pass
```

## Backend Registration (Stub)

Backend registration for kernel implementations:

```python
from brainsmith.plugin import backend

@backend(
    name="MatMulHLS",
    description="HLS backend for MatMul kernel",
    author="brainsmith-team",
    version="1.0.0"
)
class MatMulHLSBackend:
    """Stub for MatMul HLS backend - implementation by kernel team."""
    pass
```

## Hardware Transform Registration (Stub)

Hardware-specific transforms:

```python
from brainsmith.plugin import hw_transform

@hw_transform(
    name="OptimizeDSPUsage",
    description="Optimize DSP usage in hardware kernels",
    author="fpga-expert",
    version="1.0.0"
)
class OptimizeDSPUsageTransform:
    """Stub for DSP optimization transform."""
    pass
```

## Core Plugin Infrastructure

### 1. Plugin Decorators

```python
# brainsmith/plugin/decorators.py

def transform(name, stage, description=None, author=None, version=None, requires=None):
    """Decorator for registering transforms."""
    def decorator(cls):
        # Validate that it's a QONNX Transformation
        if not issubclass(cls, Transformation):
            raise TypeError(f"Transform {name} must inherit from qonnx.transformation.base.Transformation")
        
        # Add metadata
        cls._plugin_metadata = {
            "type": "transform",
            "name": name,
            "stage": stage,
            "description": description,
            "author": author,
            "version": version,
            "requires": requires or []
        }
        
        # Register with global registry
        PluginRegistry.register(cls)
        return cls
    return decorator

def kernel(name, description=None, author=None, version=None):
    """Decorator for registering kernels."""
    def decorator(cls):
        cls._plugin_metadata = {
            "type": "kernel",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        PluginRegistry.register(cls)
        return cls
    return decorator

def backend(name, description=None, author=None, version=None):
    """Decorator for registering backends."""
    def decorator(cls):
        cls._plugin_metadata = {
            "type": "backend",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        PluginRegistry.register(cls)
        return cls
    return decorator

def hw_transform(name, description=None, author=None, version=None):
    """Decorator for registering hardware transforms."""
    def decorator(cls):
        cls._plugin_metadata = {
            "type": "hw_transform",
            "name": name,
            "description": description,
            "author": author,
            "version": version
        }
        PluginRegistry.register(cls)
        return cls
    return decorator
```

### 2. Plugin Registry

```python
# brainsmith/plugin/registry.py

class PluginRegistry:
    """Central registry for all plugins."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {
                "transform": {},
                "kernel": {},
                "backend": {},
                "hw_transform": {}
            }
        return cls._instance
    
    @classmethod
    def register(cls, plugin_class):
        """Register a plugin class."""
        instance = cls()
        metadata = plugin_class._plugin_metadata
        plugin_type = metadata["type"]
        name = metadata["name"]
        
        # Check for duplicates
        if name in instance._plugins[plugin_type]:
            raise ValueError(f"{plugin_type} '{name}' already registered")
        
        # Validate dependencies
        instance._validate_requires(metadata.get("requires", []))
        
        # Register
        instance._plugins[plugin_type][name] = plugin_class
        logger.info(f"Registered {plugin_type}: {name}")
    
    def get_transform(self, name):
        """Get a transform by name."""
        return self._plugins["transform"].get(name)
    
    def get_kernel(self, name):
        """Get a kernel by name."""
        return self._plugins["kernel"].get(name)
    
    def get_backend(self, name):
        """Get a backend by name."""
        return self._plugins["backend"].get(name)
    
    def get_hw_transform(self, name):
        """Get a hardware transform by name."""
        return self._plugins["hw_transform"].get(name)
    
    def list_transforms(self, stage=None):
        """List all transforms, optionally filtered by stage."""
        transforms = []
        for name, cls in self._plugins["transform"].items():
            metadata = cls._plugin_metadata
            if stage is None or metadata.get("stage") == stage:
                transforms.append((name, cls))
        return transforms
    
    def _validate_requires(self, requires):
        """Validate that requirements can be satisfied."""
        for req in requires:
            # Simple validation - could be enhanced
            if "kernel:" in req:
                kernel_name = req.split("kernel:")[1]
                if kernel_name not in self._plugins["kernel"]:
                    logger.warning(f"Required kernel '{kernel_name}' not found")
```

### 3. Plugin Discovery

```python
# brainsmith/plugin/discovery.py

import importlib
import pkgutil
from pathlib import Path

class PluginDiscovery:
    """Discover and load plugins from various sources."""
    
    @staticmethod
    def discover_plugins():
        """Discover all available plugins."""
        # Discover from built-in locations
        PluginDiscovery._discover_builtin_plugins()
        
        # Discover from plugin directories
        plugin_dirs = [
            Path.home() / ".brainsmith" / "plugins",
            Path("./brainsmith_plugins"),
        ]
        
        for plugin_dir in plugin_dirs:
            if plugin_dir.exists():
                PluginDiscovery._discover_directory(plugin_dir)
    
    @staticmethod
    def _discover_builtin_plugins():
        """Load built-in transforms from brainsmith.libraries.transforms."""
        try:
            import brainsmith.libraries.transforms.operations
            package = brainsmith.libraries.transforms.operations
            
            for importer, modname, ispkg in pkgutil.iter_modules(
                package.__path__, package.__name__ + "."
            ):
                try:
                    importlib.import_module(modname)
                    logger.debug(f"Loaded built-in module: {modname}")
                except Exception as e:
                    logger.warning(f"Failed to load {modname}: {e}")
        except ImportError:
            logger.warning("Could not import built-in transforms")
    
    @staticmethod
    def _discover_directory(directory):
        """Discover plugins in a directory."""
        for path in directory.rglob("*.py"):
            if path.name.startswith("_"):
                continue
            
            try:
                spec = importlib.util.spec_from_file_location(
                    path.stem, path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                logger.debug(f"Loaded plugin from: {path}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {path}: {e}")
```

### 4. Integration with Compilation System

```python
# brainsmith/core/finn_v2/transform_executor.py

class TransformExecutor:
    """Execute transforms with plugin support."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        # Discover plugins on initialization
        PluginDiscovery.discover_plugins()
    
    def execute_stage(self, stage, model, config=None):
        """Execute all transforms for a given stage."""
        transforms = self.registry.list_transforms(stage=stage)
        
        for name, transform_class in transforms:
            logger.info(f"Executing transform: {name}")
            try:
                transform = transform_class()
                model, modified = transform.apply(model)
                if modified:
                    logger.info(f"Transform {name} modified the model")
            except Exception as e:
                logger.error(f"Transform {name} failed: {e}")
                raise
        
        return model
```

## Usage Examples

### 1. Creating a Simple Transform

```python
from brainsmith.plugin import transform
from qonnx.transformation.base import Transformation

@transform(
    name="RemoveIdentityOps",
    stage="graph_cleanup",
    description="Remove identity operations from graph",
    author="optimization-expert",
    version="1.0.0"
)
class RemoveIdentityOps(Transformation):
    def apply(self, model):
        graph = model.graph
        graph_modified = False
        
        for node in list(graph.node):
            if node.op_type == "Identity":
                # Bypass identity node
                self._bypass_node(model, node)
                graph.node.remove(node)
                graph_modified = True
        
        return (model, graph_modified)
```

### 2. Using Transforms in Compilation

```python
from brainsmith.core.finn_v2 import TransformExecutor

# Executor automatically discovers all plugins
executor = TransformExecutor()

# Execute all topology optimization transforms
model = executor.execute_stage("topology_optimization", model)

# Or use specific transform
registry = PluginRegistry()
ExpandNormsClass = registry.get_transform("ExpandNorms")
if ExpandNormsClass:
    transform = ExpandNormsClass()
    model, modified = transform.apply(model)
```

### 3. Listing Available Components

```python
registry = PluginRegistry()

# List all transforms for a stage
topology_transforms = registry.list_transforms(stage="topology_optimization")
print(f"Available topology transforms: {[name for name, _ in topology_transforms]}")

# Check if specific kernel exists
if registry.get_kernel("MatMul"):
    print("MatMul kernel is available")
```

## Benefits of This Approach

1. **Simple**: Minimal boilerplate, just decorate your class
2. **Discoverable**: Automatic discovery from standard locations
3. **Compatible**: Works with existing QONNX transformation pattern
4. **Extensible**: Easy to add new plugin types
5. **Dependency Aware**: Can specify requirements
6. **No Magic**: Clear, explicit registration

## Next Steps

1. Implement the core decorators and registry
2. Add plugin discovery from standard directories
3. Integrate with the compilation pipeline
4. Create documentation for plugin developers
5. Add validation and error handling

This provides a solid foundation that can be extended with more advanced features (hub, certification, etc.) later while keeping the core simple and functional.