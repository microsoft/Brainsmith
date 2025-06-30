"""
External Plugin Support

Enables pip-installable plugins to integrate with the blueprint-driven
plugin system through entry points.
"""

import logging
from typing import List, Dict, Any
from setuptools import setup

logger = logging.getLogger(__name__)


def create_external_plugin_template():
    """
    Create a template showing how external developers can create
    pip-installable plugins for BrainSmith.
    
    This template would be provided in documentation.
    """
    
    # Example plugin implementation
    plugin_py_template = '''
"""
Example External BrainSmith Plugin
"""

from brainsmith.plugin.core import transform, kernel, backend
from qonnx.transformation.base import Transformation


@transform(
    name="MyCustomTransform",
    stage="topology_opt", 
    description="Custom optimization transform",
    author="External Developer",
    version="1.0.0"
)
class MyCustomTransform(Transformation):
    def apply(self, model):
        # Custom transformation logic
        print("Applying custom transform!")
        return model, False


@kernel(
    name="MyCustomKernel",
    op_type="MyCustomOp",
    description="Custom hardware kernel"
)
class MyCustomKernel:
    pass


@backend(
    name="MyCustomKernelHLS",
    kernel="MyCustomKernel",
    backend_type="hls",
    description="HLS implementation of custom kernel"
)
class MyCustomKernelHLS:
    pass
'''
    
    # Example setup.py for pip installation
    setup_py_template = '''
from setuptools import setup, find_packages

setup(
    name="my-brainsmith-plugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "brainsmith",  # Core BrainSmith dependency
    ],
    entry_points={
        # Register plugins with BrainSmith's discovery system
        "brainsmith.plugins": [
            "MyCustomTransform = my_plugin.transforms:MyCustomTransform",
            "MyCustomKernel = my_plugin.kernels:MyCustomKernel", 
            "MyCustomKernelHLS = my_plugin.kernels:MyCustomKernelHLS",
        ],
    },
    author="External Developer",
    description="Custom BrainSmith plugins",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
'''
    
    return {
        "plugin.py": plugin_py_template,
        "setup.py": setup_py_template
    }


class ExternalPluginValidator:
    """
    Validates external plugins for security and compatibility.
    """
    
    def validate_external_plugin(self, plugin_class: type) -> Dict[str, Any]:
        """
        Validate an external plugin for safety and compatibility.
        
        Args:
            plugin_class: The plugin class to validate
            
        Returns:
            Validation result with warnings/errors
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required metadata
        if not hasattr(plugin_class, '_plugin_metadata'):
            result["warnings"].append("Plugin missing metadata - may not be properly registered")
        
        # Check for required methods based on plugin type
        metadata = getattr(plugin_class, '_plugin_metadata', {})
        plugin_type = metadata.get('type', 'unknown')
        
        if plugin_type == 'transform':
            if not hasattr(plugin_class, 'apply'):
                result["errors"].append("Transform plugin must implement 'apply' method")
                result["valid"] = False
        
        # Security checks - look for potentially dangerous operations
        dangerous_imports = ['os', 'subprocess', 'sys', 'shutil']
        try:
            import inspect
            source = inspect.getsource(plugin_class)
            for dangerous in dangerous_imports:
                if f'import {dangerous}' in source:
                    result["warnings"].append(f"Plugin imports potentially dangerous module: {dangerous}")
        except Exception:
            result["warnings"].append("Could not inspect plugin source code")
        
        # Check for proper base class inheritance
        base_classes = [cls.__name__ for cls in plugin_class.__mro__]
        if plugin_type == 'transform' and 'Transformation' not in base_classes:
            result["warnings"].append("Transform should inherit from qonnx.transformation.base.Transformation")
        
        return result


def get_external_plugin_documentation() -> str:
    """
    Generate documentation for external plugin developers.
    
    Returns:
        Markdown documentation explaining how to create external plugins
    """
    
    docs = """
# Creating External BrainSmith Plugins

External developers can create pip-installable plugins that integrate seamlessly with BrainSmith's blueprint-driven plugin system.

## Plugin Types

BrainSmith supports four types of plugins:

1. **Transforms**: Graph transformation operations
2. **Kernels**: Hardware operation definitions  
3. **Backends**: HLS/RTL implementations of kernels
4. **Kernel Inference**: Pattern detection transforms

## Creating a Plugin

### 1. Plugin Implementation

```python
from brainsmith.plugin.core import transform
from qonnx.transformation.base import Transformation

@transform(
    name="MyOptimization",
    stage="topology_opt",
    description="Custom optimization for my use case",
    author="Your Name",
    version="1.0.0"
)
class MyOptimization(Transformation):
    def apply(self, model):
        # Your transformation logic here
        modified = False
        # ... modify model ...
        return model, modified
```

### 2. Package Setup

Create a `setup.py` file to make your plugin pip-installable:

```python
from setuptools import setup, find_packages

setup(
    name="my-brainsmith-plugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["brainsmith"],
    entry_points={
        "brainsmith.plugins": [
            "MyOptimization = my_plugin:MyOptimization",
        ],
    },
)
```

### 3. Installation and Use

Users install your plugin with pip:

```bash
pip install my-brainsmith-plugin
```

Then use it in blueprints:

```yaml
hw_compiler:
  transforms:
    - "MyOptimization"  # Your plugin is automatically discovered
```

## Blueprint Integration

The blueprint-driven plugin system will automatically:

1. **Discover** your plugin via entry points
2. **Load** it only when specified in a blueprint  
3. **Execute** it at the appropriate compilation stage

This means your plugin is only loaded when actually needed, keeping the system efficient.

## Best Practices

1. **Follow naming conventions**: Clear, descriptive names
2. **Proper error handling**: Don't crash the entire pipeline
3. **Documentation**: Include docstrings and descriptions
4. **Testing**: Test with real models and blueprints
5. **Security**: Avoid dangerous operations like file system access

## Framework Attribution

Your plugins are automatically attributed to your framework name, preventing naming conflicts with built-in plugins.
"""
    
    return docs


# Global validator instance
_validator = None


def get_external_plugin_validator() -> ExternalPluginValidator:
    """Get the external plugin validator instance."""
    global _validator
    if _validator is None:
        _validator = ExternalPluginValidator()
    return _validator