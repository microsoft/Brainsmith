# Writing Custom Steps

Steps are Python functions that transform ONNX models. Adding a custom step is as simple as dropping a Python file in the right location.

## Step Function Signature

Every step must follow this signature:

```python
def my_step(model, cfg):
    """
    Transform a model.

    Args:
        model: QONNX ModelWrapper instance
        cfg: DSEConfig with blueprint settings

    Returns:
        Transformed ModelWrapper
    """
    # Your transformation logic
    return model
```

## Example 1: Simple Cleanup Step

Create `plugins/my_steps.py` in your project:

```python
"""Custom step to remove debug nodes from model."""

def remove_debug_nodes_step(model, cfg):
    """Remove nodes with 'debug' in their name."""
    from qonnx.core.modelwrapper import ModelWrapper

    graph = model.graph
    nodes_to_remove = [
        node for node in graph.node
        if 'debug' in node.name.lower()
    ]

    for node in nodes_to_remove:
        graph.node.remove(node)

    return model
```

Now register it in `plugins/__init__.py`:

```python
from brainsmith.registry import registry
from .my_steps import remove_debug_nodes_step

# Register with the plugin system
registry.step(remove_debug_nodes_step, name='remove_debug_nodes')
```

**That's it!** The step is now available:

```python
from brainsmith import get_step

cleanup = get_step('project:remove_debug_nodes')  # or just 'remove_debug_nodes' if project is default
model = cleanup(model, cfg)
```

## Example 2: Using Transforms

Steps often apply multiple transforms in sequence:

```python
"""Step that applies common QONNX optimizations."""

def qonnx_optimize_step(model, cfg):
    """Apply standard QONNX optimization transforms."""
    from brainsmith import apply_transforms

    transforms = [
        'FoldConstants',
        'InferShapes',
        'RemoveIdentityOps',
        'SimplifyGraph',
    ]

    return apply_transforms(model, transforms)
```

## Example 3: Parameterized Step

Access blueprint configuration for parameterized behavior:

```python
"""Step that quantizes activations to specified bitwidth."""

def quantize_activations_step(model, cfg):
    """Quantize activations using configured bitwidth."""
    from qonnx.transformation.quantize import QuantizeActivations

    # Read parameter from blueprint config
    bitwidth = getattr(cfg, 'activation_bitwidth', 8)

    # Use parameterized transform
    transform = QuantizeActivations(bitwidth=bitwidth)

    return model.transform(transform)
```

In your blueprint YAML:

```yaml
config:
  activation_bitwidth: 4

steps:
  - quantize_activations
```

## Example 4: Conditional Logic

Steps can have conditional behavior based on configuration:

```python
"""Step that selectively applies optimizations."""

def conditional_optimize_step(model, cfg):
    """Apply optimizations based on config flags."""
    from brainsmith import apply_transforms

    transforms = ['FoldConstants', 'InferShapes']

    # Add optional transforms based on config
    if getattr(cfg, 'aggressive_optimization', False):
        transforms.extend([
            'RemoveIdentityOps',
            'SimplifyGraph',
            'FoldConstants',  # Run again after simplification
        ])

    if getattr(cfg, 'enable_dataflow', False):
        transforms.append('ConvertToDataflow')

    return apply_transforms(model, transforms)
```

## Example 5: Integration with External Tools

Steps can call external tools or libraries:

```python
"""Step that validates model with external tool."""

def validate_with_external_tool_step(model, cfg):
    """Run external validation and modify model."""
    import subprocess
    from pathlib import Path

    # Save model temporarily
    temp_path = Path(cfg.output_dir) / "temp_model.onnx"
    model.save(str(temp_path))

    # Run external validator
    result = subprocess.run(
        ['onnx-validator', str(temp_path)],
        capture_output=True
    )

    if result.returncode != 0:
        print(f"Validation warnings: {result.stderr.decode()}")

    # Model is unchanged in this example
    return model
```

## File Naming Convention

The plugin system finds your step by converting the file name to the step name:

- File: `remove_debug_nodes.py` → Function: `remove_debug_nodes_step`
- File: `my_custom_step.py` → Function: `my_custom_step_step`

**Important**: The function must be named `{filename}_step`.

## Using Custom Steps in Blueprints

Once defined, custom steps work exactly like built-in steps:

```yaml
# blueprint.yaml
model: model.onnx

steps:
  - qonnx_to_finn
  - remove_debug_nodes      # Your custom step
  - streamline
  - qonnx_optimize          # Another custom step
  - specialize_layers
```

## Testing Custom Steps

Test your steps like any Python function:

```python
# test_custom_steps.py
from unittest.mock import MagicMock
from brainsmith import get_step

def test_remove_debug_nodes():
    """Test debug node removal."""
    step = get_step('remove_debug_nodes')

    # Mock model with debug node
    mock_model = MagicMock()
    mock_cfg = MagicMock()

    result = step(mock_model, mock_cfg)
    assert result is mock_model
```

## Advanced: Multi-file Steps

For complex steps, organize in a package:

```
plugins/
├── __init__.py              # Registers all components
└── advanced_optimization/
    ├── __init__.py          # Exports the step function
    ├── analysis.py
    └── transforms.py
```

In `plugins/advanced_optimization/__init__.py`:

```python
"""Advanced optimization step package."""

from .analysis import analyze_model
from .transforms import apply_optimizations

def advanced_optimization_step(model, cfg):
    """Multi-stage optimization pipeline."""
    analysis_results = analyze_model(model)
    return apply_optimizations(model, analysis_results, cfg)
```

In `plugins/__init__.py`:

```python
from brainsmith.registry import registry
from .advanced_optimization import advanced_optimization_step

registry.step(advanced_optimization_step, name='advanced_optimization')
```

## Error Handling

Add helpful error messages for debugging:

```python
def robust_step(model, cfg):
    """Step with error handling."""
    try:
        # Your logic here
        return model
    except Exception as e:
        print(f"Error in robust_step: {e}")
        print(f"Model: {model.model.graph.input[0].name}")
        print(f"Config: {cfg}")
        raise
```

## Next Steps

- [Plugin Quick Start](plugin-quickstart.md) - Basic plugin system usage
- [Writing Custom Kernels](custom-kernels.md) - Create FPGA hardware components
