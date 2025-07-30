# Brainsmith Plugin System

Unified registry for transforms, kernels, backends, and build steps from FINN, QONNX, and Brainsmith.

## Plugin Types

- **Transforms**: Graph transformations for optimization, quantization, and hardware mapping
- **Kernels**: Hardware implementation units (HWCustomOp subclasses)
- **Backends**: Code generators that produce HLS or RTL implementations
- **Steps**: Build pipeline stages that orchestrate transforms

## Usage

```python
from brainsmith.core.plugins import (
    get_transform, get_kernel, get_backend, get_step,
    list_transforms, list_kernels, list_backends, list_steps
)

# Get plugins by name (framework prefix optional for unique names)
transform = get_transform('Streamline')                 # Finds finn:Streamline
kernel = get_kernel('MVAU')                            # Finds finn:MVAU
step = get_step('tidy_up')                             # Finds finn:tidy_up

# Use framework prefix to disambiguate or be explicit
transform = get_transform('qonnx:RemoveIdentityOps')    # QONNX version
transform = get_transform('RemoveIdentityOps')          # Brainsmith version (default)

# List available plugins
transforms = list_transforms()  # ['RemoveIdentityOps', 'qonnx:InferShapes', 'finn:Streamline', ...]
kernels = list_kernels()       # ['LayerNorm', 'finn:MVAU', 'finn:Thresholding', ...]
backends = list_backends()     # ['LayerNormHLS', 'finn:MVAU_hls', 'finn:Thresholding_rtl', ...]
steps = list_steps()           # ['cleanup', 'finn:tidy_up', 'finn:streamline', ...]
```

## Creating Plugins

```python
from brainsmith.core.plugins import transform, kernel, backend, step

@transform  # name defaults to class name "MyTransform"
class MyTransform(Transformation):
    def apply(self, model):
        # Transform implementation

@transform(name="CustomName")  # Explicit name override
class MyTransformImpl(Transformation):
    def apply(self, model):
        # Registered as "CustomName", not "MyTransformImpl"

@step(name="my_step", category="optimization")
def my_step(model, cfg):
    # Step implementation
```

## Framework Namespacing

Plugins are automatically namespaced by framework, but the prefix is optional when names are unique:
- `InferShapes` or `qonnx:InferShapes` - QONNX transform (unique name)
- `MVAU` or `finn:MVAU` - FINN kernel (unique name)
- `RemoveIdentityOps` - Brainsmith version (exists in multiple frameworks)
- `qonnx:RemoveIdentityOps` - QONNX version (explicit prefix required)

## Advanced Usage

```python
# Query by metadata
qonnx_transforms = get_transforms_by_metadata(framework='qonnx')
rtl_backends = get_backends_by_metadata(language='rtl')

# Get backends for a specific kernel
backends = list_backends_by_kernel('MVAU')  # Returns HLS and RTL variants

# Check plugin existence
if has_transform('ConvertDivToMul'):
    transform = get_transform('ConvertDivToMul')
```

## Adding New Framework Plugins

To register plugins from external frameworks:

1. Add plugin definitions to `framework_adapters.py`
2. Follow the existing pattern for transforms, kernels, backends, or steps
3. Test that plugins load correctly

Debug with: `PYTHONLOGGING=DEBUG python my_script.py`