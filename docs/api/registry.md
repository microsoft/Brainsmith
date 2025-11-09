# Component Registry

Register components using decorators for hardware kernels, backend implementations (HLS/RTL), and pipeline transformation steps. Components are discovered automatically from brainsmith, FINN, your project, and custom plugins.

---

::: brainsmith.registry.kernel

**Example:**

```python
from brainsmith.dataflow import KernelOp
from brainsmith.registry import kernel

@kernel
class ElementwiseBinaryOp(KernelOp):
    """Polymorphic kernel for Add, Mul, Sub, Div operations."""

    def get_nodeattr_types(self):
        return {"func": ("s", True, "")}
```

---

::: brainsmith.registry.backend

**Example:**

```python
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.registry import backend, get_kernel

# Get the kernel class
ElementwiseBinaryOp = get_kernel('ElementwiseBinaryOp')

@backend(target_kernel='ElementwiseBinaryOp', language='hls')
class ElementwiseBinary_hls(ElementwiseBinaryOp, HLSBackend):
    """HLS backend for ElementwiseBinaryOp."""

    def get_nodeattr_types(self):
        # Combine attributes from kernel and backend
        my_attrs = ElementwiseBinaryOp.get_nodeattr_types(self)
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs
```

---

::: brainsmith.registry.step

**Example:**

```python
from brainsmith.registry import step
from qonnx.transformation.infer_shapes import InferShapes

@step(name='qonnx_to_finn')
def qonnx_to_finn_step(model, cfg):
    """Convert QONNX to FINN opset."""
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    return model
```

---

::: brainsmith.registry.source_context

---

::: brainsmith.registry.discover_components

---

::: brainsmith.registry.reset_registry

---

::: brainsmith.registry.is_initialized

---

::: brainsmith.registry.get_kernel

**Example:**

```python
from brainsmith.registry import get_kernel

# Get kernel class by name
kernel_class = get_kernel('ElementwiseBinaryOp')
```

---

::: brainsmith.registry.get_kernel_infer

---

::: brainsmith.registry.has_kernel

---

::: brainsmith.registry.list_kernels

**Example:**

```python
from brainsmith.registry import list_kernels

# List all available kernels
kernels = list_kernels()
for name in kernels:
    print(name)
```

---

::: brainsmith.registry.get_backend

**Example:**

```python
from brainsmith.registry import list_backends_for_kernel, get_backend

# Find backends for a kernel
backend_names = list_backends_for_kernel('ElementwiseBinaryOp', language='hls')

# Get backend classes
backends = [get_backend(name) for name in backend_names]
```

---

::: brainsmith.registry.get_backend_metadata

---

::: brainsmith.registry.list_backends

---

::: brainsmith.registry.list_backends_for_kernel

---

::: brainsmith.registry.get_step

---

::: brainsmith.registry.has_step

---

::: brainsmith.registry.list_steps

---

::: brainsmith.registry.get_component_metadata

---

::: brainsmith.registry.get_all_component_metadata

---

::: brainsmith.registry.get_domain_for_backend

---

::: brainsmith.registry.ComponentMetadata

---

::: brainsmith.registry.ComponentType

---

::: brainsmith.registry.ImportSpec

## See Also

- [Getting Started](../getting-started.md) - Installation and quickstart guide
- [GitHub](https://github.com/microsoft/brainsmith) - Issues and questions
