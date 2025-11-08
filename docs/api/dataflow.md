# Dataflow Modeling

Core abstractions for modeling hardware kernels using schema-based design spaces.

Two-phase construction separates expensive setup from fast configuration: Design Space is built once and defines valid parameter ranges, while Design Point is configured many times to represent specific hardware instances. This enables efficient exploration by avoiding redundant computation.

---

::: brainsmith.dataflow.KernelOp

**Example:**

```python
import brainsmith.dataflow as df
from brainsmith.registry import kernel
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

@kernel(description="Hardware LayerNorm", author="Your Name")
class LayerNorm(df.KernelOp):
    """Hardware LayerNorm kernel."""

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper) -> df.KernelSchema:
        """Define kernel structure."""
        return LAYERNORM_SCHEMA

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if node can be converted to this kernel."""
        return node.op_type == "FuncLayerNorm"

    @classmethod
    def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int):
        """Transform ONNX node to hardware kernel."""
        hw_node = helper.make_node(
            "LayerNorm",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
        )
        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node]
        )
```

---

::: brainsmith.dataflow.KernelOpError

---

::: brainsmith.dataflow.KernelSchema

**Example:**

```python
import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM

# Define kernel schema
LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],
            stream_tiling=["SIMD"],
            required_layout="NHWC",
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],
            stream_tiling=[df.derive_dim("input", df.ShapeHierarchy.STREAM, -1)],
            required_layout="NHWC",
        )
    ],
    kernel_params={
        "epsilon": ("f", True, 1e-5),
    },
    constraints=[
        df.AttrCompare("epsilon", ">", 0),
    ],
)
```

---

::: brainsmith.dataflow.InputSchema

---

::: brainsmith.dataflow.OutputSchema

---

::: brainsmith.dataflow.ParameterSpec

**Example:**

```python
import brainsmith.dataflow as df

# Ordered parameter (list/tuple enables navigation)
depth_param = df.ParameterSpec("depth", [128, 256, 512], default=256)

# Discrete parameter (set for unordered categories)
ram_param = df.ParameterSpec("ram_style", {"distributed", "block"}, default="distributed")

# Use in kernel schema
schema = df.KernelSchema(
    name="MyKernel",
    inputs=[...],
    outputs=[...],
    dse_parameters={
        "ram_style": ram_param,
        "depth": depth_param,
    }
)
```

---

::: brainsmith.dataflow.KernelDesignSpace

---

::: brainsmith.dataflow.InterfaceDesignSpace

---

::: brainsmith.dataflow.KernelDesignPoint

**Example:**

```python
# Get design point from kernel operator
op._ensure_ready(model)
point = op.design_point

# Configure using interface-based API (for stream parameters)
point = point.with_input_stream(0, 32)   # Set input PE=32
point = point.with_output_stream(0, 16)  # Set output PE=16

# Configure using dimension-based API (for generic DSE)
point = point.with_dimension("ram_style", "distributed")
point = point.with_dimension("depth", 256)

# Apply configuration
op.apply_design_point(point)
```

---

::: brainsmith.dataflow.InterfaceDesignPoint

---

::: brainsmith.dataflow.OrderedParameter

---

::: brainsmith.dataflow.DesignSpaceBuilder

---

::: brainsmith.dataflow.BuildContext

---

::: brainsmith.dataflow.Constraint

---

::: brainsmith.dataflow.ValidationError

---

::: brainsmith.dataflow.DesignSpaceValidationContext

---

::: brainsmith.dataflow.ConfigurationValidationContext

---

::: brainsmith.dataflow.TransformationResult

**Example:**

```python
import brainsmith.dataflow as df
from onnx import helper

# Create transformation result when converting ONNX to HW node
hw_node = helper.make_node(
    "LayerNorm",
    inputs=list(node.input),
    outputs=list(node.output),
    domain="brainsmith.kernels",
    name=f"LayerNorm_{node.name}",
)

result = df.TransformationResult(
    nodes_to_insert=[hw_node],
    nodes_to_remove=[node]
)
```

---

::: brainsmith.dataflow.Shape

---

::: brainsmith.dataflow.ShapeHierarchy

---

::: brainsmith.dataflow.FULL_DIM

---

::: brainsmith.dataflow.FULL_SHAPE

## See Also

- [Component Registry](registry.md) - Register custom kernels, backends, and steps
- [Getting Started](../getting-started.md) - Installation and quickstart
- [GitHub](https://github.com/microsoft/brainsmith) - Issues and questions
