# Writing Custom Kernels

Kernels are FPGA hardware components with multiple parts: the kernel class, an InferTransform for shape inference, and one or more backend implementations (HLS, RTL).

## Kernel Anatomy

A complete kernel consists of three components:

1. **Kernel Class** - Defines the operation and its properties
2. **InferTransform** - Performs shape/datatype inference during compilation
3. **Backend(s)** - Hardware implementations (HLS, RTL, etc.)

## Example: Creating a ReLU Kernel

### Step 1: Kernel Class

Create `plugins/my_relu/my_relu.py` in your project:

```python
"""Custom ReLU kernel implementation."""

from qonnx.custom_op.general.im2col import CustomOp

class MyReLU(CustomOp):
    """Custom ReLU activation kernel."""

    def get_nodeattr_types(self):
        """Define node attributes and their types."""
        return {
            "input_bitwidth": ("i", True, 8),      # (type, required, default)
            "output_bitwidth": ("i", True, 8),
        }

    def make_shape_compatible_op(self, model):
        """Return a shape-compatible standard ONNX op."""
        from qonnx.core.modelwrapper import ModelWrapper
        import onnx.helper as oh

        # ReLU is a standard ONNX op, use it for shape inference
        node = oh.make_node("Relu", [self.onnx_node.input[0]], [self.onnx_node.output[0]])
        return node

    def verify_node(self):
        """Verify the node configuration is valid."""
        input_bw = self.get_nodeattr("input_bitwidth")
        output_bw = self.get_nodeattr("output_bitwidth")

        assert input_bw > 0, "Input bitwidth must be positive"
        assert output_bw > 0, "Output bitwidth must be positive"
```

### Step 2: InferTransform

Create `plugins/my_relu/infer_my_relu.py`:

```python
"""Shape inference transform for MyReLU."""

from qonnx.transformation.base import Transformation
from qonnx.core.datatype import DataType

class InferMyReLU(Transformation):
    """Infer shapes and datatypes for MyReLU operations."""

    def apply(self, model):
        """Apply inference to all MyReLU nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for node in graph.node:
            node_ind += 1
            if node.op_type == "MyReLU":
                # Get input tensor info
                input_name = node.input[0]
                input_shape = model.get_tensor_shape(input_name)

                # ReLU preserves shape
                output_name = node.output[0]
                model.set_tensor_shape(output_name, input_shape)

                # Set output datatype based on configured bitwidth
                from brainsmith.kernels.my_relu.my_relu import MyReLU
                kernel_inst = MyReLU(node)
                output_bw = kernel_inst.get_nodeattr("output_bitwidth")

                output_dtype = DataType[f"INT{output_bw}"]
                model.set_tensor_datatype(output_name, output_dtype)

                graph_modified = True

        return (model, graph_modified)
```

### Step 3: HLS Backend

Create `plugins/my_relu/my_relu_hls.py`:

```python
"""HLS backend for MyReLU kernel."""

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

class MyReLU_hls(HWCustomOp):
    """HLS implementation of MyReLU kernel."""

    def get_nodeattr_types(self):
        """HLS-specific attributes."""
        attrs = super().get_nodeattr_types()
        attrs.update({
            "PE": ("i", True, 1),  # Parallelization factor
            "input_bitwidth": ("i", True, 8),
            "output_bitwidth": ("i", True, 8),
        })
        return attrs

    def generate_hdl(self, model, fpgapart, clk_ns):
        """Generate HLS code for this kernel."""
        code = []

        pe = self.get_nodeattr("PE")
        input_bw = self.get_nodeattr("input_bitwidth")
        output_bw = self.get_nodeattr("output_bitwidth")

        code.append(f"""
#include "ap_int.h"

template<unsigned int IN_BW, unsigned int OUT_BW, unsigned int PE>
void MyReLU_Batch(
    hls::stream<ap_uint<IN_BW * PE>> &in,
    hls::stream<ap_uint<OUT_BW * PE>> &out,
    unsigned int numReps
) {{
    for (unsigned int i = 0; i < numReps; i++) {{
        #pragma HLS PIPELINE II=1
        ap_uint<IN_BW * PE> input_val = in.read();
        ap_uint<OUT_BW * PE> output_val = 0;

        for (unsigned int pe = 0; pe < PE; pe++) {{
            #pragma HLS UNROLL
            ap_int<IN_BW> elem = input_val.range((pe+1)*IN_BW-1, pe*IN_BW);
            ap_int<OUT_BW> result = (elem > 0) ? elem : 0;
            output_val.range((pe+1)*OUT_BW-1, pe*OUT_BW) = result;
        }}

        out.write(output_val);
    }}
}}
""")

        # Generate top-level function
        code.append(f"""
void MyReLU_Top(
    hls::stream<ap_uint<{input_bw * pe}>> &in,
    hls::stream<ap_uint<{output_bw * pe}>> &out
) {{
    #pragma HLS INTERFACE axis port=in
    #pragma HLS INTERFACE axis port=out
    #pragma HLS INTERFACE ap_ctrl_none port=return

    MyReLU_Batch<{input_bw}, {output_bw}, {pe}>(in, out, NUM_REPS);
}}
""")

        return "\n".join(code)

    def get_exp_cycles(self):
        """Expected latency in cycles."""
        # ReLU is single-cycle with pipelining
        total_elems = self.get_total_elems()
        pe = self.get_nodeattr("PE")
        return total_elems // pe

    def get_total_elems(self):
        """Total number of elements to process."""
        input_shape = self.get_normal_input_shape()
        import numpy as np
        return np.prod(input_shape)
```

### Step 4: Register Your Kernel

Create `plugins/__init__.py` to register all your components:

```python
"""Project plugins registration."""

from brainsmith.registry import registry

# Import your kernel components
from .my_relu.my_relu import MyReLU
from .my_relu.infer_my_relu import InferMyReLU
from .my_relu.my_relu_hls import MyReLU_hls

# Register kernel with its inference transform
registry.kernel(
    MyReLU,
    name='MyReLU',
    op_type='MyReLU',
    infer_transform=InferMyReLU
)

# Register HLS backend
registry.backend(
    MyReLU_hls,
    name='MyReLU_HLS',
    target_kernel='project:MyReLU',  # Reference with source prefix
    language='hls'
)
```

## Using Your Custom Kernel

### In a Blueprint

```yaml
# blueprint.yaml
model: model_with_relu.onnx

steps:
  - qonnx_to_finn
  - streamline
  - specialize_layers  # Converts Relu → MyReLU

kernels:
  MyReLU:
    backend: hls
    config:
      PE: 4  # Process 4 elements in parallel
```

### Programmatically

```python
from brainsmith import get_kernel, get_kernel_infer, get_backend

# Get the kernel class
MyReLU = get_kernel('MyReLU')

# Apply inference transform
InferMyReLU = get_kernel_infer('MyReLU')
model = model.transform(InferMyReLU())

# Get HLS backend for hardware generation
MyReLU_HLS = get_backend('MyReLU', 'hls')
```

## Kernel Directory Structure

For complex kernels, organize with this structure:

```
plugins/
├── __init__.py              # Register all components here
└── my_relu/
    ├── __init__.py          # Package marker
    ├── my_relu.py           # Kernel class
    ├── infer_my_relu.py     # InferTransform
    ├── my_relu_hls.py       # HLS backend
    ├── my_relu_rtl.py       # RTL backend (optional)
    ├── templates/           # HLS/RTL templates
    │   ├── my_relu.cpp
    │   └── my_relu.v
    └── tests/
        └── test_my_relu.py
```

## Advanced: Multi-Backend Kernels

Support both HLS and RTL backends by registering multiple backends:

```python
# plugins/__init__.py
from brainsmith.registry import registry
from .my_relu.my_relu import MyReLU
from .my_relu.infer_my_relu import InferMyReLU
from .my_relu.my_relu_hls import MyReLU_hls
from .my_relu.my_relu_rtl import MyReLU_rtl

# Register kernel once
registry.kernel(MyReLU, name='MyReLU', infer_transform=InferMyReLU)

# Register both backends
registry.backend(MyReLU_hls, target_kernel='project:MyReLU', language='hls')
registry.backend(MyReLU_rtl, target_kernel='project:MyReLU', language='rtl')
```

Users can then choose their preferred backend in the blueprint:

```yaml
kernels:
  MyReLU:
    backend: rtl  # Use RTL instead of HLS
```

## Testing Custom Kernels

Test each component independently:

```python
# tests/test_my_relu.py
import pytest
from qonnx.core.modelwrapper import ModelWrapper
from brainsmith import get_kernel, get_kernel_infer

def test_kernel_basic():
    """Test kernel class instantiation."""
    MyReLU = get_kernel('MyReLU')
    assert MyReLU is not None

def test_infer_transform():
    """Test inference transform."""
    InferMyReLU = get_kernel_infer('MyReLU')

    # Load test model
    model = ModelWrapper("test_model.onnx")

    # Apply transform
    model_inferred = model.transform(InferMyReLU())

    # Verify shapes were inferred
    output_shape = model_inferred.get_tensor_shape("output")
    assert output_shape is not None

def test_hls_backend():
    """Test HLS backend code generation."""
    from brainsmith import get_backend

    MyReLU_HLS = get_backend('MyReLU', 'hls')

    # Create kernel instance
    # ... verify HDL generation
```

## Kernel Best Practices

1. **Single Responsibility**: Each kernel should implement one operation well
2. **Configurable**: Use node attributes for parameterization (PE, bitwidth, etc.)
3. **Documented**: Add docstrings explaining the operation and parameters
4. **Tested**: Write tests for kernel, inference, and each backend
5. **Performance**: Document expected cycles, resources, and throughput

## Example: Kernel with Configuration

More complex kernels often need extensive configuration:

```python
class AdvancedKernel(HWCustomOp):
    """Kernel with multiple configuration options."""

    def get_nodeattr_types(self):
        return {
            # Data configuration
            "input_bitwidth": ("i", True, 8),
            "weight_bitwidth": ("i", True, 8),
            "output_bitwidth": ("i", True, 16),

            # Parallelization
            "PE": ("i", True, 1),
            "SIMD": ("i", True, 1),

            # Memory interface
            "ram_style": ("s", False, "auto"),  # block, distributed, auto
            "mem_mode": ("s", False, "internal"),  # internal, external

            # Pipeline configuration
            "pipeline_depth": ("i", False, 2),

            # Resource constraints
            "use_dsp": ("i", False, 1),  # 0=no DSPs, 1=use DSPs
        }
```

## Next Steps

- [Plugin Quick Start](plugin-quickstart.md) - Basic plugin system usage
- [Writing Custom Steps](custom-steps.md) - Create pipeline steps
- Check `examples/kernel_integrator/` for a complete kernel example
