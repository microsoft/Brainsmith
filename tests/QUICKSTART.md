# Test Framework Quickstart

Get from zero to running tests in 5 minutes.

## Prerequisites

```bash
# From brainsmith root
source .venv/bin/activate && source .brainsmith/env.sh
```

---

## Option 1: Test a Single Implementation

**Use `KernelTest` when:** You want to validate one kernel implementation against a golden reference.

### Minimal Example

```python
"""Test your kernel implementation."""
import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.kernel_test import KernelTest
from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
)


class TestMyKernel(KernelTest):
    """Test MyKernel implementation."""

    @pytest.fixture(
        params=[
            KernelTestConfig(
                test_id="mykernel_baseline",
                model=ModelStructure(
                    operation="MyOperation",
                    input_shapes={"input": (1, 64)},
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Test configuration."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create ONNX model with your operation."""
        shape = list(kernel_test_config.input_shapes["input"])

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        node = helper.make_node("MyOperation", ["input"], ["output"], name="MyOp_0")
        graph = helper.make_graph([node], "test_mykernel", [inp], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["input"]

    def get_kernel_op(self):
        """Return kernel operator class."""
        from brainsmith.kernels.my_kernel import MyKernelOp
        return MyKernelOp

    def get_num_inputs(self):
        """Number of inputs."""
        return 1

    def get_num_outputs(self):
        """Number of outputs."""
        return 1
```

### Run It

```bash
# Python execution only (fast)
pytest test_my_kernel.py -m "not slow" -v

# All backends (cppsim + rtlsim, requires Vivado)
pytest test_my_kernel.py -v
```

**Result:** 6 tests automatically
- Stage 1 model structure validation
- Stage 2 kernel inference validation
- Python execution vs golden
- Stage 3 backend specialization
- C++ simulation vs golden
- RTL simulation vs golden

---

## Option 2: Compare Two Implementations (Parity Testing)

**Use `KernelParityTest` when:** You want to compare two implementations (e.g., FINN vs Brainsmith) for parity.

### Minimal Example

```python
"""Test parity between Reference (FINN) and Primary (Brainsmith) implementations."""
import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.test_config import (
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
)


class TestMyKernelParity(KernelParityTest):
    """Test parity between reference and primary implementations."""

    @pytest.fixture(
        params=[
            KernelTestConfig(
                test_id="mykernel_baseline",
                model=ModelStructure(
                    operation="MyOperation",
                    input_shapes={"input": (1, 64)},
                    input_dtypes={"input": DataType["INT8"]},
                ),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Test configuration."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create shared ONNX model."""
        shape = list(kernel_test_config.input_shapes["input"])

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

        node = helper.make_node("MyOperation", ["input"], ["output"], name="MyOp_0")
        graph = helper.make_graph([node], "test_parity", [inp], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["input"]

    # ========================================================================
    # Reference Implementation (usually FINN)
    # ========================================================================

    def infer_kernel_reference(self, model, target_node):
        """Infer reference kernel (FINN implementation)."""
        from finn.transformation.fpgadataflow.convert_to_hw_layers import InferMyOperation

        model = model.transform(InferMyOperation())

        # FINN often doesn't preserve node names, find by op_type
        nodes = model.get_nodes_by_op_type("MyFinnKernel")
        assert len(nodes) == 1, f"Expected 1 node, found {len(nodes)}"

        from qonnx.custom_op.registry import getCustomOp
        op = getCustomOp(nodes[0])

        return op, model

    def get_backend_variants_reference(self):
        """Return reference backend variants."""
        from finn.custom_op.fpgadataflow.hls.my_kernel_hls import MyKernel_hls
        return [MyKernel_hls]

    # ========================================================================
    # Primary Implementation (Brainsmith - uses inherited defaults)
    # ========================================================================

    def get_kernel_op(self):
        """Return primary kernel operator class."""
        from brainsmith.kernels.my_kernel import MyKernelOp
        return MyKernelOp

    # Primary uses inherited defaults:
    # - infer_kernel() - creates InferKernels([get_kernel_op()]) automatically
    # - get_backend_variants() - auto-detects from registry
    # - configure_kernel() - auto-configures from fixture

    # ========================================================================
    # Test Structure
    # ========================================================================

    def get_num_inputs(self):
        """Number of inputs."""
        return 1

    def get_num_outputs(self):
        """Number of outputs."""
        return 1

    # ========================================================================
    # Golden Reference (for golden validation tests)
    # ========================================================================

    def compute_golden_reference(self, inputs):
        """Compute expected output using NumPy."""
        return {"output": np.my_operation(inputs["input"])}
```

### Run It

```bash
# Python execution only (fast)
pytest test_my_kernel_parity.py -m "not slow" -v

# All backends
pytest test_my_kernel_parity.py -v
```

**Result:** 18 tests automatically
- 6 golden execution tests (primary/reference × python/cppsim/rtlsim)
- 7 core parity tests (outputs, shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources, efficiency)

---

## Quick Tips

### Common First-Time Issues

**Issue:** `AttributeError: 'TestMyKernel' object has no attribute 'kernel_test_config'`

**Solution:** Ensure your fixture is named exactly `kernel_test_config`:
```python
@pytest.fixture(params=[...])
def kernel_test_config(self, request):  # Must be this exact name
    return request.param
```

---

**Issue:** FINN node not found after transform

**Solution:** FINN doesn't preserve node names. Search by op_type instead:
```python
def infer_kernel_reference(self, model, target_node):
    model = model.transform(InferMyOperation())

    # Don't search by target_node name - search by op_type
    nodes = model.get_nodes_by_op_type("MyFinnKernel")
    assert len(nodes) == 1

    from qonnx.custom_op.registry import getCustomOp
    return getCustomOp(nodes[0]), model
```

---

**Issue:** Backend tests (cppsim/rtlsim) are skipped

**Solution:** Provide an FPGA part in your configuration:
```python
KernelTestConfig(
    test_id="test",
    model=ModelStructure(...),
    platform=PlatformConfig(fpgapart="xc7z020clg400-1"),  # Add this
)
```

---

### Test Execution Controls

```bash
# Fast tests only (skip cppsim/rtlsim)
pytest test_file.py -m "not slow" -v

# Skip backend tests
pytest test_file.py -m "not cppsim and not rtlsim" -v

# Only cppsim tests
pytest test_file.py -m "cppsim" -v

# Only rtlsim tests
pytest test_file.py -m "rtlsim" -v

# Specific test case
pytest test_file.py::TestMyKernel::test_python_execution_vs_golden[baseline] -v

# Parallel execution
pytest test_file.py -n auto -v
```

---

## Next Steps

- **Complete Reference:** See [tests/README.md](README.md) for comprehensive documentation
- **Parity Testing Deep Dive:** See [tests/frameworks/KERNEL_PARITY_TEST_GUIDE.md](frameworks/KERNEL_PARITY_TEST_GUIDE.md)
- **Real Examples:**
  - Simple: [tests/kernels/elementwise_binary/test_add_validation.py](kernels/elementwise_binary/test_add_validation.py)
  - Parity: [tests/kernels/elementwise_binary/test_add_parity.py](kernels/elementwise_binary/test_add_parity.py)
