# Wholistic Kernel Test Pipeline Design

**Date**: 2025-10-31
**Objective**: Make test infrastructure mirror the complete production transformation flow

---

## Production Flow (3 Stages)

```
Stage 1: ONNX Node
  │  Standard ONNX operator (Add, Mul, MatMul, etc.)
  │  Execution: Python/NumPy reference implementation
  │  Verify: Correct ONNX semantics
  ↓
  Transform: InferAddStreamsLayer() / InferKernelList()
  ↓
Stage 2: HWCustomOp/KernelOp (Base Kernel)
  │  Hardware kernel WITHOUT backend specialization
  │  Example: AddStreams (not AddStreams_hls)
  │  Execution: Python execute_node() (hardware semantics)
  │  Verify: Shapes, datatypes, folding, streaming logic
  ↓
  Transform: SpecializeLayers(fpgapart)
  ↓
Stage 3: Backend (_hls / _rtl)
  │  Specialized backend WITH code generation capability
  │  Example: AddStreams_hls (has HLSBackend inheritance)
  │  Execution: cppsim (C++ simulation) / rtlsim (RTL simulation)
  │  Verify: Code generation, compilation, hardware correctness
```

---

## Current Test Infrastructure (INCOMPLETE)

### What We Have:
```python
# Stage 1 → Stage 2 (works)
def run_inference_pipeline():
    model, node_name = make_test_model()  # ONNX Node
    model = model.transform(InferKernelList())  # → HWCustomOp
    return op, model

# Stage 2 verification (works)
def test_python_execution_vs_golden():
    op, model = run_inference_pipeline()
    executor = PythonExecutor()  # execute_node()
    outputs = executor.execute(op, model, inputs)
    assert outputs == golden
```

### What's Missing:
```python
# Stage 2 → Stage 3 (MISSING!)
# No way to specialize base kernel to backend

# Stage 3 verification (BROKEN!)
def test_cppsim_execution_vs_golden():
    op, model = run_inference_pipeline()  # Gets AddStreams (base)
    executor = CppSimExecutor()
    outputs = executor.execute(op, model, inputs)  # SKIPS! Not HLSBackend
```

---

## Desired Architecture

### Option 1: Extended PipelineRunner

```python
class PipelineRunner:
    def run(
        self,
        model_factory,
        transform,
        configure_fn=None,
        # NEW: Optional backend specialization
        specialize_backend: Optional[str] = None,  # "hls" or "rtl"
        fpgapart: Optional[str] = None,
    ):
        # Stage 1 → Stage 2 (existing)
        model, node_name = model_factory()
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(transform)
        op = get_hw_custom_op(...)
        if configure_fn:
            configure_fn(op, model)

        # Stage 2 → Stage 3 (NEW)
        if specialize_backend and fpgapart:
            from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
            model = model.transform(SpecializeLayers(fpgapart))
            # Re-get op (now specialized)
            specialized_node = find_node_by_name(model, op.onnx_node.name)
            op = get_hw_custom_op(specialized_node, model)

        return op, model
```

**Usage:**
```python
# Stage 2 only (current)
op, model = runner.run(
    model_factory=make_test_model,
    transform=InferKernelList()
)

# Stage 3 (NEW)
op, model = runner.run(
    model_factory=make_test_model,
    transform=InferKernelList(),
    specialize_backend="hls",
    fpgapart="xc7z020clg400-1"
)
```

### Option 2: Three-Stage Pipeline (Explicit)

```python
class KernelPipeline:
    """Represents the complete 3-stage transformation."""

    def __init__(self):
        self.stage1_model = None  # ONNX
        self.stage2_op = None     # Base kernel
        self.stage2_model = None
        self.stage3_op = None     # Backend
        self.stage3_model = None

    def to_base_kernel(self, model_factory, transform, configure_fn=None):
        """Stage 1 → Stage 2: ONNX → HWCustomOp."""
        runner = PipelineRunner()
        self.stage2_op, self.stage2_model = runner.run(
            model_factory, transform, configure_fn
        )
        return self

    def to_backend(self, fpgapart: str, backend: str = "hls"):
        """Stage 2 → Stage 3: HWCustomOp → Backend."""
        if self.stage2_op is None:
            raise RuntimeError("Must call to_base_kernel() first")

        model = self.stage2_model.transform(SpecializeLayers(fpgapart))
        # Find specialized node
        specialized_node = ...
        self.stage3_op = get_hw_custom_op(specialized_node, model)
        self.stage3_model = model
        return self

    def get_base(self):
        """Get Stage 2 (base kernel)."""
        return self.stage2_op, self.stage2_model

    def get_backend(self):
        """Get Stage 3 (specialized backend)."""
        if self.stage3_op is None:
            raise RuntimeError("Must call to_backend() first")
        return self.stage3_op, self.stage3_model
```

**Usage:**
```python
# Build pipeline
pipeline = (KernelPipeline()
    .to_base_kernel(make_test_model, InferKernelList())
    .to_backend(fpgapart="xc7z020clg400-1", backend="hls"))

# Test Stage 2
op, model = pipeline.get_base()
executor = PythonExecutor()
outputs = executor.execute(op, model, inputs)

# Test Stage 3
op, model = pipeline.get_backend()
executor = CppSimExecutor()
outputs = executor.execute(op, model, inputs)
```

### Option 3: Pipeline Factory (Flexible)

```python
class PipelineFactory:
    """Factory for creating pipelines at different stages."""

    @staticmethod
    def create_base_kernel_pipeline(model_factory, transform, configure_fn=None):
        """Create pipeline up to Stage 2."""
        runner = PipelineRunner()
        return runner.run(model_factory, transform, configure_fn)

    @staticmethod
    def create_backend_pipeline(
        model_factory,
        transform,
        fpgapart: str,
        backend: str = "hls",
        configure_fn=None
    ):
        """Create pipeline up to Stage 3."""
        # Stage 2
        op, model = PipelineFactory.create_base_kernel_pipeline(
            model_factory, transform, configure_fn
        )

        # Stage 3
        model = model.transform(SpecializeLayers(fpgapart))
        specialized_node = find_node_by_name(model, op.onnx_node.name)
        op = get_hw_custom_op(specialized_node, model)

        return op, model
```

**Usage:**
```python
# Stage 2 testing
op, model = PipelineFactory.create_base_kernel_pipeline(
    make_test_model, InferKernelList()
)

# Stage 3 testing
op, model = PipelineFactory.create_backend_pipeline(
    make_test_model, InferKernelList(),
    fpgapart="xc7z020clg400-1",
    backend="hls"
)
```

---

## Integration with Test Framework

### KernelTestConfig (Base)

```python
class KernelTestConfig(ABC):
    """Minimal configuration for kernel tests."""

    # Required
    @abstractmethod
    def make_test_model(self): pass

    @abstractmethod
    def get_num_inputs(self): pass

    @abstractmethod
    def get_num_outputs(self): pass

    # NEW: Optional backend configuration
    def get_backend_fpgapart(self) -> Optional[str]:
        """Return FPGA part for backend specialization.

        Returns:
            None: Test only base kernel (Python execution)
            str: FPGA part for backend specialization (enables cppsim/rtlsim)
        """
        return None

    def get_backend_type(self) -> str:
        """Return backend type (hls or rtl). Default: hls."""
        return "hls"
```

### SingleKernelTest (Updated)

```python
class SingleKernelTest(KernelTestConfig):
    def run_inference_pipeline(self, to_backend: bool = False):
        """Run pipeline to Stage 2 or Stage 3.

        Args:
            to_backend: If True, specialize to backend (Stage 3)

        Returns:
            (op, model) at requested stage
        """
        # Always go to Stage 2
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_test_model,
            transform=self.get_kernel_inference_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model)
        )

        # Optionally go to Stage 3
        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart is None:
                pytest.skip("Backend testing not enabled (get_backend_fpgapart() returns None)")

            backend_type = self.get_backend_type()
            model = model.transform(SpecializeLayers(fpgapart))

            # Find specialized node
            specialized_node = self._find_node_by_name(model, op.onnx_node.name)
            op = get_hw_custom_op(specialized_node, model)

        return op, model

    def test_python_execution_vs_golden(self):
        """Test Stage 2: Base kernel (Python execution)."""
        op, model = self.run_inference_pipeline(to_backend=False)
        # ... test implementation

    def test_cppsim_execution_vs_golden(self):
        """Test Stage 3: Backend (cppsim execution)."""
        op, model = self.run_inference_pipeline(to_backend=True)
        # ... test implementation

    def test_rtlsim_execution_vs_golden(self):
        """Test Stage 3: Backend (rtlsim execution)."""
        op, model = self.run_inference_pipeline(to_backend=True)
        # ... test implementation
```

---

## FINN Pattern Match

This matches FINN's test pattern exactly:

```python
# FINN's test_fpgadataflow_addstreams.py
def test_fpgadataflow_addstreams(idts, ch, fold, exec_mode):
    # Stage 1: ONNX
    model = make_addstreams_modelwrapper(ch, idts)

    # Verify Stage 1
    y_expected = x1 + x2
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    # Stage 2: Base kernel
    model = model.transform(to_hw.InferAddStreamsLayer())
    addstreams_node.set_nodeattr("PE", pe)

    # Stage 3: Backend
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Prepare execution
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))

    # Verify Stage 3
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
```

Our framework should make this pattern the **default**, not optional.

---

## Recommendation

**Option 1 (Extended PipelineRunner) + Updated SingleKernelTest**

**Why:**
1. Minimal API surface (one parameter change)
2. Backward compatible (optional parameters)
3. Clear intent: `to_backend=True` means "go to Stage 3"
4. Matches FINN pattern exactly
5. Foundation for all test frameworks

**Implementation Steps:**

1. Update `PipelineRunner` to support optional backend specialization
2. Update `SingleKernelTest.run_inference_pipeline()` to accept `to_backend` parameter
3. Update execution tests to use `to_backend=True` for cppsim/rtlsim
4. Add `get_backend_fpgapart()` hook to `KernelTestConfig`
5. Update `DualKernelTest` to use the same pattern

**Result:** Wholistic pipeline that mirrors production, with verification at each stage.

---

## Next: Implementation

Focus on perfecting this core pipeline first. Once solid, it becomes the foundation for:
- SingleKernelTest (already uses it)
- DualKernelTest (will use it)
- Any future test frameworks

**Arete**: Get the foundation right, everything else follows.
