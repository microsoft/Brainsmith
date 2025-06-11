# BrainSmith Kernels: Contributor Guide

## How to Add a Custom Kernel

After the Registry Dictionary Pattern refactoring, adding kernels is extremely simple:

### **Two-Step Process:**

1. **Create kernel directory** with files + `kernel.yaml` manifest
2. **Add one line** to the registry dictionary

**That's it!** No custom loader code, no complex registration APIs.

---

## Step-by-Step Example

### 1. Create Kernel Package Directory

```bash
brainsmith/libraries/kernels/my_awesome_conv/
├── kernel.yaml                    # Package manifest (required)
├── my_awesome_conv_source_RTL.sv  # RTL source code
├── my_awesome_conv_hw_custom_op.py # Python HW custom operation
├── my_awesome_conv_rtl_backend.py  # RTL backend implementation
├── my_awesome_conv_wrapper.v       # Verilog wrapper
└── tests/                          # Optional test files
    ├── test_basic.py
    └── test_performance.py
```

### 2. Create `kernel.yaml` Manifest

```yaml
name: "my_awesome_conv"
operator_type: "Convolution"
backend: "HLS"
version: "1.0.0"
author: "Your Name"
license: "MIT"
description: "Ultra-optimized convolution kernel for edge inference"

# Parameter specifications
parameters:
  pe_range: [1, 64]
  simd_range: [2, 32]
  supported_datatypes: ["int8", "int16"]
  memory_modes: ["internal", "external"]

# File mappings (logical name -> actual file path)
files:
  rtl_source: "my_awesome_conv_source_RTL.sv"
  hw_custom_op: "my_awesome_conv_hw_custom_op.py"
  rtl_backend: "my_awesome_conv_rtl_backend.py"
  rtl_wrapper: "my_awesome_conv_wrapper.v"

# Performance characteristics
performance:
  base_throughput: 5000
  base_latency: 4
  resource_estimates:
    luts: 8000
    dsps: 32
    brams: 6

# Validation and testing
validation:
  test_cases: ["test_basic.py", "test_performance.py"]
  verified: true
  compatibility: ["finn-0.9+", "vivado-2022.1+"]
  last_tested: "2024-12-01"
```

### 3. Register in Main Dictionary

Edit `brainsmith/libraries/kernels/__init__.py`:

```python
# Simple registry maps kernel names to their package directories
AVAILABLE_KERNELS = {
    "conv2d_hls": "conv2d_hls",
    "matmul_rtl": "matmul_rtl",
    "my_awesome_conv": "my_awesome_conv",  # ADD THIS LINE
}
```

### 4. Test Your Kernel

```python
from brainsmith.libraries.kernels import get_kernel, list_kernels

# Check that your kernel is available
kernels = list_kernels()
assert "my_awesome_conv" in kernels

# Get your kernel
my_kernel = get_kernel("my_awesome_conv")
assert my_kernel.name == "my_awesome_conv"

print("✅ Kernel successfully added!")
```

---

## Key Benefits

✅ **Simple**: Just 2 steps (create files + add registry line)  
✅ **Fast**: 5ms discovery vs 1s+ filesystem scanning  
✅ **Explicit**: No magical discovery, clear error messages  
✅ **Testable**: Unit testable with no filesystem dependencies  
✅ **Generic**: One YAML loader handles all kernels  

## What You DON'T Need

❌ Custom package loader classes  
❌ Complex Python interfaces  
❌ Understanding of discovery mechanisms  
❌ Filesystem scanning magic  

The Registry Dictionary Pattern eliminates all complexity while maintaining full functionality for complex kernel packages.