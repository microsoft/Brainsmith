# Kernel Integrator Pragma Reference

## Table of Contents

- [Overview](#overview)
- [Pragma Types](#pragma-types)
  - [TOP_MODULE](#top_module)
  - [INCLUDE_RTL](#include_rtl)
  - [BDIM (Block Dimension)](#bdim-block-dimension)
  - [SDIM (Stream Dimension)](#sdim-stream-dimension)
  - [RELATIONSHIP](#relationship)
  - [DATATYPE](#datatype)
  - [DATATYPE_CONSTRAINT](#datatype_constraint)
  - [DERIVED_PARAMETER](#derived_parameter)
  - [WEIGHT](#weight)
  - [ALIAS](#alias)
  - [AXILITE_PARAM](#axilite_param)

## Overview

Pragmas are special single-line comments that provide metadata to the Kernel Integrator. They must follow the format:

```systemverilog
// @brainsmith <PRAGMA_TYPE> <arguments...>
```

All pragmas must:
- Start with `// @brainsmith` (single-line comment)
- Be followed by a valid pragma type
- Include required arguments in the correct order
- Appear before or within the module definition

## Pragma Types

### TOP_MODULE

Specifies which module to process when multiple modules exist in a file.

**Syntax:**
```systemverilog
// @brainsmith TOP_MODULE <module_name>
```

**Arguments:**
- `module_name` - Name of the module to process

**Example:**
```systemverilog
// @brainsmith TOP_MODULE thresholding_axi
```

**Notes:**
- Required only when RTL file contains multiple modules
- Must match exact module name in the RTL

---

### INCLUDE_RTL

Specifies additional RTL files to include in the generated wrapper.

**Syntax:**
```systemverilog
// @brainsmith INCLUDE_RTL <file_path>
```

**Arguments:**
- `file_path` - Path to RTL file (absolute or relative)

**Examples:**
```systemverilog
// @brainsmith INCLUDE_RTL helper_functions.sv
// @brainsmith INCLUDE_RTL ../common/axi_infrastructure.v
// @brainsmith INCLUDE_RTL /opt/rtl_lib/protocols/axi_lite.sv
```

**Notes:**
- Files are included in order specified
- Paths are resolved relative to main RTL file
- Use for dependencies and helper modules

---

### BDIM (Block Dimension)

Defines block-level tiling dimensions for an interface.

**Syntax:**
```systemverilog
// @brainsmith BDIM <interface> <attribute_name> SHAPE=<shape_expr>
```

**Arguments:**
- `interface` - Interface name
- `attribute_name` - Name for the dimension attribute
- `shape_expr` - Python list expression for shape

**Examples:**
```systemverilog
// @brainsmith BDIM input input_bdim SHAPE=[CHANNELS]
// @brainsmith BDIM output output_bdim SHAPE=[NUM_OUTPUTS]
// @brainsmith BDIM weights weight_bdim SHAPE=[KERNEL_H, KERNEL_W]
```

**Notes:**
- Shape expressions can reference RTL parameters
- Used for FINN's dataflow analysis
- Affects memory layout and parallelism

---

### SDIM (Stream Dimension)

Defines stream-level tiling dimensions for an interface.

**Syntax:**
```systemverilog
// @brainsmith SDIM <interface> <attribute_name> SHAPE=<shape_expr>
```

**Arguments:**
- `interface` - Interface name  
- `attribute_name` - Name for the dimension attribute
- `shape_expr` - Python list expression for shape

**Examples:**
```systemverilog
// @brainsmith SDIM input input_sdim SHAPE=[PE]
// @brainsmith SDIM output output_sdim SHAPE=[NPE]
// @brainsmith SDIM input simd SHAPE=[SIMD_WIDTH]
```

**Notes:**
- Represents parallelism within the stream
- Must be compatible with BDIM settings
- Critical for performance optimization

---

### RELATIONSHIP

Defines dimensional constraints between interfaces.

**Syntax:**
```systemverilog
// @brainsmith RELATIONSHIP <source> <target> <type> [args...]
```

**Types:**
- `EQUAL` - All dimensions must match
- `DEPENDENT <src_dim> <tgt_dim> <dep_type> [scale]` - Dimension dependency
  - `dep_type`: `copy`, `scaled`, `min`
- `MULTIPLE <src_dim> <tgt_dim> [factor=N]` - Multiple relationship
- `DIVISIBLE <src_dim> <tgt_dim>` - Divisibility constraint

**Examples:**
```systemverilog
// @brainsmith RELATIONSHIP input output EQUAL
// @brainsmith RELATIONSHIP input output DEPENDENT 0 0 copy
// @brainsmith RELATIONSHIP input output DEPENDENT 1 1 scaled SCALE_FACTOR
// @brainsmith RELATIONSHIP input output MULTIPLE 0 0 factor=4
// @brainsmith RELATIONSHIP input output DIVISIBLE 1 1
```

---

### DATATYPE

Maps interface datatype properties to RTL parameters, enabling full [QONNX datatype](https://qonnx.readthedocs.io/en/latest/api/qonnx.core.datatype.html) representation. 

**Syntax:**
```systemverilog
// @brainsmith DATATYPE <interface> <property> <parameter_name>
```

**Arguments:**
- `interface` - Interface name
- `property` - Datatype property to map (see below)
- `parameter_name` - RTL parameter controlling this property

**Supported Properties:**
- `width` - Bit width of the datatype
- `signed` - Whether the datatype is signed (0 or 1)
- `format` - Format specifier for the datatype
- `bias` - Bias value for the datatype
- `fractional_width` - Number of fractional bits (fixed-point)
- `exponent_width` - Exponent bit width (floating-point)
- `mantissa_width` - Mantissa bit width (floating-point)

**Examples:**
```systemverilog
// Basic width mapping
// @brainsmith DATATYPE input width DATA_WIDTH
// @brainsmith DATATYPE output width OUT_WIDTH

// Signed/unsigned control
// @brainsmith DATATYPE input signed INPUT_SIGNED
// @brainsmith DATATYPE output signed OUTPUT_SIGNED

// Fixed-point configuration
// @brainsmith DATATYPE weights width WEIGHT_WIDTH
// @brainsmith DATATYPE weights fractional_width WEIGHT_FRAC_BITS
// @brainsmith DATATYPE weights signed WEIGHT_SIGNED

// Floating-point configuration
// @brainsmith DATATYPE input width FP_WIDTH
// @brainsmith DATATYPE input exponent_width FP_EXP_WIDTH
// @brainsmith DATATYPE input mantissa_width FP_MANT_WIDTH

// Bias configuration
// @brainsmith DATATYPE output bias OUTPUT_BIAS
```

---

### DATATYPE_CONSTRAINT

Defines allowed datatype ranges for interfaces.

**Syntax:**
```systemverilog
// @brainsmith DATATYPE_CONSTRAINT <interface> <base_type> <min_width> <max_width>
```

**Arguments:**
- `interface` - Interface name (e.g., "input", "output", "weights")
- `base_type` - Base datatype or "*" for any type
- `min_width` - Minimum bit width
- `max_width` - Maximum bit width

**Examples:**
```systemverilog
// @brainsmith DATATYPE_CONSTRAINT input * 1 32      // Any type, 1-32 bits
// @brainsmith DATATYPE_CONSTRAINT output INT 8 8    // Integer, exactly 8 bits
// @brainsmith DATATYPE_CONSTRAINT weights * 1 16    // Any type, 1-16 bits
```

**Valid Base Types:**
- `*` - Any datatype
- `INT` - Integer types (signed/unsigned)
- `FLOAT` - Floating-point types
- `FIXED` - Fixed-point types

---

### DERIVED_PARAMETER

Links module parameters to Python expressions.

**Syntax:**
```systemverilog
// @brainsmith DERIVED_PARAMETER <parameter_name> = <python_expression>
```

**Arguments:**
- `parameter_name` - RTL parameter name
- `python_expression` - Valid Python expression

**Examples:**
```systemverilog
// @brainsmith DERIVED_PARAMETER TOTAL_BITS = DATA_WIDTH * NUM_CHANNELS
// @brainsmith DERIVED_PARAMETER ADDR_BITS = math.ceil(math.log2(DEPTH))
// @brainsmith DERIVED_PARAMETER OUTPUT_SIZE = INPUT_SIZE // STRIDE + 1
```

**Available in Expressions:**
- Other RTL parameters
- Python math module functions
- Basic arithmetic operators

---

### WEIGHT

Marks an interface as containing weight data.

**Syntax:**
```systemverilog
// @brainsmith WEIGHT <interface_name>
```

**Arguments:**
- `interface_name` - Name of the weight interface

**Example:**
```systemverilog
// @brainsmith WEIGHT threshold
// @brainsmith WEIGHT kernel_weights
```

**Effects:**
- Interface is marked with `is_weight=True` in generated code
- Affects dataflow graph construction
- May change interface handling in FINN

---

### ALIAS

Exposes RTL parameters with different names in the HWCustomOp.

**Syntax:**
```systemverilog
// @brainsmith ALIAS <rtl_parameter> <exposed_name>
```

**Arguments:**
- `rtl_parameter` - Original parameter name in RTL
- `exposed_name` - Name to expose in Python interface

**Examples:**
```systemverilog
// @brainsmith ALIAS T_WIDTH threshold_width
// @brainsmith ALIAS USE_DSP enable_dsp_mode
// @brainsmith ALIAS FIFO_DEPTH input_buffer_depth
```

**Use Cases:**
- Rename parameters for better Python API
- Expose internal parameters
- Create user-friendly names

---

### AXILITE_PARAM

Links a parameter to control a specific property of an AXI-Lite interface.

**Syntax:**
```systemverilog
// @brainsmith AXILITE_PARAM <param_name> <interface_name> <property>
```

**Arguments:**
- `param_name` - Parameter to link (must exist in module parameters)
- `interface_name` - Target AXI-Lite interface name
- `property` - Interface property to control: `enable`, `data_width`, or `addr_width`

**Example:**
```systemverilog
// @brainsmith AXILITE_PARAM USE_AXILITE threshold enable
```

**Effects:**
- Moves parameter from general parameters to interface-specific control
- Parameter controls the specified interface property
- For `enable` property: Controls whether the interface is instantiated
- For `data_width`/`addr_width`: Sets the interface bus widths

---

### Validation Tips

- Use `--validate` flag to check pragmas without generation
- Enable `--verbose` for detailed parsing information
- Start with minimal pragmas and add incrementally
- Check generated metadata with `--info` flag
