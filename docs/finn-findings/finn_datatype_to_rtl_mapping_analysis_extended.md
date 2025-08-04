# FINN Datatype to RTL Parameter Mapping Analysis - Extended

## Support for Different Template Variable Names and Multiple Inputs

Based on the user's questions about RTL template variables having different names and multiple inputs, here's an extended analysis of FINN's template system flexibility.

## Template Variable Naming Flexibility

### 1. No Fixed Naming Convention Required

FINN's template system is **completely flexible** regarding variable names. The template engine performs simple string replacement:

```python
# From hwcustomop.py line 337-340
for key in code_gen_dict:
    # transform list into long string separated by '\n'
    code_gen_line = "\n".join(code_gen_dict[key])
    template_wrapper = template_wrapper.replace(key, code_gen_line)
```

**Key Points:**
- **Any variable name** can be used in templates (e.g., `$FOO$`, `$MY_CUSTOM_VAR$`, `$ACTIVATION_BITWIDTH$`)
- **No predefined set** of required variables
- Each kernel implementation defines its own template variables
- Template variables are matched by **exact string replacement**

### 2. Examples of Different Naming Patterns

**Standard MVAU Variables:**
```python
# MVAU_rtl uses these template variables:
code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]
```

**Thresholding RTL Variables (Different Names):**
```python
# thresholding_rtl.py uses different naming:
code_gen_dict["$WI$"] = [str(i_bitwidth)]  # input precision
code_gen_dict["$WT$"] = [str(wdt.bitwidth())]  # threshold precision  
code_gen_dict["$N$"] = [str(o_bitwidth)]  # output precision
```

**Convolution Input Generator Variables:**
```python
# convolutioninputgenerator_rtl.py uses domain-specific names:
code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
code_gen_dict["$SIMD$"] = [str(simd)]
code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
```

**Streaming FIFO Variables:**
```python
# streamingfifo_rtl.py uses different pattern entirely:
code_gen_dict = {
    "TOP_MODULE_NAME": topname,  # No $ delimiters!
    "COUNT_WIDTH": f"{count_width}",
    "IN_RANGE": "[{}:0]".format(in_width - 1),
}
```

## Multiple Input Support

### 1. Indexed Template Variables

For operations with multiple inputs, FINN uses **indexed naming patterns**:

#### StreamingEltwise Example (2 Inputs):
```python
# streamingeltwise_hls.py handles 2 different input datatypes:
def strm_decl(self):
    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
        'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
    )
    self.code_gen_dict["$STREAMDECLARATIONS$"].append(
        'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(self.get_instream_width(1))
    )

def blackboxfunction(self):
    self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
        """void {}(hls::stream<ap_uint<{}>> &in0_V, hls::stream<ap_uint<{}>> &in1_V,
            hls::stream<ap_uint<{}>> &out0_V)""".format(
            self.onnx_node.name,
            self.get_nodeattr("PE") * self.get_input_datatype(0).bitwidth(),  # Input 0
            self.get_nodeattr("PE") * self.get_input_datatype(1).bitwidth(),  # Input 1
            self.get_nodeattr("PE") * self.get_output_datatype().bitwidth(),
        )
    ]
```

### 2. Advanced Multiple Input Example: ElementwiseBinary

The [`ElementwiseBinaryOperation_hls`](src/finn/custom_op/fpgadataflow/hls/elementwise_binary_hls.py:44) shows sophisticated handling of multiple inputs with **different datatypes**:

#### Dynamic Input Style Detection:
```python
# Lines 264-286: Conditional stream generation based on input source
if self.lhs_style == "input":  # Left input from stream
    self.code_gen_dict["$READNPYDATA$"] += [
        f'npy2apintstream<LhsPacked, LhsType, LhsWidth, {lhs_carrier_dtype}>(',
        f'"{code_gen_dir}/lhs.npy", lhs_V, false',
        ');'
    ]

if self.rhs_style == "input":  # Right input from stream  
    self.code_gen_dict["$READNPYDATA$"] += [
        f'npy2apintstream<RhsPacked, RhsType, RhsWidth, {rhs_carrier_dtype}>(',
        f'"{code_gen_dir}/rhs.npy", rhs_V, false',
        ');'
    ]
```

#### Datatype-Specific Template Variables:
```python
# Lines 230-253: Separate type definitions for each input
self.code_gen_dict["$DEFINES$"] = [
    f"using LhsType = {self.lhs_dtype.get_hls_datatype_str()};",  # Left input type
    f"using RhsType = {self.rhs_dtype.get_hls_datatype_str()};",  # Right input type
    f"using OutType = {self.out_dtype.get_hls_datatype_str()};",   # Output type
    f"static constexpr auto LhsWidth = {self.lhs_dtype.bitwidth()};",
    f"static constexpr auto RhsWidth = {self.rhs_dtype.bitwidth()};",
    f"using LhsPacked = ap_uint<{self.get_instream_width(ind=0)}>;",
    f"using RhsPacked = ap_uint<{self.get_instream_width(ind=1)}>;",
]
```

### 3. Conditional Interface Generation

```python
# Lines 544-551: Conditional function signature based on input availability
self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
    f"void {self.onnx_node.name} (",
    "  LhsStream &lhs_V," if runtime_lhs else "",  # Conditional input
    "  RhsStream &rhs_V," if runtime_rhs else "",  # Conditional input
    "  OutStream &out_V",                          # Always present output
    ")",
]
```

### 4. Multi-Input Attribute Management

FINN supports **separate datatype attributes** for each input:

```python
# StreamingEltwise node attributes (line 62-63):
"inputDataType0": ("s", True, ""),  # First input datatype
"inputDataType1": ("s", True, ""),  # Second input datatype

# Access pattern in get_input_datatype():
def get_input_datatype(self, ind=0):
    return DataType[self.get_nodeattr("inputDataType" + str(ind))]
```

## Template Variable Resolution Process

### 1. No Schema Validation

```python
# Template replacement is simple string substitution - no validation
template_wrapper = template_wrapper.replace(key, code_gen_line)
```

**Implications:**
- **Unused variables** are left as-is in templates (could cause compilation errors)
- **Missing variables** cause no immediate error (only during SystemVerilog compilation)
- **Typos in variable names** are not caught by FINN

### 2. Variable Scope is Per-Implementation

Each [`HWCustomOp`](src/finn/custom_op/fpgadataflow/hwcustomop.py:44) implementation defines its own template variables:

```python
# MVAU_rtl defines these variables:
"$ACTIVATION_WIDTH$", "$WEIGHT_WIDTH$", "$ACCU_WIDTH$", "$SIGNED_ACTIVATIONS$"

# Thresholding_rtl defines different variables:
"$WI$", "$WT$", "$N$", "$SIGNED$", "$FPARG$"

# ConvInputGen_rtl defines domain-specific variables:
"$BIT_WIDTH$", "$SIMD$", "$MMV_IN$", "$MMV_OUT$"
```

### 3. Template and Implementation Co-Design

Templates and implementations are **tightly coupled**:

- Each RTL template expects specific variable names
- Each Python implementation must provide exactly those variables
- **No automatic mapping** between different naming conventions

## Key Insights

### 1. Complete Naming Freedom

**FINN places no restrictions on template variable names.** Kernel developers can:
- Use any naming convention (`$ACTIVATION_WIDTH$` vs `$ACT_BITS$` vs `$INPUT_PRECISION$`)
- Define domain-specific variable names (`$KERNEL_SIZE$`, `$STRIDE$`, `$DILATION$`)
- Use different delimiter patterns (some use `$VAR$`, others use `VAR`)

### 2. Multiple Input Support is Well-Established

**FINN fully supports multiple inputs** through:
- **Indexed access patterns:** `get_input_datatype(0)`, `get_input_datatype(1)`
- **Separate attribute storage:** `inputDataType0`, `inputDataType1`
- **Conditional code generation:** Different logic for stream vs constant inputs
- **Per-input template variables:** `LhsType`, `RhsType` vs generic approaches

### 3. Flexibility vs Validation Trade-off

**Benefits:**
- Maximum flexibility for kernel developers
- Easy adaptation to diverse RTL templates
- Support for domain-specific naming

**Drawbacks:**
- No compile-time validation of template variables
- Tight coupling between templates and implementations
- Potential for silent failures (unused/misspelled variables)

## Recommendations for iPar/wPar System

### 1. Interface-Based Variable Generation

```python
# Future approach: Generate variables from interface metadata
def generate_template_variables(self):
    for i, iface in enumerate(self.input_interfaces):
        key = f"$INPUT_{i}_WIDTH$"  # or f"${iface.name.upper()}_WIDTH$"
        self.code_gen_dict[key] = [str(iface.dtype.bitwidth())]
        
        key = f"$INPUT_{i}_SIGNED$"
        self.code_gen_dict[key] = [str(1 if iface.dtype.min() < 0 else 0)]
```

### 2. Template Variable Validation

```python
# Proposed: Validate template variables against expected set
def validate_template_variables(self, template_path):
    """Check that all template variables are provided"""
    with open(template_path) as f:
        template_content = f.read()
    
    # Extract all $VAR$ patterns
    import re
    expected_vars = set(re.findall(r'\$([A-Z_]+)\$', template_content))
    provided_vars = set(key.strip('$') for key in self.code_gen_dict.keys())
    
    missing = expected_vars - provided_vars
    if missing:
        raise ValueError(f"Missing template variables: {missing}")
```

### 3. Standardized Naming Convention

```python
# Proposed: Standard interface-based naming
TEMPLATE_VAR_PATTERNS = {
    "input_width": "$INPUT_{index}_WIDTH$",
    "input_signed": "$INPUT_{index}_SIGNED$", 
    "input_parallel": "$INPUT_{index}_PARALLEL$",
    "output_width": "$OUTPUT_{index}_WIDTH$",
    # ... etc
}
```

This would provide **both flexibility and consistency** while maintaining backward compatibility with existing implementations.