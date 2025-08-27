# Weight File Generation Guide

This guide explains how to implement weight file generation for Brainsmith RTL kernels that have weight or parameter inputs.

## Overview

When your RTL kernel has weight interfaces (marked with `@brainsmith weight`), you need to implement three methods in your RTL backend:

1. `generate_init_files()` - Main entry point for generating all weight files
2. `make_weight_file()` - Utility for writing a single weight file in specific format
3. `get_all_meminit_filenames()` - Returns list of all generated files

## Understanding Your RTL's Requirements

Before implementing, understand what your RTL expects:

### 1. File Organization
- **Single file**: All weights in one file
- **Multiple files**: Weights distributed across files (by PE, channel, memory bank, etc.)
- **File naming**: What naming convention does your RTL use?

### 2. Data Format
- **Encoding**: Hex (`FF`), binary (`10110101`), decimal (`255`)
- **Prefix**: With (`0xFF`) or without (`FF`) prefix
- **Width**: Fixed width with padding or variable
- **Packing**: One value per line or multiple packed values

### 3. Memory Layout
- **Linear**: Weights in sequential order
- **Interleaved**: PE-interleaved, channel-interleaved
- **Hierarchical**: Binary trees, multi-stage structures

## Implementation Steps

### Step 1: Extract Weights from Model

```python
def generate_init_files(self, model, code_gen_dir):
    # Get weight tensor from ONNX model
    # Weights are typically the second input (index 1)
    weights = model.get_initializer(self.onnx_node.input[1])
    if weights is None:
        return  # No weights to generate
    
    # For multiple weight inputs, check additional inputs
    # bias = model.get_initializer(self.onnx_node.input[2])
```

### Step 2: Get Required Parameters

```python
    # Get parameters that affect file generation
    pe = self.get_nodeattr("PE")  # Processing elements
    channels = self.get_nodeattr("NumChannels")
    
    # Get datatypes
    wdt = self.get_input_datatype(1)  # Weight datatype
    odt = self.get_output_datatype()   # Output datatype
    
    # Calculate derived values
    channel_fold = channels // pe
    o_bitwidth = odt.bitwidth()
```

### Step 3: Transform Weights

Common transformations include:

#### Reshaping for Memory Layout
```python
    # Example: Reshape for PE-first layout
    # Original shape: [channels, ...]
    # Target shape: [channel_fold, pe, ...]
    weights_reshaped = weights.reshape(channel_fold, pe, -1)
    weights_pe_first = weights_reshaped.transpose(1, 0, 2)
```

#### Quantization Adjustments
```python
    # Example: Narrow quantization padding
    expected_levels = 2**o_bitwidth - 1
    actual_levels = weights.shape[-1]
    
    if expected_levels != actual_levels:
        # Prepend minimum value
        min_val = wdt.min()
        weights = np.insert(weights, 0, min_val, axis=-1)
```

#### Broadcasting Single Values
```python
    # Example: Broadcast single threshold to all channels
    if weights.shape[0] == 1:
        weights = np.broadcast_to(weights, (pe, expected_levels))
```

### Step 4: Generate Files

#### Single File Example
```python
    # Simple case: all weights in one file
    weight_file = os.path.join(code_gen_dir, f"{self.onnx_node.name}_weights.dat")
    self.make_weight_file(weights, "decoupled", weight_file)
```

#### Multiple Files Example
```python
    # Distributed across PEs
    for pe_idx in range(pe):
        pe_weights = weights_pe_first[pe_idx]
        weight_file = os.path.join(
            code_gen_dir, 
            f"{self.onnx_node.name}_weights_pe{pe_idx}.dat"
        )
        self.make_weight_file(pe_weights, "decoupled", weight_file)
```

#### Complex Structure Example (Binary Tree)
```python
    # Thresholding-style binary tree
    for stage in range(o_bitwidth):
        for pe_idx in range(pe):
            # Calculate data for this stage/PE
            stage_size = 2**stage
            stage_data = extract_stage_data(weights, stage, pe_idx)
            
            weight_file = os.path.join(
                code_gen_dir,
                f"{self.onnx_node.name}_threshs_{pe_idx}_{stage}.dat"
            )
            self.make_weight_file(stage_data, "decoupled", weight_file)
```

### Step 5: Update RTL Parameters

In `prepare_codegen_rtl_values()`, add file path parameters:

```python
def prepare_codegen_rtl_values(self, model):
    code_gen_dict = super().prepare_codegen_rtl_values(model)
    
    # Add weight file path for RTL
    # Example: parameter WEIGHTS_PATH = "./MyKernel_weights_"
    code_gen_dict["$WEIGHTS_PATH$"] = [f'"./{self.onnx_node.name}_weights_"']
    
    return code_gen_dict
```

## Implementing make_weight_file()

This method writes weights in the specific format your RTL expects.

### Hex Format (Most Common)
```python
def make_weight_file(self, weights, weight_file_mode, weight_file_name):
    from finn.util.data_packing import pack_innermost_dim_as_hex_string
    from qonnx.util.basic import roundup_to_integer_multiple
    
    # Determine datatype (may come from weights or node attributes)
    wdt = self.get_input_datatype(1)
    
    # Pack as hex strings
    weights_flat = weights.flatten()
    weights_expanded = np.expand_dims(weights_flat, axis=-1)
    hex_bits = roundup_to_integer_multiple(wdt.bitwidth(), 4)
    
    weights_hex = pack_innermost_dim_as_hex_string(
        weights_expanded,
        wdt,
        hex_bits,
        prefix=""  # No "0x" prefix
    )
    
    # Write to file
    with open(weight_file_name, "w") as f:
        for val in weights_hex.flatten():
            f.write(val + "\n")
```

### Binary Format
```python
def make_weight_file(self, weights, weight_file_mode, weight_file_name):
    wdt = self.get_input_datatype(1)
    width = wdt.bitwidth()
    
    with open(weight_file_name, "w") as f:
        for val in weights.flatten():
            # Convert to integer representation
            if wdt.signed():
                int_val = int(val) & ((1 << width) - 1)
            else:
                int_val = int(val)
            
            # Format as binary string
            bin_str = format(int_val, f'0{width}b')
            f.write(bin_str + "\n")
```

### Decimal Format
```python
def make_weight_file(self, weights, weight_file_mode, weight_file_name):
    with open(weight_file_name, "w") as f:
        for val in weights.flatten():
            f.write(f"{int(val)}\n")
```

## Implementing get_all_meminit_filenames()

This must return the exact list of files created by `generate_init_files()`:

```python
def get_all_meminit_filenames(self, abspath=False):
    dat_files = []
    t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
    
    # Must match your generate_init_files() logic
    pe = self.get_nodeattr("PE")
    
    # Single file case
    # dat_files.append(os.path.join(t_path, f"{self.onnx_node.name}_weights.dat"))
    
    # Multiple files case
    for pe_idx in range(pe):
        weight_file = os.path.join(
            t_path, 
            f"{self.onnx_node.name}_weights_pe{pe_idx}.dat"
        )
        dat_files.append(weight_file)
    
    return dat_files
```

## Complete Example: Matrix-Vector Multiply

```python
def generate_init_files(self, model, code_gen_dir):
    # Extract weight matrix
    weights = model.get_initializer(self.onnx_node.input[1])
    if weights is None:
        return
    
    # Get parameters
    pe = self.get_nodeattr("PE")
    simd = self.get_nodeattr("SIMD")
    mw = self.get_nodeattr("MW")  # Matrix width
    mh = self.get_nodeattr("MH")  # Matrix height
    
    # Reshape for SIMD/PE parallelism
    # Original: [mh, mw]
    # Target: [mh/pe, pe, mw/simd, simd]
    weights_reshaped = weights.reshape(mh//pe, pe, mw//simd, simd)
    
    # Generate one file per PE
    for pe_idx in range(pe):
        pe_weights = weights_reshaped[:, pe_idx, :, :]
        weight_file = os.path.join(
            code_gen_dir,
            f"{self.onnx_node.name}_weights_pe{pe_idx}.dat"
        )
        self.make_weight_file(pe_weights, "decoupled", weight_file)

def make_weight_file(self, weights, weight_file_mode, weight_file_name):
    from finn.util.data_packing import pack_innermost_dim_as_hex_string
    from qonnx.util.basic import roundup_to_integer_multiple
    
    wdt = self.get_input_datatype(1)
    hex_width = roundup_to_integer_multiple(wdt.bitwidth(), 4)
    
    # Pack SIMD dimension
    weights_packed = pack_innermost_dim_as_hex_string(
        weights,
        wdt,
        hex_width,
        prefix=""
    )
    
    with open(weight_file_name, "w") as f:
        for row in weights_packed.flatten():
            f.write(row + "\n")

def get_all_meminit_filenames(self, abspath=False):
    dat_files = []
    t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
    pe = self.get_nodeattr("PE")
    
    for pe_idx in range(pe):
        weight_file = os.path.join(
            t_path,
            f"{self.onnx_node.name}_weights_pe{pe_idx}.dat"
        )
        dat_files.append(weight_file)
    
    return dat_files
```

## Debugging Tips

1. **Print shapes and values** during development:
   ```python
   print(f"Weights shape: {weights.shape}")
   print(f"First few values: {weights.flatten()[:10]}")
   ```

2. **Verify file contents** match RTL expectations:
   ```python
   # After writing, read back and check
   with open(weight_file, "r") as f:
       lines = f.readlines()
       print(f"Generated {len(lines)} lines")
       print(f"First line: {lines[0].strip()}")
   ```

3. **Check parameter consistency** between Python and RTL:
   ```python
   print(f"PE={pe}, channels={channels}, channel_fold={channel_fold}")
   # Ensure these match your RTL parameters
   ```

4. **Test with simple patterns** first:
   ```python
   # Generate test pattern for debugging
   test_weights = np.arange(expected_size).reshape(expected_shape)
   ```

## Common Pitfalls

1. **File count mismatch**: `get_all_meminit_filenames()` must return exactly the files created
2. **Path issues**: Always use `os.path.join()` for cross-platform compatibility
3. **Datatype mismatches**: Ensure packed width matches RTL memory width
4. **Indexing errors**: Complex layouts (like binary trees) need careful index calculation
5. **Missing imports**: Remember to import packing utilities from FINN

## Additional Resources

- See `thresholding_rtl.py` for complex binary tree example
- See `matrixvectoractivation_rtl.py` for PE-distributed weights
- FINN utilities: `finn.util.data_packing` for data formatting
- Design document: `_artifacts/weight_file_generation_design.md`