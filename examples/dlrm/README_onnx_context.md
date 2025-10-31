# ONNX Context Export Utility

## Overview

`export_onnx_context.py` is a utility that converts ONNX models into a comprehensive, human-readable format suitable for providing context to AI assistants or for documentation purposes.

## Features

The utility generates a structured context file containing:

1. **Model Metadata** - IR version, opset, producer information
2. **Graph Inputs/Outputs** - Names, types, and shapes
3. **Initializers Summary** - List of all weights and constants with sizes
4. **Operation Statistics** - Count of each operation type in the model
5. **Node Connectivity** - Which nodes consume each input
6. **Complete Graph Structure** - Full ONNX Script IR representation

## Usage

### Basic Usage

```bash
python export_onnx_context.py model.onnx
```

This creates `model_context.txt` in the same directory.

### Specify Output File

```bash
python export_onnx_context.py model.onnx -o custom_output.txt
```

### Verbose Mode

```bash
python export_onnx_context.py model.onnx -v
```

Shows progress and statistics during export.

### Command Line Options

```
positional arguments:
  onnx_file             Path to the ONNX model file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: {model_name}_context.txt)
  --max-array-size MAX_ARRAY_SIZE
                        Maximum number of array elements to display (default: 20)
  -v, --verbose         Print verbose output
```

## Example Output Structure

```
================================================================================
ONNX Model Context: dlrm_s_pytorch.onnx
================================================================================

# Model Metadata
--------------------------------------------------------------------------------
IR Version:      10
Opset Imports:   {'': 18}
...

# Graph Inputs
--------------------------------------------------------------------------------
1. dense_x
   Type: FLOAT
   Shape: [s32,4]
...

# Operation Statistics
--------------------------------------------------------------------------------
Total Nodes: 63

Operation Type Distribution:
  Concat                         :    6
  Gather                         :    6
...

# Complete Graph Structure (ONNX Script IR Format)
================================================================================
<Full ONNX Script IR representation of the entire graph>
...
```

## Use Cases

1. **Providing Context to AI Assistants** - Share the generated `.txt` file to give comprehensive model information
2. **Documentation** - Create readable documentation of model architecture
3. **Debugging** - Understand model structure and data flow
4. **Code Review** - Share model structure without needing ONNX visualization tools

## Tips

- The context file is plain text and can be easily read with any text editor
- For very large models, the file may be large; consider using `head` or `less` to view sections
- The ONNX Script IR format shows exact operation types, attributes, and connections
- Use verbose mode (`-v`) to see file size and node count statistics

## Integration with AI Assistants

When working with an AI assistant on ONNX models:

1. Export the model to context format:
   ```bash
   python export_onnx_context.py your_model.onnx -v
   ```

2. The assistant can then read the generated `your_model_context.txt` file to understand:
   - Model architecture
   - Input/output specifications
   - Operation types and counts
   - Complete computational graph

This provides much better context than trying to describe the model verbally!
