# Hardware Kernel Generator (HKG) Component – RTL Parser

## Objective
The RTL Parser is a key component for the larger HKG project. Its goal is to parse an Abstract Syntax Tree from a SystemVerilog file and identify, extract, and format the key information needed by the Kernel Generator. 

## Requirements
### 1. *Inputs*
- *Manual implementation*: SystemVerilog implementation of the target HW Kernel. Example: @https://raw.githubusercontent.com/Xilinx/finn/refs/heads/dev/finn-rtllib/thresholding/hdl/thresholding_axi.sv

### 2. *Data to Extract*
#### *Module Parameters*
The parameters to the Top Module.
- name: Parameter identifier
- type: Datatype of the parameter
Criteria:
- Ignore local parameters

#### *Ports*
The input/output ports to the Top Module.
- name: Port identifier
- direction: "input" or "output"
- width: Bit width
Criteria:
- Any bitwidths expressed in constant expressions should be preserved, *not* simplified or calculated

#### *Pragmas*
Comments formatted like this:
```
// @brainsmith <pragma> <input>
```
- @brainsmith: the flag to alert the parser to the pragma
- <pragma>: identifies the type of pragma from a list of valid pragma names
- <input>: one or more positional inputs to processed by the pragma function. Space separated.
Therefore the data to extract is:
- pragma: the type of pragma
- inputs: list of inputs

### 3. *Data Process*
#### *Kernel Parameters*
Module Parameters will be reformatted to Kernel Parameters. This will be implemented in the future, just create placeholder code.

#### *Interfaces*
Ports will be grouped into Interfaces This will be implemented in the future, just create placeholder code.

#### *Compiler Flags*
Compiler flags will be inferred from pragma data. This will be implemented in the future, just create placeholder code.

## *Implementation Details*
### 1. *Technology Stack*
- *Parser*: Use py-tree-sitter for parsing 
    - Documentation: @/py-tree-sitter/docs
    - Example 1: @/py-tree-sitter/examples/usage.py
    - Example 2: @/py-tree-sitter/examples/walk_tree.py
    - The grammar for SystemVerilog: @/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/sv.so

### 4. *Environment*
- Implement this tool at the path: `/brainsmith/tools/hw_kernel_gen`
