# RTL Parser: Core Design Axioms

## 1. Parser Pipeline
```
SystemVerilog → AST → Interfaces → Templates
```
All RTL parsing follows this immutable sequence: Tree-sitter AST generation, AXI interface extraction, and Jinja2 template generation.

## 2. AXI-Only Interface Model
Every SystemVerilog module must contain exactly three interface types:
- **GLOBAL_CONTROL**: Clock/reset (ap_clk, ap_rst_n required)
- **AXI_STREAM**: Dataflow (TDATA/TVALID/TREADY required, TLAST optional)
- **AXI_LITE**: Configuration (read/write variants, optional)

## 3. Port Grouping by Pattern Matching
Ports become interfaces via regex-based prefix recognition. All ports must belong to exactly one interface - no orphaned ports allowed.

## 4. Pragma-Driven Metadata
`// @brainsmith <type> <args>` comments provide essential metadata that cannot be inferred from RTL structure alone:
- **`TOP_MODULE <module_name>`** - Select target module in multi-module files
- **`DATATYPE <interface> <types> <min_bits> <max_bits>`** - Define datatype constraints (INT,UINT,FLOAT,FIXED)
- **`BDIM <interface> <chunk_index> [<chunk_sizes>]`** - Override block dimensions and chunking strategies
- **`DERIVED_PARAMETER <function> <param1> [param2...]`** - Link RTL parameters to Python derivation functions
- **`WEIGHT <interface1> [interface2...]`** - Mark interfaces as weight inputs

## 5. Module Parameters as Template Variables
Module parameters (excluding localparams) become FINN node attributes and Jinja2 template variables:
- **Direct mapping**: RTL parameter → FINN attribute → template variable
- **Expression preservation**: "DATA_WIDTH-1:0" becomes template expressions, never evaluated
- **Type information**: SystemVerilog types preserved for template generation
- **Derivation support**: Parameters can be computed via Python functions using `derived_param` pragma

## 6. Expression Preservation
Parameter expressions (e.g., "DATA_WIDTH-1:0") are never evaluated - they become template variables for runtime binding in FINN.

## 7. Dual Input Support
Parser must handle both file paths and direct SystemVerilog strings through a unified `parse()` API.

## 8. Immutable Data Structures
Once parsed and validated, HWKernel objects are immutable. This enables safe concurrent template generation.