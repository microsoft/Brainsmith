# Interface-Based Dataflow Model Toolflow Integration

The *Dataflow Model* is a unified abstraction of HW Kernels that will significantly improve automation and extensibility when integrated into the FINN and Brainsmith toolflows.
1. Allow for a refactor of `HWCustomOp` to standardize the vast majority of its methods, slimming down the subcase to primarily only the interface information.
2. Allow for fully automated `HWCustomOp` generation from RTL implementations, a crucial capability of the HW Kernel Generator (HWKG).

## HW Kernel Generator

When a new RTL implementation is created for a previously unsupported ONNX operator, contributors use the HW Kernel Generator (HWKG) to integrate it into FINN as a HW Kernel. The Dataflow Model is crucial for generating the `HWCustomOp` class for the kernel, which is then used to instantiate the kernel in the ONNX pattern. The proposed workflow of the HWKG is as follows:

### *Step 1: RTL Parser*

The RTL Parser extracts the kernel interfaces, parameters, and pragmas from the RTL implementation, which are then used to create a `HWKernel` object. This process is detailed in [RTL_Parser.md](./RTL_Parser.md), but the information relevant to the Dataflow Model is:
- The number of each interfaces in the kernel, along with their names, datatypes, and width formulas.
- Pragmas that convey compiler information such as:
    - WEIGHT: Marks an input AXI-Stream interface as a weight.
    - DATATYPE: Marks supported datatypes for an interface.
- Module parameters to expose to the compiler and consider for design space exploration.

Some additional pragmas may be necessary to support all features in the `AutoHWCustomOp`. Potenial examples include a "block shape" pragma to define the $bDim$ of an interface if it differs from what would be inferred from the ONNX layout (mainly useful for elementwise and tiled Kernels).

### *Step 2: Generate AutoHWCustomOp*

The HWKG will ingest the `HWKernel` object describing the Kernel's RTL implementation and generate several files to integrate the kernel into FINN. To bridge the gap to the ONNX DSE in FINN, it needs to generate a subclass of `HWCustomOp` and `RTLBackend`. To utilize the Dataflow Model, we will instead generate a subclass of `AutoHWCustomOp` and `AutoRTLBackend`, which are subclasses of `HWCustomOp` and `RTLBackend`, respectively, that rely on the Dataflow Model. The HWKG will use jinja2 templates to generate these subclasses.

## AutoHWCustomOp 

Current subclasses of `HWCustomOp` are manually implemented, but many of methods can be fully standardized in the base class, configured by the parameters of the Dataflow Model subclasses. To minimize disruption to existing HW Kernels, we will first implement `AutoHWCustomOp`, which will be a subclass of `HWCustomOp` but with our refactors and automations. Once all existing kernels are migrated to `AutoHWCustomOp`, we will rename it to `HWCustomOp` and remove the old `HWCustomOp` class. Similarly, we will implement `AutoRTLBackend` as a subclass of `RTLBackend`.

The following methods can be fully standardized in `AutoHWCustomOp`:
- get_input_datatype & get_output_datatype
- get_normal_input_shape & get_normal_output_shape
- get_folded_input_shape & get_folded_output_shape
- get_instream_width & get_outstream_width
- get_exp_cycles
- get_op_and_param_counts
- derive_characteristic_fxns
- generate_params

Additionally, the following new methods will need to be implemented in `AutoHWCustomOp` to support the Dataflow Model:

### 1. Block Chunking

The Dataflow Model utilizes a data hierarchy oriented towards describing Dataflow architectures, while ONNX describes hidden states and weights in a single multi-dimensional layout shape. The layout of these ONNX matrices can vary significantly depending on the model, so the `AutoHWCustomOp` needs a way to map the ONNX data layout of each interface into $num_blocks$ and $bDim$.

Assume that there will be a tool to determine the ONNX layout of each interface input and output of the ONNX node (NCHW, NHCW, NLC, etc.), this will be implemented at at a future date. For Input and Output interfaces, $num_blocks$ and $bDim$ can be determined by the following table (N is always the batch dimension):

| **ONNX Input Shape** | **qDim**      | **num_blocks** | **$bDim$** | **Example model types** |
| ------------- | ---------- | --- | --- | --- |
| \[N, C]       | \[C]       | 1    | C     | CNN (expected)          |
| \[N, C, H, W] | \[C, H, W] | C    | H * W | CNN (expected)          |
| \[N, H, W, C] | \[H, W, C] | H*W  | C     | CNN (inverted)          |
| \[N, L, C]    | \[L, C]    | L    | L     | Transformers (expected) |
| \[N, C, L]    | \[C, L]    | C    | C     | Transformers (inverted) |
| \[N, L, h, d] | \[L, h, d] | L    | h*d   | Transformers MHA        |

For Weight interfaces, 1D weights (e.g. LayerNorm) will by default have $bDim$ equal to the length of the weight and $num_blocks$ equal to 1, while 2D weights (e.g. MVAU) will have $bDim$ equal to the first dimension and $num_blocks$ equal to the second dimension.

Although correct for most cases, there are exceptions where the above rules do not hold (e.g. tiled kernels where bDim is a portion of one or more of dimensions, but not the full dimension). Therefore, this mapping should be overrideable via pragmas in the RTL implementation, defining $bDim$ in terms of parameters exposed to the compiler and linked to variables from the ONNX pattern by the `AutoHWCustomOp` constructor.


### 2. More? 
Consider what other methods are necessary to utilize the Dataflow Model and link it to the ONNX node's parameters and FINN's DSE/parallelism capabilities.