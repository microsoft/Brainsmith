# Inputs

## HWKernel (From RTL Parser)
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    interfaces: Dict[str, Interface] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

Missing pragmas
- AXI-Stream is a weight

## Python Data

### Infer Transformation Data
- This is largely placeholder until we actually explore using ONNX script for this




# Outputs
The various functions we need to generate, where, and what data does it need?

## get_nodeattr_types
- All directly mapped parameters from the RTL Parser (e.g. all module parameters that were NOT tagged in a dervied_param pragma)

## Auto-populates based on interfaces
- get_input_datatype(ind) - Implement like so based on
- get_output_datatype(ind)
- get_verilog_top_module_intf_names

## Give some simple way to tie SIMD/PE to diff dimensions, or force standardize? Then can derive based on interfaces:
- get_exp_cycles
- get_op_and_param_counts
- get_normal_input_shape(ind) - from input signal width?
- get_normal_output_shape(ind) - from input signal width?
- get_folded_input_shape(ind) - [c, h, w] -> ? /PE /SIMD ?
- get_folded_output_shape(ind) - [c, h, w] -> ?
- get_instream_width(ind) - bitwidth*... When *PE when *SIMD?
- get_outstream_width(ind) - bitwidth*... When *PE when *SIMD?

## Standardize outside of HWCOp
- generate_params - Can we standardize this? Seem similar between functions

## User implements, to automate in the future
- bram_efficiency_estimation
- uram_efficiency_estimation
- bram_estimation
- uram_estimation
- lut_estimation
- dsp_estimation
- derive_characteristic_fxns - (Optional if not 1in-1out, pending FIFO sizing refactor)
