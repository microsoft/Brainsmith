# Inputs

## HWKernel (From RTL Parser)
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    interfaces: Dict[str, Interface] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

## Python Data

### Infer Transformation Data
- This is largely placeholder until we actually explore using ONNX script for this

# Outputs
The various functions we need to generate, where, and what data does it need:

## Directly map
- get_nodeattr_types - All directly mapped parameters from the RTL Parser (e.g. all module parameters that were NOT tagged in a dervied_param pragma).

## Auto-populates based on interfaces
- get_input_datatype(ind) - Implement based on the number of input AXI-Stream interfaces
- get_output_datatype(ind) - Implement based on the number of output AXI-Stream interfaces
- get_verilog_top_module_intf_names - Implement based on the interfaces present and their names

## Give some simple way to tie SIMD/PE to diff dimensions, or force standardize? Then can derive based on interfaces:
- get_normal_input_shape(ind) - Implement based on input signal width and datatype
- get_normal_output_shape(ind) - Implement based on output signal width and datatype
- get_folded_input_shape(ind) - Implement based on get_normal_input_shape and SIMD/PE
- get_folded_output_shape(ind) - Implement based on get_normal_output_shape and SIMD/PE
- get_instream_width(ind) - Implement based on get_input_datatype and SIMD/PE
- get_outstream_width(ind) - Implement based on get_output_datatype and SIMD/PE

## User implements, to automate in the future. Do NOT implement in template or HKG
- generate_params
- get_exp_cycles
- get_op_and_param_counts
- bram_efficiency_estimation
- uram_efficiency_estimation
- bram_estimation
- uram_estimation
- lut_estimation
- dsp_estimation
- derive_characteristic_fxns - (Optional if not 1in-1out, pending FIFO sizing refactor)
