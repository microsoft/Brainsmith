\
# Golden HWKernel object for thresholding_axi.sv
# Manually generated based on analysis of the source file.

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    HWKernel, Parameter, Port, Direction, InterfaceType, Interface, PortGroup, ValidationResult # Added ValidationResult
)

# Define the expected HWKernel instance
golden_kernel = HWKernel(
    name="thresholding_axi",
    parameters=[
        Parameter(name="N", param_type="int unsigned", default_value=None),
        Parameter(name="WI", param_type="int unsigned", default_value=None),
        Parameter(name="WT", param_type="int unsigned", default_value=None),
        Parameter(name="C", param_type="int unsigned", default_value="1"),
        Parameter(name="PE", param_type="int unsigned", default_value="1"),
        Parameter(name="SIGNED", param_type="bit", default_value="1"),
        Parameter(name="FPARG", param_type="bit", default_value="0"),
        Parameter(name="BIAS", param_type="int", default_value="0"),
        Parameter(name="THRESHOLDS_PATH", param_type="parameter", default_value='""'), # Note: type 'parameter' is common for strings
        Parameter(name="USE_AXILITE", param_type="bit", default_value=None),
        Parameter(name="DEPTH_TRIGGER_URAM", param_type="int unsigned", default_value="0"),
        Parameter(name="DEPTH_TRIGGER_BRAM", param_type="int unsigned", default_value="0"),
        Parameter(name="DEEP_PIPELINE", param_type="bit", default_value="0"),
        # Localparams are typically not extracted as top-level parameters for the wrapper
        # Parameter(name="CF", param_type="localparam int unsigned", default_value="C/PE"),
        # Parameter(name="ADDR_BITS", param_type="localparam int unsigned", default_value="$clog2(CF) + $clog2(PE) + N + 2"),
        # Parameter(name="O_BITS", param_type="localparam int unsigned", default_value="BIAS >= 0? $clog2(2**N+BIAS) : 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)"),
    ],
    interfaces={
        # Corrected Interface instantiation: use 'name', 'type', add 'validation_result'
        "global_ctrl": Interface(
            name="global_ctrl",
            type=InterfaceType.GLOBAL_CONTROL,
            ports={ # Changed ports list to dict keyed by port name
                "ap_clk": Port(name="ap_clk", direction=Direction.INPUT, width="1"),
                "ap_rst_n": Port(name="ap_rst_n", direction=Direction.INPUT, width="1"),
            },
            validation_result=ValidationResult(valid=True, message="Golden Global Control") # Added placeholder
        ),
        "s_axilite": Interface(
            name="s_axilite",
            type=InterfaceType.AXI_LITE,
            ports={ # Changed ports list to dict keyed by port name
                # Write Address Channel
                "s_axilite_AWVALID": Port(name="s_axilite_AWVALID", direction=Direction.INPUT, width="1"),
                "s_axilite_AWREADY": Port(name="s_axilite_AWREADY", direction=Direction.OUTPUT, width="1"),
                "s_axilite_AWADDR": Port(name="s_axilite_AWADDR", direction=Direction.INPUT, width="ADDR_BITS-1:0"), # Width uses parameter
                # Write Data Channel
                "s_axilite_WVALID": Port(name="s_axilite_WVALID", direction=Direction.INPUT, width="1"),
                "s_axilite_WREADY": Port(name="s_axilite_WREADY", direction=Direction.OUTPUT, width="1"),
                "s_axilite_WDATA": Port(name="s_axilite_WDATA", direction=Direction.INPUT, width="31:0"),
                "s_axilite_WSTRB": Port(name="s_axilite_WSTRB", direction=Direction.INPUT, width="3:0"),
                # Write Response Channel
                "s_axilite_BVALID": Port(name="s_axilite_BVALID", direction=Direction.OUTPUT, width="1"),
                "s_axilite_BREADY": Port(name="s_axilite_BREADY", direction=Direction.INPUT, width="1"),
                "s_axilite_BRESP": Port(name="s_axilite_BRESP", direction=Direction.OUTPUT, width="1:0"),
                # Read Address Channel
                "s_axilite_ARVALID": Port(name="s_axilite_ARVALID", direction=Direction.INPUT, width="1"),
                "s_axilite_ARREADY": Port(name="s_axilite_ARREADY", direction=Direction.OUTPUT, width="1"),
                "s_axilite_ARADDR": Port(name="s_axilite_ARADDR", direction=Direction.INPUT, width="ADDR_BITS-1:0"),
                # Read Data Channel
                "s_axilite_RVALID": Port(name="s_axilite_RVALID", direction=Direction.OUTPUT, width="1"),
                "s_axilite_RREADY": Port(name="s_axilite_RREADY", direction=Direction.INPUT, width="1"),
                "s_axilite_RDATA": Port(name="s_axilite_RDATA", direction=Direction.OUTPUT, width="31:0"),
                "s_axilite_RRESP": Port(name="s_axilite_RRESP", direction=Direction.OUTPUT, width="1:0"),
            },
            validation_result=ValidationResult(valid=True, message="Golden AXI-Lite Slave") # Added placeholder
        ),
        "s_axis": Interface(
            name="s_axis",
            type=InterfaceType.AXI_STREAM,
            ports={ # Changed ports list to dict keyed by port name
                "s_axis_tready": Port(name="s_axis_tready", direction=Direction.OUTPUT, width="1"),
                "s_axis_tvalid": Port(name="s_axis_tvalid", direction=Direction.INPUT, width="1"),
                "s_axis_tdata": Port(name="s_axis_tdata", direction=Direction.INPUT, width="((PE*WI+7)/8)*8-1:0"),
            },
            validation_result=ValidationResult(valid=True, message="Golden AXI-Stream Slave") # Added placeholder
        ),
        "m_axis": Interface(
            name="m_axis",
            type=InterfaceType.AXI_STREAM,
            ports={ # Changed ports list to dict keyed by port name
                "m_axis_tready": Port(name="m_axis_tready", direction=Direction.INPUT, width="1"),
                "m_axis_tvalid": Port(name="m_axis_tvalid", direction=Direction.OUTPUT, width="1"),
                "m_axis_tdata": Port(name="m_axis_tdata", direction=Direction.OUTPUT, width="((PE*O_BITS+7)/8)*8-1:0"),
            },
            validation_result=ValidationResult(valid=True, message="Golden AXI-Stream Master") # Added placeholder
        ),
    },
)

# Helper function to get the golden kernel if needed by tests
def get_golden_kernel():
    return golden_kernel

