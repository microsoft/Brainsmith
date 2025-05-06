# Golden Verilog Wrapper for thresholding_axi
# Manually generated based on golden_thresholding_hwkernel.py and template logic.

# Placeholder - Needs actual template logic applied to golden_thresholding_hwkernel.py
# This is a *very* basic structure based on the example and golden data.
# The actual Jinja template will produce the final correct version.

module {{ kernel.name }}_WRAPPER_NAME$ #(
    // Parameters from golden_thresholding_hwkernel.py
    // Should be untyped and use template names as defaults
    parameter N = $N$,
    parameter WI = $WI$,
    parameter WT = $WT$,
    parameter C = $C$,
    parameter PE = $PE$,
    parameter SIGNED = $SIGNED$,
    parameter FPARG = $FPARG$,
    parameter BIAS = $BIAS$,
    parameter THRESHOLDS_PATH = $THRESHOLDS_PATH$,
    parameter USE_AXILITE = $USE_AXILITE$,
    parameter DEPTH_TRIGGER_URAM = $DEPTH_TRIGGER_URAM$,
    parameter DEPTH_TRIGGER_BRAM = $DEPTH_TRIGGER_BRAM$,
    parameter DEEP_PIPELINE = $DEEP_PIPELINE$,

    // Derived Parameters (These might be calculated differently in the final template)
    // Keep localparams as they are not substituted directly by top-level params
    localparam int unsigned ADDR_BITS = $clog2(C/PE) + $clog2(PE) + N + 2, // Example calculation
    localparam int unsigned O_BITS = (BIAS >= 0) ? $clog2(2**N+BIAS) : 1+$clog2((-BIAS >= 2**(N-1)) ? -BIAS : 2**N+BIAS) // Example calculation
)(
    // Global Control Interface
    input  logic ap_clk,
    input  logic ap_rst_n,

    // AXI-Lite Interface (s_axilite)
    // Write Address Channel
    input  logic                  s_axilite_AWVALID,
    output logic                  s_axilite_AWREADY,
    input  logic [ADDR_BITS-1:0]  s_axilite_AWADDR,
    // Write Data Channel
    input  logic                  s_axilite_WVALID,
    output logic                  s_axilite_WREADY,
    input  logic [31:0]           s_axilite_WDATA,
    input  logic [3:0]            s_axilite_WSTRB,
    // Write Response Channel
    output logic                  s_axilite_BVALID,
    input  logic                  s_axilite_BREADY,
    output logic [1:0]            s_axilite_BRESP,
    // Read Address Channel
    input  logic                  s_axilite_ARVALID,
    output logic                  s_axilite_ARREADY,
    input  logic [ADDR_BITS-1:0]  s_axilite_ARADDR,
    // Read Data Channel
    output logic                  s_axilite_RVALID,
    input  logic                  s_axilite_RREADY,
    output logic [31:0]           s_axilite_RDATA,
    output logic [1:0]            s_axilite_RRESP,

    // AXI-Stream Input Interface (s_axis)
    output logic                  s_axis_tready,
    input  logic                  s_axis_tvalid,
    input  logic [((PE*WI+7)/8)*8-1:0] s_axis_tdata,

    // AXI-Stream Output Interface (m_axis)
    input  logic                  m_axis_tready,
    output logic                  m_axis_tvalid,
    output logic [((PE*O_BITS+7)/8)*8-1:0] m_axis_tdata

    // Unassigned ports would go here if any
);

    // Instantiate the original kernel
    thresholding_axi #(
        .N                  (N),
        .WI                 (WI),
        .WT                 (WT),
        .C                  (C),
        .PE                 (PE),
        .SIGNED             (SIGNED),
        .FPARG              (FPARG),
        .BIAS               (BIAS),
        .THRESHOLDS_PATH    (THRESHOLDS_PATH),
        .USE_AXILITE        (USE_AXILITE),
        .DEPTH_TRIGGER_URAM (DEPTH_TRIGGER_URAM),
        .DEPTH_TRIGGER_BRAM (DEPTH_TRIGGER_BRAM),
        .DEEP_PIPELINE      (DEEP_PIPELINE)
        // Pass other parameters if needed by the kernel but not exposed by wrapper
    ) inst_thresholding_axi (
        // Connect Global Control
        .ap_clk             (ap_clk),
        .ap_rst_n           (ap_rst_n),

        // Connect AXI-Lite (s_axilite)
        .s_axilite_AWVALID  (s_axilite_AWVALID),
        .s_axilite_AWREADY  (s_axilite_AWREADY),
        .s_axilite_AWADDR   (s_axilite_AWADDR),
        .s_axilite_WVALID   (s_axilite_WVALID),
        .s_axilite_WREADY   (s_axilite_WREADY),
        .s_axilite_WDATA    (s_axilite_WDATA),
        .s_axilite_WSTRB    (s_axilite_WSTRB),
        .s_axilite_BVALID   (s_axilite_BVALID),
        .s_axilite_BREADY   (s_axilite_BREADY),
        .s_axilite_BRESP    (s_axilite_BRESP),
        .s_axilite_ARVALID  (s_axilite_ARVALID),
        .s_axilite_ARREADY  (s_axilite_ARREADY),
        .s_axilite_ARADDR   (s_axilite_ARADDR),
        .s_axilite_RVALID   (s_axilite_RVALID),
        .s_axilite_RREADY   (s_axilite_RREADY),
        .s_axilite_RDATA    (s_axilite_RDATA),
        .s_axilite_RRESP    (s_axilite_RRESP),

        // Connect AXI-Stream Input (s_axis)
        .s_axis_tready      (s_axis_tready),
        .s_axis_tvalid      (s_axis_tvalid),
        .s_axis_tdata       (s_axis_tdata),

        // Connect AXI-Stream Output (m_axis)
        .m_axis_tready      (m_axis_tready),
        .m_axis_tvalid      (m_axis_tvalid),
        .m_axis_tdata       (m_axis_tdata)

        // Connect unassigned ports if any
    );

endmodule : {{ kernel.name }}_WRAPPER_NAME$
