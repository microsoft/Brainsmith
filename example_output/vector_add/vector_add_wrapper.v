`timescale 1ns / 1ps

//=============================================================================
// Auto-generated RTL wrapper for vector_add
// Generated from: example_vector_add.sv
// Template: rtl_wrapper_v2.v.j2 (Phase 3 Enhanced)
// Generation time: 2025-06-12T22:42:07.109220
//
// Phase 2 Features:
// ✅ Validated parameter references in BDIM pragmas
// ✅ Runtime parameter extraction support
// ✅ Enhanced interface metadata with chunking strategies
//=============================================================================

module vector_add_wrapper #(
    // RTL Parameters with Phase 2 validation
parameter PE = 4,parameter VECTOR_SIZE = 1) (
    // Global control interface (required)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // AXI-Stream interfaces with validated BDIM parameters
    
    // input0: INPUT interface
    // Validated BDIM shape: [':', ':']
    // Interface type: INPUT
    input  wire [input0_TDATA_WIDTH-1:0] input0_TDATA,
    input  wire input0_TVALID,
    output wire input0_TREADY,
    
    // input1: INPUT interface
    // Validated BDIM shape: [':', ':']
    // Interface type: INPUT
    input  wire [input1_TDATA_WIDTH-1:0] input1_TDATA,
    input  wire input1_TVALID,
    output wire input1_TREADY,
    
    // output0: OUTPUT interface  
    // Validated BDIM shape: [':', ':']
    // Interface type: OUTPUT
    output wire [output0_TDATA_WIDTH-1:0] output0_TDATA,
    output wire output0_TVALID,
    input  wire output0_TREADY,
    
    // Additional control signals
    input  wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready
);

//=============================================================================
// Parameter Validation (Phase 2 guaranteed valid parameters)
//=============================================================================

// Validation for required parameters (no defaults)
initial begin
    if (VECTOR_SIZE <= 0) begin
        $error("[%s:%0d] Parameter VECTOR_SIZE must be positive, got %0d", 
               `__FILE__, `__LINE__, VECTOR_SIZE);
        $finish;
    end
end

// Validation for whitelisted parameters (with defaults)
initial begin
    if (PE <= 0) begin
        $error("[%s:%0d] Parameter PE must be positive, got %0d (default: 4)", 
               `__FILE__, `__LINE__, PE);
        $finish;
    end
end

// Parameter consistency checks
initial begin
end

//=============================================================================
// Interface Width Calculations (Based on Validated BDIM)
//=============================================================================

// input0 width calculation
localparam input0_ELEMENT_WIDTH = 8;

// BDIM-based width: [':', ':']
localparam input0_PARALLEL_ELEMENTS = 1 * 1;

localparam input0_TDATA_WIDTH = input0_ELEMENT_WIDTH * input0_PARALLEL_ELEMENTS;

// input1 width calculation
localparam input1_ELEMENT_WIDTH = 8;

// BDIM-based width: [':', ':']
localparam input1_PARALLEL_ELEMENTS = 1 * 1;

localparam input1_TDATA_WIDTH = input1_ELEMENT_WIDTH * input1_PARALLEL_ELEMENTS;

// output0 width calculation
localparam output0_ELEMENT_WIDTH = 16;

// BDIM-based width: [':', ':']
localparam output0_PARALLEL_ELEMENTS = 1 * 1;

localparam output0_TDATA_WIDTH = output0_ELEMENT_WIDTH * output0_PARALLEL_ELEMENTS;


//=============================================================================
// Module Instance with Validated Parameters
//=============================================================================

vector_add #(
.PE(PE),
.VECTOR_SIZE(VECTOR_SIZE)
) dut_inst (
    // Global control
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .ap_start(ap_start),
    .ap_done(ap_done),
    .ap_idle(ap_idle),
    .ap_ready(ap_ready),
    
    // Data interfaces
    .input0_TDATA(input0_TDATA),
    .input0_TVALID(input0_TVALID),
    .input0_TREADY(input0_TREADY),
    .input1_TDATA(input1_TDATA),
    .input1_TVALID(input1_TVALID),
    .input1_TREADY(input1_TREADY),
    .output0_TDATA(output0_TDATA),
    .output0_TVALID(output0_TVALID),
    .output0_TREADY(output0_TREADY)
);

//=============================================================================
// Debug and Monitoring (for development/testing)
//=============================================================================

`ifdef DEBUG_VECTOR_ADD

// Parameter values at elaboration time
initial begin
    $display("=== vector_add_wrapper Parameter Values ===");
    $display("PE = %0d", PE);
    $display("VECTOR_SIZE = %0d", VECTOR_SIZE);
    
    $display("=== Interface Widths ===");
    $display("input0_TDATA_WIDTH = %0d", input0_TDATA_WIDTH);
    $display("input1_TDATA_WIDTH = %0d", input1_TDATA_WIDTH);
    $display("output0_TDATA_WIDTH = %0d", output0_TDATA_WIDTH);
    $display("=======================================");
end

// Runtime monitoring
always @(posedge ap_clk) begin
    if (ap_start && ap_ready) begin
        $display("[%0t] vector_add: Starting operation", $time);
    end
    if (ap_done) begin
        $display("[%0t] vector_add: Operation complete", $time);
    end
end

`endif // DEBUG_VECTOR_ADD

//=============================================================================
// Assertions for Interface Protocol Validation
//=============================================================================

`ifdef ASSERT_ON

// input0 AXI-Stream protocol assertions
property input0_tvalid_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    input0_TVALID && !input0_TREADY |=> input0_TVALID;
endproperty

assert property(input0_tvalid_stable) 
else $error("input0_TVALID not stable when TREADY low");

property input0_tdata_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    input0_TVALID && !input0_TREADY |=> $stable(input0_TDATA);
endproperty

assert property(input0_tdata_stable) 
else $error("input0_TDATA changed when TVALID high and TREADY low");

// input1 AXI-Stream protocol assertions
property input1_tvalid_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    input1_TVALID && !input1_TREADY |=> input1_TVALID;
endproperty

assert property(input1_tvalid_stable) 
else $error("input1_TVALID not stable when TREADY low");

property input1_tdata_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    input1_TVALID && !input1_TREADY |=> $stable(input1_TDATA);
endproperty

assert property(input1_tdata_stable) 
else $error("input1_TDATA changed when TVALID high and TREADY low");

// output0 AXI-Stream protocol assertions
property output0_tvalid_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    output0_TVALID && !output0_TREADY |=> output0_TVALID;
endproperty

assert property(output0_tvalid_stable) 
else $error("output0_TVALID not stable when TREADY low");

property output0_tdata_stable;
    @(posedge ap_clk) disable iff (!ap_rst_n)
    output0_TVALID && !output0_TREADY |=> $stable(output0_TDATA);
endproperty

assert property(output0_tdata_stable) 
else $error("output0_TDATA changed when TVALID high and TREADY low");


`endif // ASSERT_ON

endmodule

//=============================================================================
// End of vector_add_wrapper
// Template: rtl_wrapper_v2.v.j2 (Phase 3 Enhanced)
//=============================================================================