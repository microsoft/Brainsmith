// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
`timescale 1ns / 1ps

`define VERILATOR

module tb_top;

    // SIM
    parameter CLK_PERIOD = 10ns;
    parameter RST_PERIOD = 2.5 * CLK_PERIOD;
    parameter AST_PERIOD = 4.5 * CLK_PERIOD;
    parameter TT = 2ns;
    parameter TA = 1ns;

    logic clk = 1'b1;
    logic clk_dp = 1'b1;
    logic resetn = 1'b0;

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;
    always #(CLK_PERIOD/4) clk_dp = ~clk_dp;
    
    // Reset Generation
    initial begin
        resetn = 1'b0;
        #(RST_PERIOD) resetn = 1'b1;
    end

    // Control and status
    logic [1:0] status_int;
    logic [63:0] interval;
    logic [63:0] latency;
    logic [15:0] checksum;
    logic [31:0] n_beats_in;
    logic [15:0] n_runs_in;
    logic [31:0] n_beats_out;
    logic [15:0] n_runs_out;
    logic [31:0] seed;
    logic start;
    logic ready;
    
    // Data
    parameter ILEN_BITS = 32;
    parameter OLEN_BITS = 32;
    
    AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) finn_in ();
    AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) finn_out ();
    AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) finn_out_t ();
    
    logic [31:0] cnt_out;

    //
    // DUT
    //
    
    instrumentation_top inst_instrumentation (
        .aclk(clk),
        .aresetn(resetn),
        
        .m_finn_in(finn_in),
        .s_finn_out(finn_out),
        
        .s_n_beats_in(n_beats_in),
        .s_n_runs_in(n_runs_in),
        .s_n_beats_out(n_beats_out),
        .s_n_runs_out(n_runs_out),
        .s_seed(seed),
        .s_start(start),
        
        .m_latency_out(latency),
        .m_interval_out(interval),
        .m_checksum_out(checksum),
        .m_ready(ready)
    );
    
     axis_data_fifo_0 inst_data_fifo (
        .s_axis_aclk(clk),
        .s_axis_aresetn(resetn),
        
        .s_axis_tvalid(finn_in.tvalid),
        .s_axis_tready(finn_in.tready),
        .s_axis_tdata (finn_in.tdata),
        .m_axis_tvalid(finn_out_t.tvalid),
        .m_axis_tready(finn_out_t.tready),
        .m_axis_tdata (finn_out_t.tdata)
    );
    
    axis_data_fifo_0 inst_data_fifo_2 (
        .s_axis_aclk(clk),
        .s_axis_aresetn(resetn),
        
        .s_axis_tvalid(finn_out_t.tvalid),
        .s_axis_tready(finn_out_t.tready),
        .s_axis_tdata (finn_out_t.tdata),
        .m_axis_tvalid(finn_out.tvalid),
        .m_axis_tready(finn_out.tready),
        .m_axis_tdata (finn_out.tdata)
    );
    
    always_ff @(posedge clk) begin
        if(~resetn) begin
            cnt_out <= 0;
        end
        else begin
            cnt_out <= (finn_in.tvalid & finn_in.tready) ? cnt_out + 1 : cnt_out;
        end
    end
    
    
    // 
    initial begin
        n_beats_in = 12288;
        n_runs_in = 1;
        n_beats_out = 12288;
        n_runs_out = 1;
        seed = 23;
        start = 1'b0;

        #(100*CLK_PERIOD)
        start = 1'b1;
    end

    // Dump
    initial begin
        $dumpfile("dump.vcd"); $dumpvars;
    end

endmodule