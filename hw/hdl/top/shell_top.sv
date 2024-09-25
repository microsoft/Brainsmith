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

// ================-----------------------------------------------------------------
// ================-----------------------------------------------------------------
// TOP LEVEL - v80
// ================-----------------------------------------------------------------
// ================-----------------------------------------------------------------

import cTypes::*;

module shell_top (
);

    // Pipeline stages  
    parameter N_STGS = N_LAYERS*2+1;

    // Params
    parameter ILEN_BITS = SIMD_P0 * ACTIVATION_WIDTH;
    parameter OLEN_BITS = PE_P2 * ACTIVATION_WIDTH;

    // Clocks
    wire ps_clk;
    wire [0:0] ps_resetn;
    wire aclk;
    wire aclk_dp;
    wire [0:0] aresetn;

    // Control and status (32-bit)
    logic ready_in;
    logic ready_out;
    logic [63:0] interval;
    logic [63:0] latency;
    logic [15:0] checksum;
    logic overflow;
    logic start;
    logic [31:0] n_beats_in;
    logic [15:0] n_runs_in_pow2;
    logic [31:0] n_beats_out;
    logic [15:0] n_runs_out_pow2;
    logic [31:0] lfsr_seed;
    
    AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) finn_in ();
    AXI4S #(.AXI4S_DATA_BITS(OLEN_BITS)) finn_out ();
    AXI4S #(.AXI4S_DATA_BITS(ILEN_BITS)) finn_s [N_STGS] ();
 
    // ================-----------------------------------------------------------------
    // CIPS 
    // ================-----------------------------------------------------------------

    design_static inst_cips (
        // Clocks
        .ps_clk(ps_clk),
        .ps_resetn(ps_resetn),
        .aclk(aclk),
        .aclk_dp(aclk_dp),
        .aresetn(aresetn)
    );

    // ================-----------------------------------------------------------------
    // VIOs control
    // ================-----------------------------------------------------------------

    vio_top inst_vio_top (
        .clk(aclk),
        
        .probe_in0(ready_in),
        .probe_in1(ready_out),
        .probe_in2(interval), // 64
        .probe_in3(latency), // 64
        .probe_in4(checksum), // 16
        .probe_in5(overflow),
        
        .probe_out0(start), 
        .probe_out1(n_beats_in), // 32
        .probe_out2(n_runs_in_pow2), // 16
        .probe_out3(n_beats_out), // 32
        .probe_out4(n_runs_out_pow2), // 16
        .probe_out5(lfsr_seed) // 32
    );

    // ================-----------------------------------------------------------------
    // Instrumentation 
    // ================-----------------------------------------------------------------

    instrumentation_producer inst_producer (
        .aclk(aclk),
        .aresetn(aresetn),

        .m_finn_in(finn_in),

        .s_n_beats_in(n_beats_in),
        .s_n_runs_in(n_runs_in_pow2),
        .s_seed(lfsr_seed),
        .s_start(start),
        .m_ready(ready_in)
    );

    instrumentation_consumer inst_consumer (
        .aclk(aclk),
        .aresetn(aresetn),

        .s_finn_out(finn_out),

        .s_n_beats_out(n_beats_out),
        .s_n_runs_out(n_runs_out_pow2),
        .s_start(start),
        
        .m_latency_out(latency),
        .m_interval_out(interval),
        .m_checksum_out(checksum),
        .m_overflow_out(overflow),
        .m_ready(ready_out)
    );

    // ================-----------------------------------------------------------------
    // FINN wrapper 
    // ================-----------------------------------------------------------------
    
    // Buffer in
    dwc_buff_top #(.I_BITS(ILEN_BITS), .O_BITS(ILEN_BITS)) inst_dwc_buff_finn_in (.ap_clk(aclk), .ap_rst_n(aresetn), .s_axis(finn_in), .m_axis(finn_s[0]));
    
    // P0-P1-P2 test
    for(genvar i = 0; i < N_LAYERS; i++) begin
        p012_top #(
            // P0
            .SIMD_P0(SIMD_P0),
            .PE_P0(PE_P0),

            .SIMD_P0_MM(SIMD_P0_MM),
            .PE_P0_MM(PE_P0_MM),
            .PE_P0_THR(PE_P0_THR),

            .INIT_FILE_P0_0(INIT_FILE_P0_0),
            .INIT_FILE_P0_1(INIT_FILE_P0_1),
            .INIT_FILE_P0_2(INIT_FILE_P0_2),
            
            // P1
            .SIMD_P1(SIMD_P1),
            .PE_P1(PE_P1),
            
            .SIMD_P1_MM_0(SIMD_P1_MM_0),
            .PE_P1_MM_0(PE_P1_MM_0),
            .PE_P1_THR_0(PE_P1_THR_0),

            .SIMD_P1_MM_1(SIMD_P1_MM_1),
            .PE_P1_MM_1(PE_P1_MM_1),
            .PE_P1_THR_1(PE_P1_THR_1),

            .THRESHOLDS_PATH_P1_0(THRESHOLDS_PATH_P1_0),
            .THRESHOLDS_PATH_P1_1(THRESHOLDS_PATH_P1_1),

            // P2
            .SIMD_P2_0(SIMD_P2_0),
            .SIMD_P2_1(SIMD_P2_1),
            .PE_P2(PE_P2),

            .SIMD_P2_MM_0(SIMD_P2_MM_0),
            .PE_P2_MM_0(PE_P2_MM_0),
            .PE_P2_THR_0(PE_P2_THR_0),

            .SIMD_P2_MM_1(SIMD_P2_MM_1),
            .PE_P2_MM_1(PE_P2_MM_1),
            .PE_P2_THR_1(PE_P2_THR_1),

            .SIMD_P2_MM_2(SIMD_P2_MM_2),
            .PE_P2_MM_2(PE_P2_MM_2),
            .PE_P2_THR_2(PE_P2_THR_2),

            .INIT_FILE_P2_0(INIT_FILE_P2_0),
            .INIT_FILE_P2_1(INIT_FILE_P2_1),
            .INIT_FILE_P2_2(INIT_FILE_P2_2),

            .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
            .PUMPED_COMPUTE(0),
            .MM_KERNEL(MM_KERNEL)
        ) inst_p012_top (
            .ap_clk(aclk),
            .ap_clk2x(aclk_dp),
            .ap_rst_n(aresetn),

            .s_axis(finn_s[i*2+0]),
            .m_axis(finn_s[i*2+1])
        );

        dwc_buff_top #(.I_BITS(OLEN_BITS), .O_BITS(OLEN_BITS)) inst_dwc_buff_finn_s (.ap_clk(aclk), .ap_rst_n(aresetn), .s_axis(finn_s[i*2+1]), .m_axis(finn_s[i*2+2]));
    end

    // Assign
    assign finn_out.tvalid = finn_s[N_STGS-1].tvalid;
    assign finn_out.tdata  = finn_s[N_STGS-1].tdata;
    assign finn_s[N_STGS-1].tready = finn_out.tready;
    
endmodule