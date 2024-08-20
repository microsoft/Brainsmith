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

import cTypes::*;

module partition_0_mm #(
    parameter integer                               MH = 384,
    parameter integer                               MW = 384,
    parameter integer                               SIMD_P0 = 4,
    parameter integer                               PE_P0 = 4,
    parameter integer                               SIMD_MM_P0 = 48,
    parameter integer                               PE_MM_P0 = 32,
    parameter integer                               ACTIVATION_WIDTH = 8,
    parameter integer                               PUMPED_COMPUTE = 1
) (
    input  logic                                    ap_clk,
    input  logic                                    ap_clk2x,
    input  logic                                    ap_rst_n,
    
    AXI4S.s                                         s_axis_0,
    AXI4S.m                                         m_axis_0,
    AXI4S.m                                         m_axis_1,
    AXI4S.m                                         m_axis_2
);

AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_s0_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_s0_1 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_s0_2 ();

AXI4S #(.AXI4S_DATA_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) axis_s1_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) axis_s1_1 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) axis_s1_2 ();

AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_s2_0 ();
AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_s2_1 ();
AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_s2_2 ();

// Splitter
dup_wrapper_p0_0 inst_splitter (
    .ap_clk             (ap_clk),
    .ap_rst_n           (ap_rst_n),

    .src_TVALID         (s_axis_0.tvalid),
    .src_TREADY         (s_axis_0.tready),
    .src_TDATA          (s_axis_0.tdata),

    .dst_0_TVALID       (axis_s0_0.tvalid),
    .dst_0_TREADY       (axis_s0_0.tready),
    .dst_0_TDATA        (axis_s0_0.tdata),

    .dst_1_TVALID       (axis_s0_1.tvalid),
    .dst_1_TREADY       (axis_s0_1.tready),
    .dst_1_TDATA        (axis_s0_1.tdata),

    .dst_2_TVALID       (axis_s0_2.tvalid),
    .dst_2_TREADY       (axis_s0_2.tready),
    .dst_2_TDATA        (axis_s0_2.tdata)
);

dwc_buff_top #(.I_BITS(SIMD_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s0_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_0), .m_axis(axis_s1_0));
dwc_buff_top #(.I_BITS(SIMD_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s0_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_1), .m_axis(axis_s1_1));
dwc_buff_top #(.I_BITS(SIMD_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_MM_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s0_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_2), .m_axis(axis_s1_2));

// MatMuls
mm_matmul_sta #(
    .MH(MH),
    .MW(MW),
    .PE(PE_MM_P0),
    .SIMD(SIMD_MM_P0),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .THRESHOLDS_PATH(THRESHOLDS_PATH_P0_0),
    .INIT_FILE(INIT_FILE_P0_0)
) inst_matmul_0 (
    .ap_clk(ap_clk),
    .ap_clk2x(ap_clk2x),
    .ap_rst_n(ap_rst_n),

    .s_axis_a_tdata (axis_s1_0.tdata),
    .s_axis_a_tvalid(axis_s1_0.tvalid),
    .s_axis_a_tready(axis_s1_0.tready),

    .m_axis_c_tdata (axis_s2_0.tdata),
    .m_axis_c_tvalid(axis_s2_0.tvalid),
    .m_axis_c_tready(axis_s2_0.tready)
);

mm_matmul_sta #(
    .MH(MH),
    .MW(MW),
    .PE(PE_MM_P0),
    .SIMD(SIMD_MM_P0),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .THRESHOLDS_PATH(THRESHOLDS_PATH_P0_1),
    .INIT_FILE(INIT_FILE_P0_1)
) inst_matmul_1 (
    .ap_clk(ap_clk),
    .ap_clk2x(ap_clk2x),
    .ap_rst_n(ap_rst_n),

    .s_axis_a_tdata (axis_s1_1.tdata),
    .s_axis_a_tvalid(axis_s1_1.tvalid),
    .s_axis_a_tready(axis_s1_1.tready),

    .m_axis_c_tdata (axis_s2_1.tdata),
    .m_axis_c_tvalid(axis_s2_1.tvalid),
    .m_axis_c_tready(axis_s2_1.tready)
);

mm_matmul_sta #(
    .MH(MH),
    .MW(MW),
    .PE(PE_MM_P0),
    .SIMD(SIMD_MM_P0),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .THRESHOLDS_PATH(THRESHOLDS_PATH_P0_2),
    .INIT_FILE(INIT_FILE_P0_2)
) inst_matmul_2 (
    .ap_clk(ap_clk),
    .ap_clk2x(ap_clk2x),
    .ap_rst_n(ap_rst_n),

    .s_axis_a_tdata (axis_s1_2.tdata),
    .s_axis_a_tvalid(axis_s1_2.tvalid),
    .s_axis_a_tready(axis_s1_2.tready),

    .m_axis_c_tdata (axis_s2_2.tdata),
    .m_axis_c_tvalid(axis_s2_2.tvalid),
    .m_axis_c_tready(axis_s2_2.tready)
);

dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(PE_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s2_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_0), .m_axis(m_axis_0));
dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(PE_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s2_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_1), .m_axis(m_axis_1));
dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(PE_P0*ACTIVATION_WIDTH)) inst_dwc_buff_s2_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_2), .m_axis(m_axis_2));

endmodule