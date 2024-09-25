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

/**
 * Partitions top
 *
 */
module p012_top #(
    // Partition 0
    parameter int unsigned                          SIMD_P0 = 4,
    parameter int unsigned                          PE_P0 = 4,

    parameter integer                               MH_P0_MM = 384,
    parameter integer                               MW_P0_MM = 384,
    parameter integer                               SIMD_P0_MM = 48,
    parameter integer                               PE_P0_MM = 32,
    parameter integer                               TH_P0_MM = 2*PE_P0_MM,
    parameter integer                               PE_P0_THR = 4,

    parameter                                       INIT_FILE_P0_0 = "",
    parameter                                       INIT_FILE_P0_1 = "",
    parameter                                       INIT_FILE_P0_2 = "",   
    
    // Partition 1
    parameter int unsigned                          SIMD_P1 = 4,
    parameter int unsigned                          PE_P1 = 4,

    parameter int unsigned                          SIMD_P1_MM_0 = 16,
	parameter int unsigned                          PE_P1_MM_0 = 32,
    parameter int unsigned                          MH_P1_MM_0 = 32,
    parameter int unsigned                          MW_P1_MM_0 = 128,
    parameter int unsigned                          TH_P1_MM_0 = PE_P1_MM_0,
    parameter int unsigned                          MH_P1_OUTER_0 = 128,
    parameter int unsigned                          PE_P1_THR_0 = 16,

    parameter int unsigned                          SIMD_P1_MM_1 = 16,
    parameter int unsigned                          PE_P1_MM_1 = 32,
    parameter int unsigned                          MH_P1_MM_1 = 128,
    parameter int unsigned                          MW_P1_MM_1 = 32,
    parameter int unsigned                          TH_P1_MM_1 = PE_P1_MM_1,
    parameter int unsigned                          MH_P1_OUTER_1 = 128,
    parameter int unsigned                          PE_P1_THR_1 = 4,

    parameter                                       THRESHOLDS_PATH_P1_0 = "",
    parameter                                       THRESHOLDS_PATH_P1_1 = "",

    // Partition 2
    parameter int unsigned                          SIMD_P2_0 = 48,
    parameter int unsigned                          SIMD_P2_1 = 4,
    parameter int unsigned                          PE_P2 = 4,

    parameter integer                               MH_P2_MM_0 = 384,
    parameter integer                               MW_P2_MM_0 = 384,
    parameter integer                               SIMD_P2_MM_0 = 48,
    parameter integer                               PE_P2_MM_0 = 32,
    parameter integer                               TH_P2_MM_0 = 2*PE_P2_MM_0,
    parameter integer                               PE_P2_THR_0 = 4,

    parameter integer                               MH_P2_MM_1 = 384,
    parameter integer                               MW_P2_MM_1 = 1536,
    parameter integer                               SIMD_P2_MM_1 = 96,
    parameter integer                               PE_P2_MM_1 = 64,
    parameter integer                               TH_P2_MM_1 = 2*PE_P2_MM_1,
    parameter integer                               PE_P2_THR_1 = 16,

    parameter integer                               MH_P2_MM_2 = 1536,
    parameter integer                               MW_P2_MM_2 = 384,
    parameter integer                               SIMD_P2_MM_2 = 96,
    parameter integer                               PE_P2_MM_2 = 64,
    parameter integer                               TH_P2_MM_2 = 2*PE_P2_MM_2,
    parameter integer                               PE_P2_THR_2 = 4,

    parameter                                       INIT_FILE_P2_0 = "",
    parameter                                       INIT_FILE_P2_1 = "",
    parameter                                       INIT_FILE_P2_2 = "",   

    // Rest
    parameter int unsigned                          ACTIVATION_WIDTH = 8,
    parameter int unsigned                          PUMPED_COMPUTE = 0,
    parameter int unsigned                          MM_KERNEL = 0
) (
    input  logic                                    ap_clk,
    input  logic                                    ap_clk2x,
    input  logic                                    ap_rst_n,

    AXI4S.slave                                     s_axis,
    AXI4S.master                                    m_axis
);

//
// Instantiations
//

AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_outer ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_att ();

AXI4S #(.AXI4S_DATA_BITS(SIMD_P0*ACTIVATION_WIDTH)) axis_p0_in ();
AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_p0_out_0 ();
AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_p0_out_1 ();
AXI4S #(.AXI4S_DATA_BITS(PE_P0*ACTIVATION_WIDTH)) axis_p0_out_2 ();

AXI4S #(.AXI4S_DATA_BITS(SIMD_P1*ACTIVATION_WIDTH)) axis_p1_in_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P1*ACTIVATION_WIDTH)) axis_p1_in_1 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P1*ACTIVATION_WIDTH)) axis_p1_in_2 ();
AXI4S #(.AXI4S_DATA_BITS(PE_P1*ACTIVATION_WIDTH)) axis_p1_out ();

AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) axis_p2_in_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_1*ACTIVATION_WIDTH)) axis_p2_in_1 ();

// Broadcast
broadcast #(
    .M_COUNT(2),
    .DATA_WIDTH(SIMD_P0*ACTIVATION_WIDTH)
) inst_bcast (
    .ap_clk             (ap_clk),
    .ap_rst_n           (ap_rst_n),
    .s_axis_tvalid      (s_axis.tvalid),
    .s_axis_tready      (s_axis.tready),
    .s_axis_tdata       (s_axis.tdata),

    .m_axis_tvalid      ({axis_att.tvalid, axis_outer.tvalid}),
    .m_axis_tready      ({axis_att.tready, axis_outer.tready}),
    .m_axis_tdata       ({axis_att.tdata,  axis_outer.tdata})
);

dwc_buff_top #(.I_BITS(SIMD_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_P0*ACTIVATION_WIDTH)) inst_dwc_buff_p0_in (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_att), .m_axis(axis_p0_in));

// Bypass
axis_buff inst_outer_buff (
    .s_axis_aclk        (ap_clk),
    .s_axis_aresetn     (ap_rst_n),
    
    .s_axis_tvalid      (axis_outer.tvalid),
    .s_axis_tready      (axis_outer.tready),
    .s_axis_tdata       (axis_outer.tdata),
    
    .m_axis_tvalid      (axis_p2_in_1.tvalid),
    .m_axis_tready      (axis_p2_in_1.tready),
    .m_axis_tdata       (axis_p2_in_1.tdata)
);

// Partition - 0
p0_top #(
    .SIMD(SIMD_P0),
    .PE(PE_P0),

    .MH_MM(MH_P0_MM),
    .MW_MM(MW_P0_MM),
    .SIMD_MM(SIMD_P0_MM),
    .PE_MM(PE_P0_MM),
    .TH_MM(TH_P0_MM),
    .PE_THR(PE_P0_THR),

    .INIT_FILE_0(INIT_FILE_P0_0),
    .INIT_FILE_1(INIT_FILE_P0_1),
    .INIT_FILE_2(INIT_FILE_P0_2),

    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .MM_KERNEL(MM_KERNEL)
) inst_p0_top (
    .ap_clk                     (ap_clk),
    .ap_clk2x                   (ap_clk2x),
    .ap_rst_n                   (ap_rst_n),

    .s_axis_0                   (axis_p0_in),
    .m_axis_0                   (axis_p0_out_0),
    .m_axis_1                   (axis_p0_out_1),
    .m_axis_2                   (axis_p0_out_2)
);

dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_P1*ACTIVATION_WIDTH)) inst_dwc_buff_p1_in_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_p0_out_0), .m_axis(axis_p1_in_0));
dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_P1*ACTIVATION_WIDTH)) inst_dwc_buff_p1_in_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_p0_out_1), .m_axis(axis_p1_in_1));
dwc_buff_top #(.I_BITS(PE_P0*ACTIVATION_WIDTH), .O_BITS(SIMD_P1*ACTIVATION_WIDTH)) inst_dwc_buff_p1_in_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_p0_out_2), .m_axis(axis_p1_in_2));

// Partition - 1
p1_top #(
    .SIMD_MM_0(SIMD_P1_MM_0),
    .PE_MM_0(PE_P1_MM_0),
    .MH_MM_0(MH_P1_MM_0),
    .MW_MM_0(MW_P1_MM_0),
    .TH_MM_0(TH_P1_MM_0),
    .MH_OUTER_0(MH_P1_OUTER_0),
    .PE_THR_0(PE_P1_THR_0),
    .THRESHOLDS_PATH_0(THRESHOLDS_PATH_P1_0), 

    .SIMD_MM_1(SIMD_P1_MM_1),
    .PE_MM_1(PE_P1_MM_1),
    .MH_MM_1(MH_P1_MM_1),
    .MW_MM_1(MW_P1_MM_1),
    .TH_MM_1(TH_P1_MM_1),
    .MH_OUTER_1(MH_P1_OUTER_1),
    .PE_THR_1(PE_P1_THR_1),
    .THRESHOLDS_PATH_1(THRESHOLDS_PATH_P1_1), 
    
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .MM_KERNEL(0) // No need here ...
) inst_p1_top (
    .ap_clk                     (ap_clk),
    .ap_clk2x                   (ap_clk2x),
    .ap_rst_n                   (ap_rst_n),

    .s_axis_0_a                 (axis_p1_in_0),
    .s_axis_0_b                 (axis_p1_in_1),
    .s_axis_1_b                 (axis_p1_in_2),
    .m_axis_1_c                 (axis_p1_out)
);

dwc_buff_top #(.I_BITS(PE_P1*ACTIVATION_WIDTH), .O_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) inst_dwc_buff_p1_out (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_p1_out), .m_axis(axis_p2_in_0));

AXI4S #(.AXI4S_DATA_BITS(PE_P2*ACTIVATION_WIDTH)) m_axis_int();

// Partition 2
p2_top #(
    .SIMD_0(SIMD_P2_0),
    .SIMD_1(SIMD_P2_1),
    .PE(PE_P2),

    .MH_MM_0(MH_P2_MM_0),
    .MW_MM_0(MW_P2_MM_0),
    .SIMD_MM_0(SIMD_P2_MM_0),
    .PE_MM_0(PE_P2_MM_0),
    .TH_MM_0(TH_P2_MM_0),
    .PE_THR_0(PE_P2_THR_0),

    .MH_MM_1(MH_P2_MM_1),
    .MW_MM_1(MW_P2_MM_1),
    .SIMD_MM_1(SIMD_P2_MM_1),
    .PE_MM_1(PE_P2_MM_1),
    .TH_MM_1(TH_P2_MM_1),
    .PE_THR_1(PE_P2_THR_1),

    .MH_MM_2(MH_P2_MM_2),
    .MW_MM_2(MW_P2_MM_2),
    .SIMD_MM_2(SIMD_P2_MM_2),
    .PE_MM_2(PE_P2_MM_2),
    .TH_MM_2(TH_P2_MM_2),
    .PE_THR_2(PE_P2_THR_2),

    .INIT_FILE_0(INIT_FILE_P2_0),
    .INIT_FILE_1(INIT_FILE_P2_1),
    .INIT_FILE_2(INIT_FILE_P2_2),

    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),
    .MM_KERNEL(MM_KERNEL)
) inst_p2_top (
    .ap_clk             (ap_clk),
    .ap_clk2x           (ap_clk2x),
    .ap_rst_n           (ap_rst_n),

    .s_axis_0           (axis_p2_in_0),
    .s_axis_1           (axis_p2_in_1),
    .m_axis_0           (m_axis)
);

endmodule