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

/******************************************************************************
 * @brief	Partition 2 top level
 * @author	Dario Korolija <dario.korolija@amd.com>
 *****************************************************************************/
`timescale 1ns / 1ps

/**
 * Control
 *
 */
module p2_top #(
    parameter                                       SIMD_P2_0 = 48,
    parameter                                       SIMD_P2_1 = 4,
    parameter                                       PE_P2 = 4,
    parameter                                       ACTIVATION_WIDTH = 8,
    parameter                                       MM_KERNEL = 0,
    parameter                                       PUMPED_COMPUTE = 1
) (
    input  logic                                    ap_clk,
    input  logic                                    ap_clk2x,
    input  logic                                    ap_rst_n,
    
    AXI4S.s                                         s_axis_0,
    AXI4S.s                                         s_axis_1,
    AXI4S.m                                         m_axis_0
);

if(MM_KERNEL == 1) begin
    partition_2_mm #(
        .SIMD_P2_0(SIMD_P2_0),
        .SIMD_P2_1(SIMD_P2_1),
        .PE_P2(PE_P2),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE)
    ) inst_p2_top (
        .ap_clk                         (ap_clk),
        .ap_clk2x                       (ap_clk2x),
        .ap_rst_n                       (ap_rst_n),

        .s_axis_0_tvalid                (s_axis_0.tvalid),
        .s_axis_0_tready                (s_axis_0.tready),
        .s_axis_0_tdata                 (s_axis_0.tdata),

        .s_axis_1_tvalid                (s_axis_1.tvalid),
        .s_axis_1_tready                (s_axis_1.tready),
        .s_axis_1_tdata                 (s_axis_1.tdata),

        .m_axis_0_tvalid                (m_axis_0.tvalid),
        .m_axis_0_tready                (m_axis_0.tready),
        .m_axis_0_tdata                 (m_axis_0.tdata)
    );
end
else begin
    if(PUMPED_COMPUTE == 1) begin
        partition_2 inst_p2_top_dp (
            .ap_clk                         (ap_clk),
            .ap_clk2x                       (ap_clk2x),
            .ap_rst_n                       (ap_rst_n),

            .s_axis_0_tvalid                (s_axis_0.tvalid),
            .s_axis_0_tready                (s_axis_0.tready),
            .s_axis_0_tdata                 (s_axis_0.tdata),

            .s_axis_1_tvalid                (s_axis_1.tvalid),
            .s_axis_1_tready                (s_axis_1.tready),
            .s_axis_1_tdata                 (s_axis_1.tdata),

            .m_axis_0_tvalid                (m_axis_0.tvalid),
            .m_axis_0_tready                (m_axis_0.tready),
            .m_axis_0_tdata                 (m_axis_0.tdata)
        );
    end
    else begin
        partition_2 inst_p2_top (
            .ap_clk                         (ap_clk),
            .ap_rst_n                       (ap_rst_n),

            .s_axis_0_tvalid                (s_axis_0.tvalid),
            .s_axis_0_tready                (s_axis_0.tready),
            .s_axis_0_tdata                 (s_axis_0.tdata),

            .s_axis_1_tvalid                (s_axis_1.tvalid),
            .s_axis_1_tready                (s_axis_1.tready),
            .s_axis_1_tdata                 (s_axis_1.tdata),

            .m_axis_0_tvalid                (m_axis_0.tvalid),
            .m_axis_0_tready                (m_axis_0.tready),
            .m_axis_0_tdata                 (m_axis_0.tdata)
        );
    end
end




endmodule