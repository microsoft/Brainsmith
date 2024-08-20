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
 * @brief	Partition 1 top level
 * @author	Dario Korolija <dario.korolija@amd.com>
 *****************************************************************************/
`timescale 1ns / 1ps

module p1_top #(
    // MatMul 0
	int unsigned PE_0 = 32,
	int unsigned SIMD_0 = 16,
    int unsigned MH_0 = 32,
    int unsigned MW_0 = 128,
    int unsigned MH_OUTER_0 = 128,
    
    int unsigned PE_THR_0 = PE_0/(MH_0/SIMD_0),
    bit PUMPED_COMPUTE_0 = 1,
    parameter  THRESHOLDS_PATH_0 = "",

    // MatMul 1
	int unsigned PE_1 = 32,
	int unsigned SIMD_1 = 16,
    int unsigned MH_1 = 128,
    int unsigned MW_1 = 32,
    int unsigned MH_OUTER_1 = 128,

    int unsigned PE_THR_1 = PE_1/(MH_1/SIMD_1),
    bit PUMPED_COMPUTE_1 = 1,
    parameter  THRESHOLDS_PATH_1 = "",

    // MV
    int unsigned N_VECTORS_0 = MH_OUTER_0,
    int unsigned N_VECTORS_1 = MH_OUTER_1,
    // MM
    int unsigned MM_KERNEL = 0,
    int unsigned TILE_H_0 = PE_0,
    int unsigned TILE_H_1 = PE_1,
    int unsigned N_TILES_0 = MH_OUTER_0 / TILE_H_0,
    int unsigned N_TILES_1 = MH_OUTER_1 / TILE_H_1,

    // Shuffles
    int unsigned PE_SHUFFLE_A = 4,
    int unsigned PE_SHUFFLE_B = 4,

    // Softmax
    int unsigned EN_SOFTMAX = 0,
    int unsigned PE_SOFTMAX = PE_THR_0,

    // Activation bits
    int unsigned ACTIVATION_WIDTH = 8,

    // Safely deducible parameters
    localparam int unsigned  B_STREAM_WIDTH_0           = PE_0 * ACTIVATION_WIDTH,
	localparam int unsigned  B_STREAM_WIDTH_BA_0        = (B_STREAM_WIDTH_0 + 7)/8 * 8,
	localparam int unsigned  A_STREAM_WIDTH_0           = SIMD_0 * ACTIVATION_WIDTH,
	localparam int unsigned  A_STREAM_WIDTH_BA_0        = (A_STREAM_WIDTH_0  + 7)/8 * 8,
	localparam int unsigned  C_STREAM_WIDTH_0           = PE_THR_0 * ACTIVATION_WIDTH,
	localparam int unsigned  C_STREAM_WIDTH_BA_0        = (C_STREAM_WIDTH_0 + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN_0  = SIMD_0 % 2,

	localparam int unsigned  B_STREAM_WIDTH_1           = PE_1 * ACTIVATION_WIDTH,
	localparam int unsigned  B_STREAM_WIDTH_BA_1        = (B_STREAM_WIDTH_1 + 7)/8 * 8,
	localparam int unsigned  A_STREAM_WIDTH_1           = SIMD_1 * ACTIVATION_WIDTH,
	localparam int unsigned  A_STREAM_WIDTH_BA_1        = (A_STREAM_WIDTH_1  + 7)/8 * 8,
	localparam int unsigned  C_STREAM_WIDTH_1           = PE_THR_1 * ACTIVATION_WIDTH,
	localparam int unsigned  C_STREAM_WIDTH_BA_1        = (C_STREAM_WIDTH_1 + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN_1  = SIMD_1 % 2,

    // Rest
    parameter Q_DEPTH = 16,
    bit IS_MVU = 1,
    int unsigned SEGMENTLEN = 2,
    bit SIGNED_ACTIVATIONS = 0,
    bit FORCE_BEHAVIORAL = 0,
    bit M_REG_LUT = 1
) ( 
    // Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	        // synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// B matrix stream MatMul 0
    AXI4S.s                             s_axis_0_b,

    // A matrix stream MatMul 0
    AXI4S.s                             s_axis_0_a,

    // B matrix stream MatMul 1
    AXI4S.s                             s_axis_1_b,

    // C matrix stream MatMul 1
    AXI4S.m                             m_axis_1_c
);

//
// Params
//

localparam COMPUTE_CORE = (ACTIVATION_WIDTH == 8) ? "mvu_vvu_8sx9_dsp58" : "mvu_4sx4u";

//
// Instantiation
//

AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_B*ACTIVATION_WIDTH)) axis_0_b ();
AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) axis_0_a ();
AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) axis_1_b ();
AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) axis_1_c ();

AXI4S #(.AXI4S_DATA_BITS(B_STREAM_WIDTH_BA_0)) axis_dwc_0_b ();
AXI4S #(.AXI4S_DATA_BITS(A_STREAM_WIDTH_BA_0)) axis_dwc_0_a ();
AXI4S #(.AXI4S_DATA_BITS(B_STREAM_WIDTH_BA_1)) axis_dwc_1_b ();
AXI4S #(.AXI4S_DATA_BITS(C_STREAM_WIDTH_BA_1)) axis_dwc_1_c ();

AXI4S #(.AXI4S_DATA_BITS(C_STREAM_WIDTH_BA_0)) axis_mm_0_c ();
AXI4S #(.AXI4S_DATA_BITS(PE_SOFTMAX*ACTIVATION_WIDTH)) axis_sm_in ();
AXI4S #(.AXI4S_DATA_BITS(PE_SOFTMAX*ACTIVATION_WIDTH)) axis_sm_out ();
AXI4S #(.AXI4S_DATA_BITS(A_STREAM_WIDTH_BA_1)) axis_mm_1_a ();

// Shuffle-A
shuffleB_0 inst_shuffle_0_b (
    .ap_clk         (ap_clk),
    .ap_rst_n       (ap_rst_n),
    
    .src_TDATA      (s_axis_0_b.tdata),
    .src_TVALID     (s_axis_0_b.tvalid),
    .src_TREADY     (s_axis_0_b.tready),

    .dst_TDATA      (axis_0_b.tdata),
    .dst_TVALID     (axis_0_b.tvalid),
    .dst_TREADY     (axis_0_b.tready)
);

dwc_buff_top #(.I_BITS(PE_SHUFFLE_B*ACTIVATION_WIDTH), .O_BITS(B_STREAM_WIDTH_BA_0)) inst_dwc_buff_0_b (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_0_b), .m_axis(axis_dwc_0_b));

// Shuffle
shuffleA_0 inst_shuffle_0_a (
    .ap_clk         (ap_clk),
    .ap_rst_n       (ap_rst_n),
    
    .src_TDATA      (s_axis_0_a.tdata),
    .src_TVALID     (s_axis_0_a.tvalid),
    .src_TREADY     (s_axis_0_a.tready),

    .dst_TDATA      (axis_0_a.tdata),
    .dst_TVALID     (axis_0_a.tvalid),
    .dst_TREADY     (axis_0_a.tready)
);

dwc_buff_top #(.I_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH), .O_BITS(A_STREAM_WIDTH_BA_0)) inst_dwc_buff_0_a (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_0_a), .m_axis(axis_dwc_0_a));

// Shuffle
shuffleA_0 inst_shuffle_1_b (
    .ap_clk         (ap_clk),
    .ap_rst_n       (ap_rst_n),
    
    .src_TDATA      (s_axis_1_b.tdata),
    .src_TVALID     (s_axis_1_b.tvalid),
    .src_TREADY     (s_axis_1_b.tready),

    .dst_TDATA      (axis_1_b.tdata),
    .dst_TVALID     (axis_1_b.tvalid),
    .dst_TREADY     (axis_1_b.tready)
);

dwc_buff_top #(.I_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH), .O_BITS(B_STREAM_WIDTH_BA_0), .O_QDEPTH((MH_1 * MW_1) / PE_1)) inst_dwc_buff_1_b (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_1_b), .m_axis(axis_dwc_1_b));

// MatMul
if(MM_KERNEL == 0) begin
    mv_matmul_dyn #(
        .PE(PE_0),
        .SIMD(SIMD_0),
        .MH(MH_0),
        .MW(MW_0),
        .N_VECTORS(N_VECTORS_0),
        .PE_THR(PE_THR_0),
        .TH(TILE_H_0),
        .PUMPED_COMPUTE(PUMPED_COMPUTE_0),
        .THRESHOLDS_PATH(THRESHOLDS_PATH_0),
        .COMPUTE_CORE(COMPUTE_CORE),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) inst_MatMul_0 (
        .ap_clk             (ap_clk),
        .ap_clk2x           (ap_clk2x),
        .ap_rst_n           (ap_rst_n),

        .s_axis_b_tvalid    (axis_dwc_0_b.tvalid),
        .s_axis_b_tready    (axis_dwc_0_b.tready),
        .s_axis_b_tdata     (axis_dwc_0_b.tdata),

        .s_axis_a_tvalid    (axis_dwc_0_a.tvalid),
        .s_axis_a_tready    (axis_dwc_0_a.tready),
        .s_axis_a_tdata     (axis_dwc_0_a.tdata),

        .m_axis_c_tvalid    (axis_mm_0_c.tvalid),
        .m_axis_c_tready    (axis_mm_0_c.tready),
        .m_axis_c_tdata     (axis_mm_0_c.tdata)
    );
end
else begin
    mm_matmul_dyn #(
        .PE(PE_0),
        .SIMD(SIMD_0),
        .MH(MH_0),
        .MW(MW_0),
        .TH(TILE_H_0),
        .N_TILES(N_TILES_0),
        .PE_THR(PE_THR_0),
        .PUMPED_COMPUTE(PUMPED_COMPUTE_0),
        .THRESHOLDS_PATH(THRESHOLDS_PATH_0),
        .COMPUTE_CORE(COMPUTE_CORE),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) inst_MatMul_0 (
        .ap_clk             (ap_clk),
        .ap_clk2x           (ap_clk2x),
        .ap_rst_n           (ap_rst_n),

        .s_axis_b_tvalid    (axis_dwc_0_b.tvalid),
        .s_axis_b_tready    (axis_dwc_0_b.tready),
        .s_axis_b_tdata     (axis_dwc_0_b.tdata),

        .s_axis_a_tvalid    (axis_dwc_0_a.tvalid),
        .s_axis_a_tready    (axis_dwc_0_a.tready),
        .s_axis_a_tdata     (axis_dwc_0_a.tdata),

        .m_axis_c_tvalid    (axis_mm_0_c.tvalid),
        .m_axis_c_tready    (axis_mm_0_c.tready),
        .m_axis_c_tdata     (axis_mm_0_c.tdata)
    );
end

if(EN_SOFTMAX) begin
dwc_buff_top #(.I_BITS(C_STREAM_WIDTH_BA_0), .O_BITS(PE_SOFTMAX*ACTIVATION_WIDTH)) inst_buff_sm_in (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_mm_0_c), .m_axis(axis_sm_in));

// Softmax
softmaxquant_0 inst_softmax (
    .ap_clk         (ap_clk),
    .ap_rst_n       (ap_rst_n),
    
    .src_TDATA      (axis_sm_in.tdata),
    .src_TVALID     (axis_sm_in.tvalid),
    .src_TREADY     (axis_sm_in.tready),

    .dst_TDATA      (axis_sm_out.tdata),
    .dst_TVALID     (axis_sm_out.tvalid),
    .dst_TREADY     (axis_sm_out.tready)
);

dwc_buff_top #(.I_BITS(PE_SOFTMAX*ACTIVATION_WIDTH), .O_BITS(A_STREAM_WIDTH_BA_1)) inst_buff_sm_out (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_sm_out), .m_axis(axis_mm_1_a));
end
else begin
dwc_buff_top #(.I_BITS(C_STREAM_WIDTH_BA_0), .O_BITS(A_STREAM_WIDTH_BA_1)) inst_buff_sm (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_mm_0_c), .m_axis(axis_mm_1_a));
end

// MatMul
if(MM_KERNEL == 0) begin
    mv_matmul_dyn #(
        .PE(PE_1),
        .SIMD(SIMD_1),
        .MH(MH_1),
        .MW(MW_1),
        .N_VECTORS(N_VECTORS_1),
        .PE_THR(PE_THR_1),
        .PUMPED_COMPUTE(PUMPED_COMPUTE_1),
        .THRESHOLDS_PATH(THRESHOLDS_PATH_1),
        .TH(TILE_H_1),
        .COMPUTE_CORE(COMPUTE_CORE),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) inst_MatMul_1 (
        .ap_clk             (ap_clk),
        .ap_clk2x           (ap_clk2x),
        .ap_rst_n           (ap_rst_n),

        .s_axis_b_tvalid    (axis_dwc_1_b.tvalid),
        .s_axis_b_tready    (axis_dwc_1_b.tready),
        .s_axis_b_tdata     (axis_dwc_1_b.tdata),

        .s_axis_a_tvalid    (axis_mm_1_a.tvalid),
        .s_axis_a_tready    (axis_mm_1_a.tready),
        .s_axis_a_tdata     (axis_mm_1_a.tdata),

        .m_axis_c_tvalid    (axis_dwc_1_c.tvalid),
        .m_axis_c_tready    (axis_dwc_1_c.tready),
        .m_axis_c_tdata     (axis_dwc_1_c.tdata)
    );
end
else begin
    mm_matmul_dyn #(
        .PE(PE_1),
        .SIMD(SIMD_1),
        .MH(MH_1),
        .MW(MW_1),
        .TH(TILE_H_1),
        .N_TILES(N_TILES_1),
        .PE_THR(PE_THR_1),
        .PUMPED_COMPUTE(PUMPED_COMPUTE_1),
        .THRESHOLDS_PATH(THRESHOLDS_PATH_1),
        .COMPUTE_CORE(COMPUTE_CORE),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) inst_MatMul_1 (
        .ap_clk             (ap_clk),
        .ap_clk2x           (ap_clk2x),
        .ap_rst_n           (ap_rst_n),

        .s_axis_b_tvalid    (axis_dwc_1_b.tvalid),
        .s_axis_b_tready    (axis_dwc_1_b.tready),
        .s_axis_b_tdata     (axis_dwc_1_b.tdata),

        .s_axis_a_tvalid    (axis_mm_1_a.tvalid),
        .s_axis_a_tready    (axis_mm_1_a.tready),
        .s_axis_a_tdata     (axis_mm_1_a.tdata),

        .m_axis_c_tvalid    (axis_dwc_1_c.tvalid),
        .m_axis_c_tready    (axis_dwc_1_c.tready),
        .m_axis_c_tdata     (axis_dwc_1_c.tdata)
    );
end

dwc_buff_top #(.I_BITS(C_STREAM_WIDTH_BA_1), .O_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) inst_dwc_buff_1_c (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_dwc_1_c), .m_axis(axis_1_c));

// Shuffle

shuffleC_0 inst_shuffle_1_c (
    .ap_clk         (ap_clk),
    .ap_rst_n       (ap_rst_n),
    
    .src_TDATA      (axis_1_c.tdata),
    .src_TVALID     (axis_1_c.tvalid),
    .src_TREADY     (axis_1_c.tready),

    .dst_TDATA      (m_axis_1_c.tdata),
    .dst_TVALID     (m_axis_1_c.tvalid),
    .dst_TREADY     (m_axis_1_c.tready)
);


endmodule