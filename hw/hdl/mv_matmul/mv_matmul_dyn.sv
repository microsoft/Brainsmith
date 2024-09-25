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
module mv_matmul_dyn #(
    // MatMul
	int unsigned PE,
	int unsigned SIMD,
    int unsigned MH,
    int unsigned MW,
	int unsigned N_VECTORS,

    int unsigned ACTIVATION_WIDTH = 8,
	int unsigned ACCU_WIDTH = 2*ACTIVATION_WIDTH+$clog2(MH),

    parameter COMPUTE_CORE = "mvu_vvu_8sx9_dsp58",
    bit PUMPED_COMPUTE = 1,

    // Safely deducible parameters
	localparam int unsigned  B_STREAM_WIDTH             = PE * ACTIVATION_WIDTH,
	localparam int unsigned  B_STREAM_WIDTH_BA          = (B_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  A_STREAM_WIDTH             = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  A_STREAM_WIDTH_BA          = (A_STREAM_WIDTH  + 7)/8 * 8,
    localparam int unsigned  C_STREAM_WIDTH             = PE * ACCU_WIDTH,
	localparam int unsigned  C_STREAM_WIDTH_BA          = (C_STREAM_WIDTH + 7)/8 * 8,

	localparam bit  		 SIMD_UNEVEN  = SIMD % 2,

    // Rest
    bit IS_MVU = 1,
    int unsigned SEGMENTLEN = 1,
    bit SIGNED_ACTIVATIONS = 1,
    bit FORCE_BEHAVIORAL = 0,
    bit M_REG_LUT = 1,
    parameter  THRESHOLDS_PATH = ""	// Directory with initial threshold data
) ( 
    // Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	// synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// B matrix stream MatMul 0
	input	logic [B_STREAM_WIDTH_BA-1:0]  s_axis_b_tdata,
	input	logic  s_axis_b_tvalid,
	output	logic  s_axis_b_tready,

    // A matrix stream MatMul 0
	input	logic [A_STREAM_WIDTH_BA-1:0]  s_axis_a_tdata,
	input	logic  s_axis_a_tvalid,
	output	logic  s_axis_a_tready,

	// C matrix stream MatMul 1
	output	logic [C_STREAM_WIDTH_BA-1:0]  m_axis_c_tdata, // after thresholding
	output	logic  m_axis_c_tvalid,
	input	logic  m_axis_c_tready
);

//
// Signals
//

// Input weights
typedef logic [PE-1:0][ACTIVATION_WIDTH-1:0] dyn_w_t;
typedef logic [PE-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0] mu_w_t;
uwire mu_w_t axis_b_tdata;
logic axis_b_tvalid;
logic axis_b_tready;

//
// Instantiations
//

// Matrix load
mv_matrix_load_2 #(
    .PE(PE), .SIMD(SIMD),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .MH(MH), .MW(MW),
    .N_REPS(N_VECTORS)
) inst_matrix_load (
    .clk(ap_clk), 
    .rst(~ap_rst_n),
    .ivld(s_axis_b_tvalid), 
    .irdy(s_axis_b_tready), 
    .idat(dyn_w_t'(s_axis_b_tdata)),
    .ovld(axis_b_tvalid), 
    .ordy(axis_b_tready), 
    .odat(axis_b_tdata)
);

// MVAU
mvau #(
    .COMPUTE_CORE(COMPUTE_CORE),
    .MW(MW),
    .MH(MH),
    .PE(PE),
    .SIMD(SIMD),
    .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
    .WEIGHT_WIDTH(ACTIVATION_WIDTH),
    .ACCU_WIDTH(ACCU_WIDTH),
    .PUMPED_COMPUTE(PUMPED_COMPUTE),

    .IS_MVU(IS_MVU),
    .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
    .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL),
    .M_REG_LUT(M_REG_LUT)
) inst_MvauDyn (
    .ap_clk             (ap_clk),
    .ap_clk2x           (ap_clk2x),
    .ap_rst_n           (ap_rst_n),

    .s_axis_b_tdata     (axis_b_tdata),
    .s_axis_b_tvalid    (axis_b_tvalid),
    .s_axis_b_tready    (axis_b_tready),

    .s_axis_a_tdata     (s_axis_a_tdata),
    .s_axis_a_tvalid    (s_axis_a_tvalid),
    .s_axis_a_tready    (s_axis_a_tready),

    .m_axis_c_tdata     (m_axis_c_tdata),
    .m_axis_c_tvalid    (m_axis_c_tvalid),
    .m_axis_c_tready    (m_axis_c_tready)
);

endmodule