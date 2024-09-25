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

module add_sta #(
	parameter int unsigned  ACTIVATION_WIDTH_0,
    parameter int unsigned  ACTIVATION_WIDTH_1,
    parameter int unsigned  PE = 1,

    parameter int unsigned  ACTIVATION_WIDTH_OUT = ACTIVATION_WIDTH_0 > ACTIVATION_WIDTH_1 ? ACTIVATION_WIDTH_0 + 1 : ACTIVATION_WIDTH_1 + 1,
    parameter int unsigned  INPUT_STREAM_WIDTH_0 = PE * ACTIVATION_WIDTH_0,
    parameter int unsigned  INPUT_STREAM_WIDTH_BA_0 = (INPUT_STREAM_WIDTH_0 + 7)/8 * 8,
    parameter int unsigned  INPUT_STREAM_WIDTH_1 = PE * ACTIVATION_WIDTH_1,
    parameter int unsigned  INPUT_STREAM_WIDTH_BA_1 = (INPUT_STREAM_WIDTH_1 + 7)/8 * 8,
    parameter int unsigned  OUTPUT_STREAM_WIDTH = PE * ACTIVATION_WIDTH_OUT,
    parameter int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8
)(
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	output	logic  s_axis_0_tready,
	input	logic  s_axis_0_tvalid,
	input	logic [INPUT_STREAM_WIDTH_BA_0-1:0] s_axis_0_tdata,

    output	logic  s_axis_1_tready,
	input	logic  s_axis_1_tvalid,
	input	logic [INPUT_STREAM_WIDTH_BA_1-1:0] s_axis_1_tdata,

	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0] m_axis_tdata
);

logic signed [PE-1:0][ACTIVATION_WIDTH_OUT-1:0] add_s;
logic add_vld;
logic en;

assign en = m_axis_tready;
assign s_axis_0_tready = en & s_axis_0_tvalid & s_axis_1_tvalid;
assign s_axis_1_tready = en & s_axis_0_tvalid & s_axis_1_tvalid;


assign m_axis_tdata = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){add_s[PE-1][ACTIVATION_WIDTH_OUT-1]}}, add_s};
assign m_axis_tvalid = add_vld;

always_ff @( posedge ap_clk ) begin : REG_PROC
    if(~ap_rst_n) begin
        add_s <= 0;
        add_vld <= 1'b0;
    end
    else begin
        if(en) begin
            for(int i = 0; i < PE; i++) begin
                add_s[i] <= $signed(s_axis_0_tdata[i*ACTIVATION_WIDTH_0+:ACTIVATION_WIDTH_0]) + $signed(s_axis_1_tdata[i*ACTIVATION_WIDTH_1+:ACTIVATION_WIDTH_1]);
                add_vld <= s_axis_0_tvalid & s_axis_1_tvalid;
            end
        end
    end
end

endmodule
