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
module dwc_buff #(
    parameter I_BITS = 8,
    parameter O_BITS = 64,
    parameter I_BUFF = 1,
    parameter O_BUFF = 1,
    parameter I_QDEPTH = 16,
    parameter O_QDEPTH = 16
) (
    input  logic                            ap_clk,
    input  logic                            ap_rst_n,

    input  logic                            s_axis_tvalid,
    output logic                            s_axis_tready,
    input  logic [I_BITS-1:0]               s_axis_tdata,

    output logic                            m_axis_tvalid,
    input  logic                            m_axis_tready,
    output logic [O_BITS-1:0]               m_axis_tdata,

    output logic [$clog2(I_QDEPTH + 1)-1:0] c_i,
    output logic [$clog2(I_QDEPTH)-1:0]     m_i,
    output logic [$clog2(O_QDEPTH + 1)-1:0] c_o,
    output logic [$clog2(O_QDEPTH)-1:0]     m_o
);

if(I_BITS != O_BITS) begin
    // DWC and two buffers

    // Buffer input
    logic [I_BITS-1:0] axis_s0_tdata;
    logic axis_s0_tvalid;
    logic axis_s0_tready;

    if(I_BUFF == 1) begin
        Q_srl #(
            .depth(I_QDEPTH), .width(I_BITS)
        ) inst_q_in (
            .clock(ap_clk),
            .reset(!ap_rst_n),
            .count(c_i),
            .maxcount(m_i),
            .i_d(s_axis_tdata),
            .i_v(s_axis_tvalid),
            .i_r(s_axis_tready),
            .o_d(axis_s0_tdata),
            .o_v(axis_s0_tvalid),
            .o_r(axis_s0_tready)
        );
    end
    else begin
        assign axis_s0_tvalid = s_axis_tvalid;
        assign axis_s0_tdata  = s_axis_tdata;
        assign s_axis_tready  = axis_s0_tready;

        assign c_i = 0;
        assign m_i = 0;
    end

    // Data width conversion
    logic [O_BITS-1:0] axis_s1_tdata;
    logic axis_s1_tvalid;
    logic axis_s1_tready;

    dwc_axi #(
        .IBITS(I_BITS), .OBITS(O_BITS)
    ) inst_q (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_tdata (axis_s0_tdata),
        .s_axis_tvalid(axis_s0_tvalid),
        .s_axis_tready(axis_s0_tready),
        .m_axis_tdata (axis_s1_tdata),
        .m_axis_tvalid(axis_s1_tvalid),
        .m_axis_tready(axis_s1_tready)
    );

    // Buffer output
    if(O_BUFF == 1) begin
        Q_srl #(
            .depth(O_QDEPTH), .width(O_BITS)
        ) inst_q_out (
            .clock(ap_clk),
            .reset(!ap_rst_n),
            .count(c_o),
            .maxcount(m_o),
            .i_d(axis_s1_tdata),
            .i_v(axis_s1_tvalid),
            .i_r(axis_s1_tready),
            .o_d(m_axis_tdata),
            .o_v(m_axis_tvalid),
            .o_r(m_axis_tready)
        );
    end
    else begin
        assign m_axis_tvalid  = axis_s1_tvalid;
        assign m_axis_tdata   = axis_s1_tdata;
        assign axis_s1_tready = m_axis_tready;

        assign c_o = 0;
        assign m_o = 0;
    end
end
else begin
    // Just a single buffer (input depth)
    if(I_BUFF == 1) begin
        Q_srl #(
            .depth(I_QDEPTH), .width(I_BITS)
        ) inst_q_in (
            .clock(ap_clk),
            .reset(!ap_rst_n),
            .count(c_i),
            .maxcount(m_i),
            .i_d(s_axis_tdata),
            .i_v(s_axis_tvalid),
            .i_r(s_axis_tready),
            .o_d(m_axis_tdata),
            .o_v(m_axis_tvalid),
            .o_r(m_axis_tready)
        );
    end
    else begin
        assign m_axis_tvalid  = s_axis_tvalid;
        assign m_axis_tdata   = s_axis_tdata;
        assign s_axis_tready  = m_axis_tready;

        assign c_i = 0;
        assign m_i = 0;
    end

    assign c_0 = 0;
    assign m_0 = 0;
end


endmodule