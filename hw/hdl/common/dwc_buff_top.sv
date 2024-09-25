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
module dwc_buff_top #(
    parameter I_BITS = 32,
    parameter O_BITS = 32,
    parameter I_BUFF = 1,
    parameter O_BUFF = 1,
    parameter I_QDEPTH = 16,
    parameter O_QDEPTH = 16
) (
    input  logic                            ap_clk,
    input  logic                            ap_rst_n,

    AXI4S.slave                                 s_axis,
    AXI4S.master                                m_axis
);

dwc_buff #(
    .I_BITS(I_BITS),
    .O_BITS(O_BITS),
    .I_BUFF(I_BUFF),
    .O_BUFF(O_BUFF),
    .I_QDEPTH(I_QDEPTH),
    .O_QDEPTH(O_QDEPTH)
) inst_dwc_buff_int (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .s_axis_tvalid(s_axis.tvalid),
    .s_axis_tready(s_axis.tready),
    .s_axis_tdata (s_axis.tdata),
    .m_axis_tvalid(m_axis.tvalid),
    .m_axis_tready(m_axis.tready),
    .m_axis_tdata (m_axis.tdata),
    .c_i(),
    .m_i(),
    .c_o(),
    .m_o()
);

endmodule