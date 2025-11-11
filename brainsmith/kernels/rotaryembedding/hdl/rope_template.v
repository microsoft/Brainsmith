/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

//- AXI Stream - Input --------------
output	in_V_TREADY,
input	in_V_TVALID,
input	[$STREAM_BITS$-1:0] in_V_TDATA,

//- AXI Stream - Output -------------
input	out_V_TREADY,
output	out_V_TVALID,
output	[$STREAM_BITS$-1:0] out_V_TDATA

);


rope_axi #(
.HEAD_DIM($HEAD_DIM$),
.SEQ_LEN($SEQ_LEN$),
.HIDDEN_DIM($HIDDEN_DIM$),
.SIMD($SIMD$),
.ELEM_BITS($ELEM_BITS$),
.SINCOS_WIDTH($WEIGHT_BITS$),
.COS_INIT_FILE($COS_INIT_FILE$),
.SIN_INIT_FILE($SIN_INIT_FILE$)
)
impl_0
(
 .ap_clk(ap_clk),
 .ap_rst_n(ap_rst_n),

 .s_axis_tready(in_V_TREADY),
 .s_axis_tvalid(in_V_TVALID),
 .s_axis_tdata(in_V_TDATA),

 .m_axis_tready(out_V_TREADY),
 .m_axis_tvalid(out_V_TVALID),
 .m_axis_tdata(out_V_TDATA)

);

endmodule
