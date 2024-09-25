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

import cTypes::*;

module partition_0_mm #(
    parameter integer                               SIMD = 4,
    parameter integer                               PE = 4,
    
    // MM
    parameter integer                               MH_MM = 384,
    parameter integer                               MW_MM = 384,
    parameter integer                               SIMD_MM = 48,
    parameter integer                               PE_MM = 32,
    parameter integer                               TH_MM = 2*PE_MM,
    parameter integer                               PE_THR = 4,
    
    // Init
    parameter                                       INIT_FILE_0 = "",
    parameter                                       INIT_FILE_1 = "",
    parameter                                       INIT_FILE_2 = "",

    // Config
    parameter integer                               ACTIVATION_WIDTH = 8,
    parameter integer                               ACCU_WIDTH = 22,//2*ACTIVATION_WIDTH+$clog2(MH_MM),
    parameter integer                               PUMPED_COMPUTE = 1,
    parameter integer                               MM_KERNEL = 1
) (
    input  logic                                    ap_clk,
    input  logic                                    ap_clk2x,
    input  logic                                    ap_rst_n,
    
    AXI4S.slave                                         s_axis_0,
    AXI4S.master                                        m_axis_0,
    AXI4S.master                                        m_axis_1,
    AXI4S.master                                        m_axis_2
);

//
// Params
//

localparam int unsigned  MM_STREAM_WIDTH  = PE_MM * ACCU_WIDTH;
localparam int unsigned  MM_STREAM_WIDTH_BA  = (MM_STREAM_WIDTH + 7)/8 * 8;
localparam int unsigned  DWC_STREAM_WIDTH  = PE_THR * ACCU_WIDTH;

//
// Signals
//

AXI4S #(.AXI4S_DATA_BITS(SIMD*ACTIVATION_WIDTH)) axis_s0_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD*ACTIVATION_WIDTH)) axis_s0_1 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD*ACTIVATION_WIDTH)) axis_s0_2 ();

AXI4S #(.AXI4S_DATA_BITS(SIMD_MM*ACTIVATION_WIDTH)) axis_s1_0 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM*ACTIVATION_WIDTH)) axis_s1_1 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM*ACTIVATION_WIDTH)) axis_s1_2 ();

AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA)) axis_s2_0 ();
AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA)) axis_s2_1 ();
AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA)) axis_s2_2 ();

AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH)) axis_s3_0 ();
AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH)) axis_s3_1 ();
AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH)) axis_s3_2 ();

AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s4_0 ();
AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s4_1 ();
AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s4_2 ();

//
// Instantiations
//

// Broadcast
broadcast #(
    .M_COUNT(3),
    .DATA_WIDTH(SIMD*ACTIVATION_WIDTH)
) inst_bcast (
    .ap_clk             (ap_clk),
    .ap_rst_n           (ap_rst_n),

    .s_axis_tvalid      (s_axis_0.tvalid),
    .s_axis_tready      (s_axis_0.tready),
    .s_axis_tdata       (s_axis_0.tdata),

    .m_axis_tvalid      ({axis_s0_2.tvalid, axis_s0_1.tvalid, axis_s0_0.tvalid}),
    .m_axis_tready      ({axis_s0_2.tready, axis_s0_1.tready, axis_s0_0.tready}),
    .m_axis_tdata       ({axis_s0_2.tdata,  axis_s0_1.tdata,  axis_s0_0.tdata})
);

dwc_buff_top #(.I_BITS(SIMD*ACTIVATION_WIDTH), .O_BITS(SIMD_MM*ACTIVATION_WIDTH)) inst_dwc_buff_s0_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_0), .m_axis(axis_s1_0));
dwc_buff_top #(.I_BITS(SIMD*ACTIVATION_WIDTH), .O_BITS(SIMD_MM*ACTIVATION_WIDTH)) inst_dwc_buff_s0_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_1), .m_axis(axis_s1_1));
dwc_buff_top #(.I_BITS(SIMD*ACTIVATION_WIDTH), .O_BITS(SIMD_MM*ACTIVATION_WIDTH)) inst_dwc_buff_s0_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0_2), .m_axis(axis_s1_2));

// MatMul 0
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),
        .TH(TH_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_0)
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
end
else begin
    mv_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_0)
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
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA), .O_BITS(DWC_STREAM_WIDTH)) inst_dwc_buff_mm_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_0), .m_axis(axis_s3_0));

// Thr 0
p0_Thresholding_rtl_0_axi_wrapper inst_thr_0 (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),

    .s_axilite_AWVALID(1'b0),
    .s_axilite_AWREADY(),
    .s_axilite_AWADDR(0),
    .s_axilite_WVALID(0),
    .s_axilite_WREADY(),
    .s_axilite_WDATA(0),
    .s_axilite_WSTRB(0),
    .s_axilite_BVALID(),
    .s_axilite_BREADY(1'b1),
    .s_axilite_BRESP(),
    .s_axilite_ARVALID(1'b0),
    .s_axilite_ARREADY(),
    .s_axilite_ARADDR(0),
    .s_axilite_RVALID(),
    .s_axilite_RREADY(1'b1),
    .s_axilite_RDATA(),
    .s_axilite_RRESP(),
    
    .in0_V_TDATA (axis_s3_0.tdata),
    .in0_V_TVALID(axis_s3_0.tvalid),
    .in0_V_TREADY(axis_s3_0.tready),
    
    .out_V_TDATA (axis_s4_0.tdata),
    .out_V_TVALID(axis_s4_0.tvalid),
    .out_V_TREADY(axis_s4_0.tready)
);

dwc_buff_top #(.I_BITS(PE_THR*ACTIVATION_WIDTH), .O_BITS(PE*ACTIVATION_WIDTH)) inst_dwc_buff_thr_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s4_0), .m_axis(m_axis_0));

// MatMul 1
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),
        .TH(TH_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_1)
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
end
else begin
    mv_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_1)
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
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA), .O_BITS(DWC_STREAM_WIDTH)) inst_dwc_buff_mm_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_1), .m_axis(axis_s3_1));

// Thr 1
p0_Thresholding_rtl_1_axi_wrapper inst_thr_1 (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),

    .s_axilite_AWVALID(1'b0),
    .s_axilite_AWREADY(),
    .s_axilite_AWADDR(0),
    .s_axilite_WVALID(0),
    .s_axilite_WREADY(),
    .s_axilite_WDATA(0),
    .s_axilite_WSTRB(0),
    .s_axilite_BVALID(),
    .s_axilite_BREADY(1'b1),
    .s_axilite_BRESP(),
    .s_axilite_ARVALID(1'b0),
    .s_axilite_ARREADY(),
    .s_axilite_ARADDR(0),
    .s_axilite_RVALID(),
    .s_axilite_RREADY(1'b1),
    .s_axilite_RDATA(),
    .s_axilite_RRESP(),
    
    .in0_V_TDATA (axis_s3_1.tdata),
    .in0_V_TVALID(axis_s3_1.tvalid),
    .in0_V_TREADY(axis_s3_1.tready),
    
    .out_V_TDATA (axis_s4_1.tdata),
    .out_V_TVALID(axis_s4_1.tvalid),
    .out_V_TREADY(axis_s4_1.tready)
);

dwc_buff_top #(.I_BITS(PE_THR*ACTIVATION_WIDTH), .O_BITS(PE*ACTIVATION_WIDTH)) inst_dwc_buff_thr_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s4_1), .m_axis(m_axis_1));

// MatMul 2
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),
        .TH(TH_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_2)
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
end
else begin
    mv_matmul_sta #(
        .MH(MW_MM),
        .MW(MW_MM),
        .PE(PE_MM),
        .SIMD(SIMD_MM),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_2)
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
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA), .O_BITS(DWC_STREAM_WIDTH)) inst_dwc_buff_mm_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2_2), .m_axis(axis_s3_2));

// Thr 2
p0_Thresholding_rtl_2_axi_wrapper inst_thr_2 (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),

    .s_axilite_AWVALID(1'b0),
    .s_axilite_AWREADY(),
    .s_axilite_AWADDR(0),
    .s_axilite_WVALID(0),
    .s_axilite_WREADY(),
    .s_axilite_WDATA(0),
    .s_axilite_WSTRB(0),
    .s_axilite_BVALID(),
    .s_axilite_BREADY(1'b1),
    .s_axilite_BRESP(),
    .s_axilite_ARVALID(1'b0),
    .s_axilite_ARREADY(),
    .s_axilite_ARADDR(0),
    .s_axilite_RVALID(),
    .s_axilite_RREADY(1'b1),
    .s_axilite_RDATA(),
    .s_axilite_RRESP(),
    
    .in0_V_TDATA (axis_s3_2.tdata),
    .in0_V_TVALID(axis_s3_2.tvalid),
    .in0_V_TREADY(axis_s3_2.tready),
    
    .out_V_TDATA (axis_s4_2.tdata),
    .out_V_TVALID(axis_s4_2.tvalid),
    .out_V_TREADY(axis_s4_2.tready)
);

dwc_buff_top #(.I_BITS(PE_THR*ACTIVATION_WIDTH), .O_BITS(PE*ACTIVATION_WIDTH)) inst_dwc_buff_thr_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s4_2), .m_axis(m_axis_2));

endmodule