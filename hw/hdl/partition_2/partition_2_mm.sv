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

module partition_2_mm #(
    parameter integer                               SIMD_0 = 48,
    parameter integer                               SIMD_1 = 4,
    parameter integer                               PE = 4,
    
    // MM
    parameter integer                               MH_MM_0 = 384,
    parameter integer                               MW_MM_0 = 384,
    parameter integer                               SIMD_MM_0 = 48,
    parameter integer                               PE_MM_0 = 32,
    parameter integer                               TH_MM_0 = 2*PE_MM_0,
    parameter integer                               PE_THR_0 = 4,

    parameter integer                               MH_MM_1 = 384,
    parameter integer                               MW_MM_1 = 1536,
    parameter integer                               SIMD_MM_1 = 96,
    parameter integer                               PE_MM_1 = 64,
    parameter integer                               TH_MM_1 = 2*PE_MM_1,
    parameter integer                               PE_THR_1 = 16,

    parameter integer                               MH_MM_2 = 1536,
    parameter integer                               MW_MM_2 = 384,
    parameter integer                               SIMD_MM_2 = 96,
    parameter integer                               PE_MM_2 = 64,
    parameter integer                               TH_MM_2 = 2*PE_MM_2,
    parameter integer                               PE_THR_2 = 4,
    
    // Init
    parameter                                       INIT_FILE_0 = "",
    parameter                                       INIT_FILE_1 = "",
    parameter                                       INIT_FILE_2 = "",

    // Config
    parameter integer                               ACTIVATION_WIDTH = 8,
    parameter integer                               ACCU_WIDTH_0 = 22,//2*ACTIVATION_WIDTH+$clog2(MH_MM_0),
    parameter integer                               ACCU_WIDTH_1 = 22,//2*ACTIVATION_WIDTH+$clog2(MH_MM_1),
    parameter integer                               ACCU_WIDTH_2 = 23,//2*ACTIVATION_WIDTH+$clog2(MH_MM_2),
    parameter integer                               PUMPED_COMPUTE = 1,
    parameter integer                               MM_KERNEL = 1
) (
    input  logic                                    ap_clk,
    input  logic                                    ap_clk2x,
    input  logic                                    ap_rst_n,
    
    AXI4S.slave                                        s_axis_0,
    AXI4S.slave                                        s_axis_1,
    AXI4S.master                                        m_axis_0
);

//
// Params
//

localparam int unsigned  MM_STREAM_WIDTH_0  = PE_MM_0 * ACCU_WIDTH_0;
localparam int unsigned  MM_STREAM_WIDTH_BA_0  = (MM_STREAM_WIDTH_0 + 7)/8 * 8;
localparam int unsigned  DWC_STREAM_WIDTH_0  = PE_THR_0 * ACCU_WIDTH_0;
localparam int unsigned  MM_STREAM_WIDTH_1  = PE_MM_1 * ACCU_WIDTH_1;
localparam int unsigned  MM_STREAM_WIDTH_BA_1  = (MM_STREAM_WIDTH_1 + 7)/8 * 8;
localparam int unsigned  DWC_STREAM_WIDTH_1  = PE_THR_1 * ACCU_WIDTH_1;
localparam int unsigned  MM_STREAM_WIDTH_2  = PE_MM_2 * ACCU_WIDTH_2;
localparam int unsigned  MM_STREAM_WIDTH_BA_2  = (MM_STREAM_WIDTH_2 + 7)/8 * 8;
localparam int unsigned  DWC_STREAM_WIDTH_2  = PE_THR_2 * ACCU_WIDTH_2;
localparam int unsigned  ADD_STREAM_WIDTH_0 = PE * (ACTIVATION_WIDTH + 1);
localparam int unsigned  ADD_STREAM_WIDTH_BA_0 = (ADD_STREAM_WIDTH_0 + 7)/8 * 8;
localparam int unsigned  ADD_STREAM_WIDTH_1 = PE * (ACTIVATION_WIDTH + 2);
localparam int unsigned  ADD_STREAM_WIDTH_BA_1 = (ADD_STREAM_WIDTH_1 + 7)/8 * 8;

//
// Signals
//


AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA_0)) axis_s0 ();
AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH_0)) axis_s0_buf ();

AXI4S #(.AXI4S_DATA_BITS(PE_THR_0*ACTIVATION_WIDTH)) axis_s1 ();
AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s1_buf ();

AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s2 ();
AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s2_buf ();

AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s3_0 ();
AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s3_1 ();
AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s3_0_buf ();
AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_0)) axis_s3_1_buf ();

AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s4 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM_1*ACTIVATION_WIDTH)) axis_s4_buf ();

AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA_1)) axis_s5 ();
AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH_1)) axis_s5_buf ();

AXI4S #(.AXI4S_DATA_BITS(PE_THR_1*ACTIVATION_WIDTH)) axis_s6 ();
AXI4S #(.AXI4S_DATA_BITS(SIMD_MM_2*ACTIVATION_WIDTH)) axis_s6_buf ();

AXI4S #(.AXI4S_DATA_BITS(MM_STREAM_WIDTH_BA_2)) axis_s7 ();
AXI4S #(.AXI4S_DATA_BITS(DWC_STREAM_WIDTH_2)) axis_s7_buf ();

AXI4S #(.AXI4S_DATA_BITS(PE_THR_2*ACTIVATION_WIDTH)) axis_s8 ();
AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s8_buf ();

AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_1)) axis_s9 ();
AXI4S #(.AXI4S_DATA_BITS(ADD_STREAM_WIDTH_BA_1)) axis_s9_buf ();

AXI4S #(.AXI4S_DATA_BITS(PE*ACTIVATION_WIDTH)) axis_s10 ();

// MatMul 0
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MH_MM_0),
        .MW(MW_MM_0),
        .PE(PE_MM_0),
        .SIMD(SIMD_MM_0),
        .TH(TH_MM_0),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_0),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_0)
    ) inst_matmul_0 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (s_axis_0.tdata),
        .s_axis_a_tvalid(s_axis_0.tvalid),
        .s_axis_a_tready(s_axis_0.tready),

        .m_axis_c_tdata (axis_s0.tdata),
        .m_axis_c_tvalid(axis_s0.tvalid),
        .m_axis_c_tready(axis_s0.tready)
    );
end
else begin
    mv_matmul_sta #(
        .MH(MH_MM_0),
        .MW(MW_MM_0),
        .PE(PE_MM_0),
        .SIMD(SIMD_MM_0),
    
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_0),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_0)
    ) inst_matmul_0 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (s_axis_0.tdata),
        .s_axis_a_tvalid(s_axis_0.tvalid),
        .s_axis_a_tready(s_axis_0.tready),

        .m_axis_c_tdata (axis_s0.tdata),
        .m_axis_c_tvalid(axis_s0.tvalid),
        .m_axis_c_tready(axis_s0.tready)
    );
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA_0), .O_BITS(DWC_STREAM_WIDTH_0)) inst_dwc_mm_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s0), .m_axis(axis_s0_buf));

// Thr 0
p2_Thresholding_rtl_0_axi_wrapper inst_thr_0 (
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
    
    .in0_V_TDATA (axis_s0_buf.tdata),
    .in0_V_TVALID(axis_s0_buf.tvalid),
    .in0_V_TREADY(axis_s0_buf.tready),
    
    .out_V_TDATA (axis_s1.tdata),
    .out_V_TVALID(axis_s1.tvalid),
    .out_V_TREADY(axis_s1.tready)
);

dwc_buff_top #(.I_BITS(PE_THR_0*ACTIVATION_WIDTH), .O_BITS(PE*ACTIVATION_WIDTH)) inst_dwc_thr_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s1), .m_axis(axis_s1_buf));

// Add 0
add_sta #(
    .ACTIVATION_WIDTH_0(ACTIVATION_WIDTH),
    .ACTIVATION_WIDTH_1(ACTIVATION_WIDTH),
    .PE(PE)  
) inst_add_0 (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),

    .s_axis_0_tvalid(axis_s1_buf.tvalid),
    .s_axis_0_tready(axis_s1_buf.tready),
    .s_axis_0_tdata (axis_s1_buf.tdata),

    .s_axis_1_tvalid(s_axis_1.tvalid),
    .s_axis_1_tready(s_axis_1.tready),
    .s_axis_1_tdata (s_axis_1.tdata),

    .m_axis_tvalid(axis_s2.tvalid),
    .m_axis_tready(axis_s2.tready),
    .m_axis_tdata (axis_s2.tdata)
);

dwc_buff_top #(.I_BITS(ADD_STREAM_WIDTH_BA_0), .O_BITS(ADD_STREAM_WIDTH_BA_0)) inst_dwc_add_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s2), .m_axis(axis_s2_buf));

// Broadcast
broadcast #(
    .M_COUNT(2),
    .DATA_WIDTH(ADD_STREAM_WIDTH_BA_0)
) inst_bcast (
    .ap_clk             (ap_clk),
    .ap_rst_n           (ap_rst_n),

    .s_axis_tvalid      (axis_s2_buf.tvalid),
    .s_axis_tready      (axis_s2_buf.tready),
    .s_axis_tdata       (axis_s2_buf.tdata),

    .m_axis_tvalid      ({axis_s3_1.tvalid, axis_s3_0.tvalid}),
    .m_axis_tready      ({axis_s3_1.tready, axis_s3_0.tready}),
    .m_axis_tdata       ({axis_s3_1.tdata,  axis_s3_0.tdata})
);

dwc_buff_top #(.I_BITS(ADD_STREAM_WIDTH_BA_0), .O_BITS(ADD_STREAM_WIDTH_BA_0)) inst_dwc_bcast_0 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s3_0), .m_axis(axis_s3_0_buf));
axis_buff_p2 inst_outer_buff (
    .s_axis_aclk        (ap_clk),
    .s_axis_aresetn     (ap_rst_n),
    
    .s_axis_tvalid      (axis_s3_1.tvalid),
    .s_axis_tready      (axis_s3_1.tready),
    .s_axis_tdata       (axis_s3_1.tdata),
    
    .m_axis_tvalid      (axis_s3_1_buf.tvalid),
    .m_axis_tready      (axis_s3_1_buf.tready),
    .m_axis_tdata       (axis_s3_1_buf.tdata)
);

// Thr 1
p2_Thresholding_rtl_1_axi_wrapper inst_thr_1 (
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
    
    .in0_V_TDATA (axis_s3_0_buf.tdata),
    .in0_V_TVALID(axis_s3_0_buf.tvalid),
    .in0_V_TREADY(axis_s3_0_buf.tready),
    
    .out_V_TDATA (axis_s4.tdata),
    .out_V_TVALID(axis_s4.tvalid),
    .out_V_TREADY(axis_s4.tready)
);

dwc_buff_top #(.I_BITS(PE*ACTIVATION_WIDTH), .O_BITS(SIMD_MM_1*ACTIVATION_WIDTH)) inst_dwc_thr_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s4), .m_axis(axis_s4_buf));

// MatMul 1
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MH_MM_1),
        .MW(MW_MM_1),
        .PE(PE_MM_1),
        .SIMD(SIMD_MM_1),
        .TH(TH_MM_1),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_1),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_1)
    ) inst_matmul_1 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (axis_s4_buf.tdata),
        .s_axis_a_tvalid(axis_s4_buf.tvalid),
        .s_axis_a_tready(axis_s4_buf.tready),

        .m_axis_c_tdata (axis_s5.tdata),
        .m_axis_c_tvalid(axis_s5.tvalid),
        .m_axis_c_tready(axis_s5.tready)
    );
end
else begin
    mv_matmul_sta #(
        .MH(MH_MM_1),
        .MW(MW_MM_1),
        .PE(PE_MM_1),
        .SIMD(SIMD_MM_1),
    
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_1),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_1)
    ) inst_matmul_1 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (axis_s4_buf.tdata),
        .s_axis_a_tvalid(axis_s4_buf.tvalid),
        .s_axis_a_tready(axis_s4_buf.tready),

        .m_axis_c_tdata (axis_s5.tdata),
        .m_axis_c_tvalid(axis_s5.tvalid),
        .m_axis_c_tready(axis_s5.tready)
    );
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA_1), .O_BITS(DWC_STREAM_WIDTH_1)) inst_dwc_mm_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s5), .m_axis(axis_s5_buf));

// Thr 2
p2_Thresholding_rtl_2_axi_wrapper inst_thr_2 (
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
    
    .in0_V_TDATA (axis_s5_buf.tdata),
    .in0_V_TVALID(axis_s5_buf.tvalid),
    .in0_V_TREADY(axis_s5_buf.tready),
    
    .out_V_TDATA (axis_s6.tdata),
    .out_V_TVALID(axis_s6.tvalid),
    .out_V_TREADY(axis_s6.tready)
);

dwc_buff_top #(.I_BITS(PE_THR_1*ACTIVATION_WIDTH), .O_BITS(SIMD_MM_2*ACTIVATION_WIDTH)) inst_dwc_thr_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s6), .m_axis(axis_s6_buf));

// MatMul 2
if(MM_KERNEL == 1) begin
    mm_matmul_sta #(
        .MH(MH_MM_2),
        .MW(MW_MM_2),
        .PE(PE_MM_2),
        .SIMD(SIMD_MM_2),
        .TH(TH_MM_2),

        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_2),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_2)
    ) inst_matmul_2 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (axis_s6_buf.tdata),
        .s_axis_a_tvalid(axis_s6_buf.tvalid),
        .s_axis_a_tready(axis_s6_buf.tready),

        .m_axis_c_tdata (axis_s7.tdata),
        .m_axis_c_tvalid(axis_s7.tvalid),
        .m_axis_c_tready(axis_s7.tready)
    );
end
else begin
    mv_matmul_sta #(
        .MH(MH_MM_2),
        .MW(MW_MM_2),
        .PE(PE_MM_2),
        .SIMD(SIMD_MM_2),
    
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH),
        .ACCU_WIDTH(ACCU_WIDTH_2),
        .PUMPED_COMPUTE(PUMPED_COMPUTE),
        .INIT_FILE(INIT_FILE_2)
    ) inst_matmul_2 (
        .ap_clk(ap_clk),
        .ap_clk2x(ap_clk2x),
        .ap_rst_n(ap_rst_n),

        .s_axis_a_tdata (axis_s6_buf.tdata),
        .s_axis_a_tvalid(axis_s6_buf.tvalid),
        .s_axis_a_tready(axis_s6_buf.tready),

        .m_axis_c_tdata (axis_s7.tdata),
        .m_axis_c_tvalid(axis_s7.tvalid),
        .m_axis_c_tready(axis_s7.tready)
    );
end

dwc_buff_top #(.I_BITS(MM_STREAM_WIDTH_BA_2), .O_BITS(DWC_STREAM_WIDTH_2)) inst_dwc_mm_2 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s7), .m_axis(axis_s7_buf));

// Thr 3
p2_Thresholding_rtl_3_axi_wrapper inst_thr_3 (
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
    
    .in0_V_TDATA (axis_s7_buf.tdata),
    .in0_V_TVALID(axis_s7_buf.tvalid),
    .in0_V_TREADY(axis_s7_buf.tready),
    
    .out_V_TDATA (axis_s8.tdata),
    .out_V_TVALID(axis_s8.tvalid),
    .out_V_TREADY(axis_s8.tready)
);

dwc_buff_top #(.I_BITS(PE_THR_2*ACTIVATION_WIDTH), .O_BITS(PE*ACTIVATION_WIDTH)) inst_dwc_thr_3 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s8), .m_axis(axis_s8_buf));

// Add 1
add_sta #(
    .ACTIVATION_WIDTH_0(ACTIVATION_WIDTH),
    .ACTIVATION_WIDTH_1(ACTIVATION_WIDTH+1),
    .PE(PE)
) inst_add_1 (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),

    .s_axis_0_tvalid(axis_s8_buf.tvalid),
    .s_axis_0_tready(axis_s8_buf.tready),
    .s_axis_0_tdata (axis_s8_buf.tdata),

    .s_axis_1_tvalid(axis_s3_1_buf.tvalid),
    .s_axis_1_tready(axis_s3_1_buf.tready),
    .s_axis_1_tdata (axis_s3_1_buf.tdata),

    .m_axis_tvalid(axis_s9.tvalid),
    .m_axis_tready(axis_s9.tready),
    .m_axis_tdata (axis_s9.tdata)
);

dwc_buff_top #(.I_BITS(ADD_STREAM_WIDTH_BA_1), .O_BITS(ADD_STREAM_WIDTH_BA_1)) inst_dwc_add_1 (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis(axis_s9), .m_axis(axis_s9_buf));

// Thr 4
p2_Thresholding_rtl_4_axi_wrapper inst_thr_4 (
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
    
    .in0_V_TDATA (axis_s9_buf.tdata),
    .in0_V_TVALID(axis_s9_buf.tvalid),
    .in0_V_TREADY(axis_s9_buf.tready),
    
    .out_V_TDATA (m_axis_0.tdata),
    .out_V_TVALID(m_axis_0.tvalid),
    .out_V_TREADY(m_axis_0.tready)
);

endmodule