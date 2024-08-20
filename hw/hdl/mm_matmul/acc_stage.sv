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
module acc_stage #(
    parameter                                               PE,
    parameter                                               ACCU_WIDTH,
    parameter                                               TH
) (
    input  logic                                            clk,
    input  logic                                            rst,
    input  logic                                            en,

    input  logic [PE-1:0][ACCU_WIDTH-1:0]                   idat,
    input  logic                                            ival,
    input  logic                                            ilast,

    output logic [PE-1:0][ACCU_WIDTH-1:0]                   o_acc,
    input  logic                                            inc_acc,

    output logic [PE-1:0][ACCU_WIDTH-1:0]                   odat,
    output logic                                            oval
);

(* ram_style = "distributed" *) logic [PE-1:0][TH-1:0][ACCU_WIDTH-1:0] acc;

localparam integer TILE_IDX = $clog2(TH);

logic [TILE_IDX-1:0] rd_pntr;
logic [TILE_IDX-1:0] wr_pntr;

always_ff @(posedge clk) begin
    if(rst) begin
        acc <= 0;
        odat <= 0;
        oval <= 1'b0;

        rd_pntr <= 0;
        wr_pntr <= 0;
    end
    else begin
        if(en) begin
            rd_pntr <= inc_acc ? rd_pntr + 1 : rd_pntr;
            
            if(ival) begin
                if(wr_pntr == TH-1) begin
                    wr_pntr <= 0;
                end
                else begin
                    wr_pntr <= wr_pntr + 1;
                end

                for(int i = 0; i < PE; i++) begin
                    acc[i][wr_pntr] <= ilast ? 0 : idat[i];
                    odat[i] <= idat[i];
                end
                
                oval <= ival && ilast;
            end
        end 
    end
end

for(genvar i = 0; i < PE; i++) begin
    assign o_acc[i] = acc[i][rd_pntr];
end

endmodule