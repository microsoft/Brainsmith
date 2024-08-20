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

/*******************************************************************************
 * @brief	v80 FINN instrumentation
 * @author	Dario Korolija <dario.korolija@amd.com>
 *******************************************************************************/

`timescale 1ns / 1ps

module instrumentation_producer (
    input  logic                    aclk,
    input  logic                    aresetn,

    // Data
    AXI4S.m                         m_finn_in,

    // Control
    input  logic [31:0]             s_n_beats_in, // in bytes
    input  logic [15:0]             s_n_runs_in, // pow2
    input  logic [31:0]             s_seed,
    input  logic                    s_start,
    output logic                    m_ready
);

//
// Reg I/O
//
logic [31:0] n_beats_in;
logic [15:0] n_runs_in;
logic [31:0] seed;
logic start;
logic started_gen;
logic start_edge;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        n_beats_in <= 0;
        n_runs_in <= 0;
        seed <= 0;
        start <= 1'b0;
        
        m_ready <= 1'b0;
    end
    else begin
        n_beats_in <= s_n_beats_in;
        n_runs_in <= s_n_runs_in;
        seed <= s_seed;
        start <= s_start;

        m_ready <= ~started_gen;
    end
end

AXI4S #(.AXI4S_DATA_BITS(32)) finn_in ();

axis_reg_32 inst_axis_reg_finn_in (
    .aclk                   (aclk),
    .aresetn                (aresetn),

    .s_axis_tvalid          (finn_in.tvalid),
    .s_axis_tready          (finn_in.tready),
    .s_axis_tdata           (finn_in.tdata),

    .m_axis_tvalid          (m_finn_in.tvalid),
    .m_axis_tready          (m_finn_in.tready),
    .m_axis_tdata           (m_finn_in.tdata)
);

//
// LFSR
//
parameter N_LFSR = 4;
parameter LFSR_START_STATE = 3223334;

logic [31:0] curr_data = 0, next_data;
logic [N_LFSR-1:0][30:0] curr_state = 0, next_state;
logic [31:0] cnt_data_finn_in;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        start_edge <= 1'b0;
        started_gen <= 1'b0;

        curr_data <= 0;
        for(int i = 0; i < N_LFSR; i++)
            curr_state[i] <= LFSR_START_STATE;
        cnt_data_finn_in <= 0;
    end
    else begin
        start_edge <= start;
        
        if(started_gen) begin
            curr_data <= finn_in.tready ? next_data : curr_data;
            for(int i = 0; i < N_LFSR; i++)
                curr_state[i] <= finn_in.tready ? next_state[i] : curr_state[i];
        
            if(finn_in.tready) begin
                if(cnt_data_finn_in == ((n_beats_in << n_runs_in) - 1)) begin
                    started_gen <= 1'b0;
                    cnt_data_finn_in <= 0;
                end
                else begin
                    cnt_data_finn_in <= cnt_data_finn_in + 1;
                end
            end
        end
        else begin
            if(start & ~start_edge) begin
                started_gen <= 1'b1;
                curr_data <= seed;
                for(int i = 0; i < N_LFSR; i++)
                    curr_state[i] <= LFSR_START_STATE;
            end
        end
    end
end

assign finn_in.tdata = curr_data;
assign finn_in.tvalid = started_gen;

for(genvar i = 0; i < N_LFSR; i++) begin
    lfsr #(
        .DATA_WIDTH(8)
    ) inst_lfsr (
        .data_in(curr_data[i*8+:8]),
        .state_in(curr_state[i]),
        .data_out(next_data[i*8+:8]),
        .state_out(next_state[i])
    );
end

endmodule

