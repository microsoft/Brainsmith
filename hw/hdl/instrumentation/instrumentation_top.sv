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

module instrumentation_top (
    input  logic                    aclk,
    input  logic                    aresetn,

    // Data
    AXI4S.slave                         s_finn_out [N_STREMS_IN],
    AXI4S.master                        m_finn_in [N_STREAMS_OUT],

    // Control
    input  logic [31:0]             s_n_beats_in, // in bytes
    input  logic [15:0]             s_n_runs_in,
    input  logic [31:0]             s_n_beats_out, // out bytes
    input  logic [15:0]             s_n_runs_out,
    input  logic [31:0]             s_seed,
    input  logic                    s_start,

    output logic [63:0]             m_latency_out,
    output logic [63:0]             m_interval_out,
    output logic [15:0]             m_checksum_out,
    output logic                    m_ready
);

//
// Reg I/O
//
logic [31:0] n_beats_in;
logic [15:0] n_runs_in;
logic [31:0] n_beats_out;
logic [15:0] n_runs_out;
logic [31:0] seed;
logic start;
logic started_gen;
logic started_mon;
logic start_edge;

logic [63:0] latency_out;
logic [63:0] interval_out;
logic [15:0] checksum_out;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        n_beats_in <= 0;
        n_runs_in <= 0;
        n_beats_out <= 0;
        n_runs_out <= 0;
        seed <= 0;
        start <= 1'b0;
        
        m_latency_out <= 0;
        m_interval_out <= 0;
        m_checksum_out <= 0;
        m_ready <= 1'b0;
    end
    else begin
        n_beats_in <= s_n_beats_in;
        n_runs_in <= s_n_runs_in;
        n_beats_out <= s_n_beats_out;
        n_runs_out <= s_n_runs_out;
        seed <= s_seed;
        start <= s_start;

        m_latency_out <= latency_out;
        m_interval_out <= interval_out;
        m_checksum_out <= checksum_out;
        m_ready <= ~started_mon;
    end
end

AXI4S #(.AXI4S_DATA_BITS(32)) finn_out ();
AXI4S #(.AXI4S_DATA_BITS(32)) finn_in ();

axis_reg_32 inst_axis_reg_finn_out (
    .aclk                   (aclk),
    .aresetn                (aresetn),

    .s_axis_tvalid          (s_finn_out.tvalid),
    .s_axis_tready          (s_finn_out.tready),
    .s_axis_tdata           (s_finn_out.tdata),

    .m_axis_tvalid          (finn_out.tvalid),
    .m_axis_tready          (finn_out.tready),
    .m_axis_tdata           (finn_out.tdata)
);

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

//
// Checksum
//
logic [15:0] sum;
logic [15:0] csum;

always_comb begin
    sum = finn_out.tdata[31:16] + finn_out.tdata[15:0];
    if (sum & 32'hffff0000) begin
        csum = sum;
        csum = csum + 1;
    end else
        csum = sum;
end

//
// Measurements
//
logic [63:0] interval;
logic [31:0] cnt_data_latency;
logic [31:0] cnt_data_interval;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        interval <= 0;
        cnt_data_latency <= 0;
        cnt_data_interval <= 0;
        latency_out <= 0;
        interval_out <= 0;
        checksum_out <= 0;

        started_mon <= 1'b0;
    end
    else begin
        if(started_mon) begin
            latency_out <= latency_out + 1;
            interval <= interval + 1;

            if(finn_out.tvalid & finn_out.tready) begin
                checksum_out <= checksum_out + csum;

                if(cnt_data_latency == ((n_beats_out << n_runs_out) - 1)) begin
                    started_mon <= 1'b0;
                    cnt_data_latency <= 0;
                end
                else begin
                    cnt_data_latency <= cnt_data_latency + 1;
                end

                if(cnt_data_interval == (n_beats_out - 1)) begin
                    cnt_data_interval <= 0;

                    interval_out <= interval + 1;
                    interval <= 0;
                end
                else begin
                    cnt_data_interval <= cnt_data_interval + 1;
                end
            end
        end
        else begin
            if(start & ~start_edge) begin
                started_mon <= 1'b1;
                latency_out <= 0;
                interval_out <= 0;
                checksum_out <= 0;
            end
            interval <= 0;
        end
    end
end

assign finn_out.tready = started_mon;

endmodule

