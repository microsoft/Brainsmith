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
`timescale 1ns / 1ps

import cTypes::*;

`include "c_env.svh"

task delay(input integer n_clk_prds);
    #(n_clk_prds*CLK_PERIOD);
endtask

`define VERILATOR

module tb_top;

    logic clk = 1'b1;
    logic clk_dp = 1'b1;
    logic resetn = 1'b0;

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;
    always #(CLK_PERIOD/4) clk_dp = ~clk_dp;
    
    // Reset Generation
    initial begin
        resetn = 1'b0;
        #(RST_PERIOD) resetn = 1'b1;
    end

    // Interfaces and drivers
    AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) matrix_a (clk);
    AXI4S #(.AXI4S_DATA_BITS(PE_SHUFFLE_A*ACTIVATION_WIDTH)) matrix_c (clk);

    c_env matrix_drv = new(matrix_a, matrix_c, "", "");

    //
    // DUTs
    //
    
    replay_buff_tile #(
        .SF(2),
        .NF(3),
        .DATA_BITS(32),
        .N_RPLYS(4)
    ) inst_replay_buf (
        .clk            (clk),
        .rst            (~resetn),
        
        .idat           (matrix_a.tdata),
        .ivld           (matrix_a.tvalid),
        .irdy           (matrix_a.tready),

        .odat           (matrix_c.tdata),
        .ovld           (matrix_c.tvalid),
        .ordy           (matrix_c.tready)
    );
    

    // Stream threads
    task env_threads();
        fork
            matrix_drv.run();
        join_any
    endtask
    
    // Stream completion
    task env_done();
        wait(matrix_drv.done.triggered);
    endtask
    
    // 
    initial begin
        env_threads();
        env_done();
        $display("Test done!");
        $finish;
    end

    // Dump
    initial begin
        $dumpfile("dump.vcd"); $dumpvars;
    end

endmodule