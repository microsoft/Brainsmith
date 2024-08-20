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

`ifndef __C_ENV__
`define __C_ENV__

`include "c_trs.svh"
`include "c_gen.svh"
`include "c_drv.svh"
`include "c_mon.svh"
`include "c_scb.svh"

import cTypes::*;

// Environment
class c_env;

    // Instances
    c_gen #(.DATA_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) gen_a_0;
    c_drv #(.DATA_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) drv_a_0;
    c_gen #(.DATA_BITS(SIMD_P2_1*ACTIVATION_WIDTH)) gen_a_1;
    c_drv #(.DATA_BITS(SIMD_P2_1*ACTIVATION_WIDTH)) drv_a_1;
    c_mon #(.DATA_BITS(PE_P2*ACTIVATION_WIDTH)) mon_c;
    c_scb scb;

    // Mailboxes
    mailbox gen2drv_a_0;
    mailbox drv2scb_a_0;
    mailbox gen2drv_a_1;
    mailbox drv2scb_a_1;
    mailbox mon2scb_c;

    // Interface handle
    virtual AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) matrix_a_0;
    virtual AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_1*ACTIVATION_WIDTH)) matrix_a_1;
    virtual AXI4S #(.AXI4S_DATA_BITS(PE_P2*ACTIVATION_WIDTH)) matrix_c;

    // Completion
    event done;

    // I/O ref
    string refio_a_0;
    string refio_a_1;
    string refio_c;

    // 
    // C-tor
    //
    function new(
        virtual AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_0*ACTIVATION_WIDTH)) matrix_a_0, 
        virtual AXI4S #(.AXI4S_DATA_BITS(SIMD_P2_1*ACTIVATION_WIDTH)) matrix_a_1, 
        virtual AXI4S #(.AXI4S_DATA_BITS(PE_P2*ACTIVATION_WIDTH)) matrix_c,
        string refio_a_0,
        string refio_a_1,
        string refio_c
    );
        // Interface
        this.matrix_a_0 = matrix_a_0;
        this.matrix_a_1 = matrix_a_1;
        this.matrix_c   = matrix_c;
        this.refio_a_0 = refio_a_0;
        this.refio_a_1 = refio_a_1;
        this.refio_c = refio_c;

        // Mailbox
        gen2drv_a_0 = new();
        drv2scb_a_0 = new();
        gen2drv_a_1 = new();
        drv2scb_a_1 = new();
        mon2scb_c = new();

        // Env
        gen_a_0 = new("A_0", gen2drv_a_0, refio_a_0, (N_LOOPS * MH * MW) / SIMD_P2_0, 0);
        gen_a_1 = new("A_1", gen2drv_a_1, refio_a_1, (N_LOOPS * MH * MW) / SIMD_P2_1, 0);    
        drv_a_0 = new("A_0", matrix_a_0, gen2drv_a_0, drv2scb_a_0);
        drv_a_1 = new("A_1", matrix_a_1, gen2drv_a_1, drv2scb_a_1);
        mon_c = new("C", matrix_c, mon2scb_c);
        scb = new("SCB", drv2scb_a_0, drv2scb_a_1, mon2scb_c, refio_c);
    endfunction

    // 
    // Reset
    //
    task reset();
        drv_a_0.reset_m();
        drv_a_1.reset_m();
        mon_c.reset_s();
        #(AST_PERIOD);
    endtask

    //
    // Run
    //
    task env_threads();
        fork
            gen_a_0.run();
            drv_a_0.run();
            gen_a_1.run();
            drv_a_1.run();
            mon_c.run();
            scb.run();
        join_any
    endtask

    //
    // Finish
    //
    task env_done();
        wait(gen_a_0.done.triggered);
        wait(gen_a_1.done.triggered);
        wait(scb.done.triggered);
    endtask
    
    //
    // Run
    //
    task run;
        reset();
        env_threads();
        env_done();
        if(scb.fail == 0) begin 
            $display("Run completed succesffully!");
            $display("Overall latency %d", (integer($time) / CLK_PERIOD - 5));
            $display("");
        end
        else begin
            $display("ERR:  Run failed");
            $display("");
        end
        -> done;
    endtask

endclass

`endif