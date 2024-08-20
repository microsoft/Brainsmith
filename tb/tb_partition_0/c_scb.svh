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

`ifndef __C_SCB__
`define __C_SCB__

import cTypes::*;

// AXIS Scoreboard
class c_scb;

  // Name
  string name;
   
  // Mailbox handle
  mailbox drv2scb_a;
  mailbox mon2scb_b_0;
  mailbox mon2scb_b_1;
  mailbox mon2scb_b_2;

  // Completion
  event done;
  
  // Fail flag
  integer fail;

  // Matrices
  logic [2:0][N_LOOPS*MH*MW-1:0][ACTIVATION_WIDTH-1:0] mtrx_b = 0;

  // I/O ref
  string refio_b_0;
  string refio_b_1;
  string refio_b_2;
  
  //
  // C-tor
  //
  function new(string name,  mailbox drv2scb_a, mailbox mon2scb_b_0, mailbox mon2scb_b_1, mailbox mon2scb_b_2, 
    input string refio_b_0, input string refio_b_1, input string refio_b_2);
    this.name = name;
    this.drv2scb_a = drv2scb_a;
    this.mon2scb_b_0 = mon2scb_b_0;
    this.mon2scb_b_1 = mon2scb_b_1;
    this.mon2scb_b_2 = mon2scb_b_2;
    this.refio_b_0 = refio_b_0;
    this.refio_b_1 = refio_b_1;
    this.refio_b_2 = refio_b_2;
  endfunction
  
  //
  // Run
  //

  task run;
    //
    parameter N_TRS_OUT = (MH * MW) / PE_P0;
    
    int fh [2:0];
    logic [ACTIVATION_WIDTH-1:0] tmp_val;
    int j = 0;

    c_trs #(.DATA_BITS(PE_P0*ACTIVATION_WIDTH)) trs_mon_b_0;
    c_trs #(.DATA_BITS(PE_P0*ACTIVATION_WIDTH)) trs_mon_b_1;
    c_trs #(.DATA_BITS(PE_P0*ACTIVATION_WIDTH)) trs_mon_b_2;
    fail = 0;
    
    // Output
    for(int l = 0; l < N_LOOPS; l++) begin

      for(int i = 0; i < N_TRS_OUT; i++) begin
        mon2scb_b_0.get(trs_mon_b_0);
        mon2scb_b_1.get(trs_mon_b_1);
        mon2scb_b_2.get(trs_mon_b_2);

        for(int pe = 0; pe < PE_P0; pe++) begin
          mtrx_b[0][i*PE_P0+pe] = trs_mon_b_0.tdata[pe*ACTIVATION_WIDTH+:ACTIVATION_WIDTH];
          mtrx_b[1][i*PE_P0+pe] = trs_mon_b_1.tdata[pe*ACTIVATION_WIDTH+:ACTIVATION_WIDTH];
          mtrx_b[2][i*PE_P0+pe] = trs_mon_b_2.tdata[pe*ACTIVATION_WIDTH+:ACTIVATION_WIDTH];
        end
      end

      // Interval
      $display("Interval %d latency %0d", l, (integer($time) / CLK_PERIOD - 5));

      // Check output
      fh[0] = $fopen(refio_b_0, "r");
      fh[1] = $fopen(refio_b_1, "r");
      fh[2] = $fopen(refio_b_2, "r");

      for(int i = 0; i < 3; i++) begin
        if (fh[i] == 0) begin
          $fatal("Failed to open output file '%d' for reading", i);
        end
      
        j = 0;
        while (!$feof(fh[i])) begin
          // Read hex value from file
          if ($fscanf(fh[i], "%h", tmp_val) == 1) begin
            if(tmp_val != mtrx_b[i][j]) begin
              fail = 1;
            end
            j++;
          end
        end

        $fclose(fh[i]);
      end

      if(fail) begin
        $display("ERR:  Results do not match .onnx graph!");
      end
      else begin
        $display("Results match .onnx graph!");
      end
  
    end
  
    $display("");
    -> done;
  endtask
  
endclass

`endif