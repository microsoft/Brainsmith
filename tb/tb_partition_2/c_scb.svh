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
  mailbox drv2scb_a_0;
  mailbox drv2scb_a_1;
  mailbox mon2scb_c;

  // Completion
  event done;
  
  // Fail flag
  integer fail;

  // Matrices
  logic [MH*MW-1:0][ACTIVATION_WIDTH-1:0] mtrx_c = 0;

  // I/O ref
  string refio_c;
  
  //
  // C-tor
  //
  function new(string name,  mailbox drv2scb_a_0, mailbox drv2scb_a_1, mailbox mon2scb_c, string refio_c);
    this.name = name;
    this.drv2scb_a_0 = drv2scb_a_0;
    this.drv2scb_a_1 = drv2scb_a_1;
    this.mon2scb_c   = mon2scb_c;
    this.refio_c = refio_c;
  endfunction
  
  //
  // Run
  //

  task run;
    parameter N_TRS_OUT = (MH * MW) / PE_P2;

    int fh;
    int fo;
    logic [ACTIVATION_WIDTH-1:0] tmp_val;
    int j = 0;

    c_trs #(.DATA_BITS(PE_P2*ACTIVATION_WIDTH)) trs_mon_c;
    fail = 0;

    // Output
    for(int l = 0; l < N_LOOPS; l++) begin
      for(int i = 0; i < N_TRS_OUT; i++) begin
        mon2scb_c.get(trs_mon_c);

        for(int pe = 0; pe < PE_P0; pe++) begin
          mtrx_c[i*PE_P2+pe] = trs_mon_c.tdata[pe*ACTIVATION_WIDTH+:ACTIVATION_WIDTH];
        end
      end

      // Interval
      $display("Interval %d latency %d", l, (integer($time) / CLK_PERIOD - 5));

      // Check output
      fh = $fopen(refio_c, "r");
      if (fh == 0) begin
        $fatal("Failed to open output file for reading");
      end
      
      fo = $fopen("/scratch/dkorolij/ms-bert/scripts_finn/refio/output.txt", "w");
      if (fo == 0) begin
        $fatal("Failed to open output file for reading");
      end
      
      j = 0;
      while (!$feof(fh)) begin
        // Read hex value from file
        if ($fscanf(fh, "%h", tmp_val) == 1) begin
          if(tmp_val != mtrx_c[j]) begin
            fail = 1;
            $fwrite(fo,"-----------NO MATCH %d - DUT: %x, RD: %x\n", j, mtrx_c[j], tmp_val);
          end
          else begin
            $fwrite(fo,"MATCH %d - DUT: %x, RD: %x\n", j, mtrx_c[j], tmp_val);      
          end
          j++;
        end
      end

      $fclose(fh);
      $fclose(fo);

      if(fail) begin
        $display("\nERR:  Results do not match .onnx graph!");
      end
      else begin
        $display("\nResults match .onnx graph!");
      end

    end
    
    $display("");
    -> done;
  endtask
  
endclass

`endif