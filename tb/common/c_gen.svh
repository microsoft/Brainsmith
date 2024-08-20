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

`ifndef __C_GEN__
`define __C_GEN__

import cTypes::*;

// AXIS Generator
class c_gen #(
    parameter integer DATA_BITS = 8
);
  
  // Name
  string name;

  // Type
  int read_stim;
  string rd_file;

  // Send to driver (mailbox)
  mailbox gen2drv;

  // Params
  integer n_trs;
  integer delay;

  // Completion
  event done;
  
  //
  // C-tor
  //
  function new(string name, mailbox gen2drv, string rd_file, integer n_trs, integer delay);
    this.name = name;
    this.gen2drv = gen2drv;
    this.rd_file = rd_file;
    this.n_trs = n_trs;
    this.delay = delay;

    if(rd_file != "")
      read_stim = 1;
    else
      read_stim = 0;
  endfunction
  
  //
  // Run
  //
  
  task run();
    c_trs #(.DATA_BITS(DATA_BITS)) trs;

    int fh;
    logic [ACTIVATION_WIDTH-1:0] tmp_val;
    logic [DATA_BITS-1:0] tmp_data;
    int i = 0, j = 0;

    #(delay*CLK_PERIOD);

    
    if(!read_stim) begin
      // Stimulus generated
      for(int i = 0; i < n_trs; i++) begin
        trs = new();
        if(!trs.randomize()) $fatal("ERR:  Generator randomization failed");
        gen2drv.put(trs);
      end
    end
    else begin
      // Stimulus read
      for(int l = 0; l < N_LOOPS; l++) begin
        fh = $fopen(rd_file, "r");
        if (fh == 0) begin
            $fatal("Failed to open file '%s' for reading", rd_file);
        end

        while (!$feof(fh)) begin
          // Read hex value from file
          if ($fscanf(fh, "%h", tmp_val) == 1) begin
              tmp_data[i*ACTIVATION_WIDTH+:ACTIVATION_WIDTH] = tmp_val;
              i++; j++;

              if(i*ACTIVATION_WIDTH == DATA_BITS) begin
                trs = new();
                trs.tdata = tmp_data;
                i = 0;
                gen2drv.put(trs);
              end
          end
        end
        $fclose(fh);
      end
    end
     
    -> done;
  endtask

endclass

`endif