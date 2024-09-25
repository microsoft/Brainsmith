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
import mvauDynTypes::*;

// AXIS Scoreboard
class c_scb;

  // Name
  string name;
   
  // Mailbox handle
  mailbox mon2scb;
  mailbox drv2scb_a;

  // Completion
  event done;
  
  // Fail flag
  integer fail;

  // Matrices
  logic [MH_TMP-1:0][MW_TMP-1:0][ACTIVATION_WIDTH-1:0] mtrx_a = 0;
  logic [MW_TMP-1:0][MW_TMP-1:0][ACTIVATION_WIDTH-1:0] mtrx_b = 0;
  logic [MH_TMP-1:0][MW_TMP-1:0][ACCU_WIDTH-1:0] mtrx_c_cmptd = 0;
  logic [MH_TMP-1:0][MW_TMP-1:0][ACCU_WIDTH-1:0] mtrx_c_dut = 0;
  
  //
  // C-tor
  //
  function new(string name, mailbox mon2scb, mailbox drv2scb_a);
    this.name = name;
    this.mon2scb = mon2scb;
    this.drv2scb_a = drv2scb_a;
  endfunction
  
  //
  // Run
  //

  task run;
  int fh;
  logic [ACCU_WIDTH-1:0] tmp_val;
    c_trs #(.DATA_BITS(A_BITS_BA)) trs_drv_a;
    c_trs #(.DATA_BITS(C_BITS_BA)) trs_mon;
    fail = 0;

    

    // A
    $display("MATRIX A:");
    for(int i = 0; i < MH_TMP; i++) begin
      for(int j = 0; j < MW_TMP / SIMD_TMP; j++) begin
        drv2scb_a.get(trs_drv_a);
        for(int k = 0; k < SIMD_TMP; k++) begin
          mtrx_a[i][j*SIMD_TMP+k] = trs_drv_a.tdata[k*ACTIVATION_WIDTH+:ACTIVATION_WIDTH];
          $write("%x ", mtrx_a[i][j*SIMD_TMP+k]);
        end
      end
      $display("");
    end

    fh = $fopen("/scratch/dkorolij/tmp_files/random_hex_output.txt", "r");
    if (fh == 0) begin
      $fatal("Failed to open output file for reading");
    end
    
    // B
    $display("MATRIX B:");
    for(int i = 0; i < MW_TMP; i++) begin
      for(int j = 0; j < MW_TMP; j++) begin
        if ($fscanf(fh, "%h", tmp_val) == 1) begin
          mtrx_b[i][j] = tmp_val;
          $write("%x ", mtrx_b[i][j]);
        end
      end
      $display("");
    end

    // Compute
    $display("MATRIX CMPTD:");
    for(int i = 0; i < MH_TMP; i++) begin
      for(int j = 0; j < MW_TMP; j++) begin
        for(int k = 0; k < MW_TMP; k++) begin
          mtrx_c_cmptd[i][j] = $signed(mtrx_c_cmptd[i][j]) + $signed(mtrx_a[i][k]) * $signed(mtrx_b[k][j]);
        end
        $write("%x ", $signed(mtrx_c_cmptd[i][j]));
      end
      $display("");
    end

    // OUT
    $display("MATRIX DUT:");
    for(int i = 0; i < MH_TMP; i++) begin
      for(int j = 0; j < MW_TMP / PE_TMP; j++) begin
        mon2scb.get(trs_mon);
        for(int k = 0; k < PE_TMP; k++) begin
          mtrx_c_dut[i][j*PE_TMP+k] = trs_mon.tdata[k*ACCU_WIDTH+:ACCU_WIDTH];
          $write("%x ", $signed(mtrx_c_dut[i][j*PE_TMP+k]));
        end
      end
      $display("");
    end

    // Check
    for(int i = 0; i < MH_TMP; i++) begin
      for(int j = 0; j < MW_TMP; j++) begin
        if(mtrx_c_cmptd[i][j] != mtrx_c_dut[i][j]) begin
          fail = 1;
        end
      end
    end

    if(fail) begin
      $display("\nERR:  Results do not match .onnx graph!");
    end
    else begin
      $display("\nResults match .onnx graph!");
    end
    
    $display("");
    -> done;
  endtask
  
endclass

`endif