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

package mvauDynTypes;

    import cTypes::*;

    // Params
    parameter integer ACCU_WIDTH = 2*ACTIVATION_WIDTH+$clog2(H_SIZE);
    parameter integer A_BITS = SIMD_P1_0 * ACTIVATION_WIDTH;
    parameter integer A_BITS_BA = (A_BITS + 7)/8 * 8;
    parameter integer B_BITS = PE_P1_0 * ACTIVATION_WIDTH;
    parameter integer B_BITS_BA = (B_BITS + 7)/8 * 8;
    parameter integer C_BITS = PE_P1_0 * ACCU_WIDTH;
    parameter integer C_BITS_BA = (C_BITS + 7)/8 * 8;

    parameter integer RF = 1;
    parameter integer MH_A = MH / RF;
    parameter integer MW_A = H_SIZE / RF;
    parameter integer MH_B = H_SIZE / RF;
    parameter integer MW_B = MH / RF;
    parameter integer MH_C = MH / RF;
    parameter integer MW_C = MH / RF;

    parameter integer N_TRS_A = (N_LOOPS * MH_A * MW_A) / SIMD_P1_0;
    parameter integer N_TRS_B = (N_LOOPS * MH_B * MW_B) / PE_P1_0;
    parameter integer N_TRS_C = (N_LOOPS * MH_C * MW_C) / PE_P1_0;

    parameter integer TH = PE_P1_0;
    
   
    
endpackage