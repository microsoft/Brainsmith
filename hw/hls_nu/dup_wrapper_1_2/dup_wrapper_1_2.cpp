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
#ifndef DUP_HPP
#define DUP_HPP

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <algorithm>

constexpr bool  GUARD = true;
constexpr unsigned  N    = 2;
using T = ap_uint<32>;

template<
	bool  GUARD = true,
	unsigned  N,
	typename  T
>
void dup(
	hls::stream<T>  &src,
	hls::stream<T> (&dst)[N]
) {
#pragma HLS pipeline II=1 style=flp
	if(!GUARD || !src.empty()) {
		T const  x = src.read();
		for(unsigned  i = 0; i < N; i++) {
#pragma HLS unroll
			dst[i].write(x);
		}
	}

} // dup()

template<typename  T>
static void move(
	hls::stream<T> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read());
}


// --------------------------------------------------
void dup_wrapper_1_2(
	hls::stream<T> &src,
	hls::stream<T> (&dst)[N]
) {
#pragma HLS interface axis port=src
#pragma HLS interface axis port=dst

#pragma HLS dataflow disable_start_propagation
	static hls::stream<T>  src0;
	static hls::stream<T>  dst0 [N];
#pragma HLS stream variable=src0 depth=2
#pragma HLS stream variable=dst0 depth=2

#pragma HLS interface ap_ctrl_none port=return

	// AXI-Stream -> FIFO
	move(src, src0);

	// Main
	dup<GUARD, N, T>(src0, dst0);

	// FIFO -> AXI-Stream
	for(unsigned  i = 0; i < N; i ++) {
#pragma HLS unroll
		move(dst0[i], dst[i]);
	}

} // dup_wrapper_1_2

#endif