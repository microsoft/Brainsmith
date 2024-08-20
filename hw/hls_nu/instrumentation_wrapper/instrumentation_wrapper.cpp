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
/******************************************************************************
 * @brief	Instrumentation wrapper module for FINN IP characterization.
 * @author	Thomas B. Preusser <thomas.preusser@amd.com>
 * @details
 *	Instrumentation wrapper intercepting the feature map input to and
 *	the feature map output from a FINN IP to measure processing latency and
 *	initiation interval in terms of clock cycles. The most recent readings
 *	are exposed via AXI-light.
 *	This wrapper can run the FINN IP detached from an external data source
 *	and sink by feeding LFSR-generated data and sinking the output without
 *	backpressure.
 *	This module is currently not integrated with the FINN compiler. It must
 *	be instantiated and integrated with the rest of the system in a manual
 *	process.
 *
 * @param PENDING	maximum number of feature maps in the FINN dataflow pipeline
 * @param ILEN		number of input transactions per IFM
 * @param OLEN		number of output transactions per OFM
 * @param KO           number of subwords within output payload vector
 * @param TI		type of input payload vector
 * @param TO		type of output payload vector
 *******************************************************************************/

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <algorithm>

// Example Module Configuration
constexpr unsigned  PENDING = 128;
constexpr unsigned  ILEN    = (128 * 384) / 4;
constexpr unsigned  OLEN    = (128 * 384) / 4;
constexpr unsigned  KO = 8;
using  TI = ap_uint<32>;
using  TO = ap_uint<32>;
//using  TO = hls::axis<ap_uint<32>, 0, 0, 0>;

//---------------------------------------------------------------------------
// Utility Functions
static constexpr unsigned clog2  (unsigned  x) { return  x<2? 0 : 1+clog2((x+1)/2); }
static constexpr unsigned clog2nz(unsigned  x) { return  clog2(x); }

template<typename  T>
static void move(
	hls::stream<T> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read());
}

template<typename  T>
static void move(
	hls::stream<hls::axis<T, 0, 0, 0>> &src,
	hls::stream<T> &dst
) {
#pragma HLS pipeline II=1 style=flp
	dst.write(src.read().data);
}

template<typename  T>
class Payload {
public:
	using  type = T;
};
template<typename  T>
class Payload<hls::axis<T, 0, 0, 0>> {
public:
	using  type = T;
};

/**
 * Computes a checksum over a forwarded stream assumed to carry frames of
 * N words further subdivided into K subwords.
 *      - Subword slicing can be customized typically by using a lambda.
 *        The provided DefaultSubwordSlicer assumes an `ap_(u)int`-like word
 *        type with a member `width` and a range-based slicing operator. It
 *        further assumes a little-endian arrangement of subwords within words
 *        for the canonical subword stream order.
 *      - Subwords wider than 23 bits are folded using bitwise XOR across
 *        slices of 23 bits starting from the LSB.
 *      - The folded subword values are weighted according to their position
 *        in the stream relative to the start of frame by a periodic weight
 *        sequence 1, 2, 3, ...
 *      - The weighted folded subword values are reduced to a checksum by an
 *        accumulation module 2^24.
 *      - A checksum is emitted for each completed frame. It is the concatenation
 *        of an 8-bit (modulo 256) frame counter and the 24-bit frame checksum.
 */
template<typename T, unsigned K>
class DefaultSubwordSlicer {
	static_assert(T::width%K == 0, "Word size must be subword multiple.");
	static constexpr unsigned  W = T::width/K;
public:
	ap_uint<W> operator()(T const &x, unsigned const  j) const {
#pragma HLS inline
		return  x((j+1)*W-1, j*W);
	}
};

//---------------------------------------------------------------------------
// Instrumentation Core
template<
	unsigned  PENDING,
	unsigned  ILEN,
	unsigned  OLEN,
	unsigned  KO,
	typename  TI,
	typename  TO
>
void instrument(
	hls::stream<TI> &finnix,
	hls::stream<TO> &finnox,
	ap_uint<32>  cfg,   	// [0] - 0:hold, 1:lfsr; [31:16] - LFSR seed
	ap_uint<2> &status,	// [0] - timestamp overflow; [1] - timestamp underflow
	ap_uint<32> &latency,
	ap_uint<32> &interval,
	ap_uint<32> &checksum,
	ap_uint<32> &ocnt_out
) {
#pragma HLS pipeline II=1 style=flp

	// Timestamp Management State
	using clock_t = ap_uint<32>;
	static clock_t  cnt_clk = 0;
#pragma HLS reset variable=cnt_clk
	hls::stream<clock_t>  timestamps;
#pragma HLS stream variable=timestamps depth=PENDING
	static bool  timestamp_ovf = false;
	static bool  timestamp_unf = false;
#pragma HLS reset variable=timestamp_ovf
#pragma HLS reset variable=timestamp_unf

	// Input Feed & Generation
	constexpr unsigned  LFSR_WIDTH = (TI::width+15)/16 * 16;
	static ap_uint<clog2nz(ILEN)>  icnt = 0;
	static ap_uint<LFSR_WIDTH>  lfsr;
#pragma HLS reset variable=icnt
#pragma HLS reset variable=lfsr off
	if(!finnix.full()) {

		bool const  first = icnt == 0;
		bool  wr;
		if(first) {
			// Start of new feature map
			wr = cfg[0];
			for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
#pragma HLS unroll
				lfsr(15+i, i) = cfg(31, 16) ^ (i>>4)*33331;
			}
		}
		else {
			// Advance LFSR
			wr = true;
			for(unsigned  i = 0; i < LFSR_WIDTH; i += 16) {
#pragma HLS unroll
				lfsr(15+i, i) = (lfsr(15+i, i) >> 1) ^ ap_uint<16>(lfsr[i]? 0 : 0x8805);
			}
		}

		if(wr) {
			finnix.write_nb(lfsr);
			if(first)  timestamp_ovf |= !timestamps.write_nb(cnt_clk);
			icnt = icnt == ILEN-1? decltype(icnt)(0) : decltype(icnt)(icnt + 1);
		}
	}

	// Output Tracking
	static ap_uint<clog2nz(OLEN)>  ocnt = 0;
#pragma HLS reset variable=ocnt
	static clock_t  ts1 = 0;	// last output timestamp
	static clock_t  last_latency = 0;
	static clock_t  last_interval = 0;
#pragma HLS reset variable=ts1
#pragma HLS reset variable=last_latency
#pragma HLS reset variable=last_interval

	static ap_uint<8>  pkts = 0;
#pragma HLS reset variable=pkts
	static ap_uint< 2>  coeff[3];
	static ap_uint<24>  psum;
	static ap_uint<32>  last_checksum = 0;
#pragma HLS reset variable=coeff off
#pragma HLS reset variable=psum off
#pragma HLS reset variable=last_checksum

	TO  oval;
	if(finnox.read_nb(oval)) {
		// Start of new output feature map
		if(ocnt == 0) {
			for(unsigned  i = 0; i < 3; i++)  coeff[i] = i+1;
			psum = 0;
		}

		// Update checksum
		for(unsigned  j = 0; j < KO; j++) {
#pragma HLS unroll
			auto const  v0 = DefaultSubwordSlicer<TO, KO>()(oval, j);
			constexpr unsigned  W = 1 + (decltype(v0)::width-1)/23;
			ap_uint<KO*23>  v = v0;
			ap_uint<   23>  w = 0;
			for(unsigned  k = 0; k < W; k++)  w ^= v(23*k+22, 23*k);
			psum += (coeff[j%3][1]? (w, ap_uint<1>(0)) : ap_uint<24>(0)) + (coeff[j%3][0]? w : ap_uint<23>(0));
		}

		// Re-align coefficients
		for(unsigned  j = 0; j < 3; j++) {
#pragma HLS unroll
				ap_uint<3> const  cc = coeff[j] + ap_uint<3>(KO%3);
				coeff[j] = cc(1, 0) + cc[2];
		}

		// Track frame position
		if(ocnt != OLEN-1)  ocnt++;
		else {
			clock_t  ts0;
			if(!timestamps.read_nb(ts0))  timestamp_unf = true;
			else {
				last_latency  = cnt_clk - ts0;	// completion - start
				last_interval = cnt_clk - ts1;	// completion - previous completion
				ts1 = cnt_clk;	// mark completion ^
			}
			ocnt = 0;

			last_checksum = (pkts++, psum);
		}
	}

	// Advance Timestamp Counter
	cnt_clk++;

	// Copy Status Outputs
	status[0] = timestamp_ovf;
	status[1] = timestamp_unf;
	latency  = last_latency;
	interval = last_interval;
	checksum = last_checksum;
	ocnt_out = ocnt;

} // instrument()

void instrumentation_wrapper(
	hls::stream<TI> &finnix,
	hls::stream<TO> &finnox,
	ap_uint<32>  cfg,
	ap_uint<2> &status,
	ap_uint<32> &latency,
	ap_uint<32> &interval,
	ap_uint<32> &checksum,
	ap_uint<32> &ocnt_out
) {
#pragma HLS interface axis port=finnix
#pragma HLS interface axis port=finnox

#pragma HLS interface ap_none register port=cfg
#pragma HLS INTERFACE ap_vld port=status
#pragma HLS INTERFACE ap_vld port=latency
#pragma HLS INTERFACE ap_vld port=interval
#pragma HLS INTERFACE ap_vld port=checksum
#pragma HLS INTERFACE ap_vld port=ocnt_out
#pragma HLS interface ap_ctrl_none port=return

// #pragma HLS interface s_axilite bundle=ctrl port=cfg
// #pragma HLS interface s_axilite bundle=ctrl port=status
// #pragma HLS interface s_axilite bundle=ctrl port=latency
// #pragma HLS interface s_axilite bundle=ctrl port=interval
// #pragma HLS interface s_axilite bundle=ctrl port=checksum
//#pragma HLS interface ap_ctrl port=return
//#pragma HLS interface ap_ctrl_hs port=return

#pragma HLS dataflow disable_start_propagation
	static hls::stream<TI>  finnix0;
	static hls::stream<Payload<TO>::type>  finnox0;
#pragma HLS stream variable=finnix0 depth=2
#pragma HLS stream variable=finnox0 depth=2

	// AXI-Stream -> FIFO
	move(finnox, finnox0);

	// Main
	instrument<PENDING, ILEN, OLEN, KO>(finnix0, finnox0, cfg, status, latency, interval, checksum, ocnt_out);

	// FIFO -> AXI-Stream
	move(finnix0, finnix);
	
} // instrumentation_wrapper