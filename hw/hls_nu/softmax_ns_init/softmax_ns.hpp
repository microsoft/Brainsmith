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
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>

#include "utils.hpp"

// Recursive reduction tree for the total summation
// Code for the Nth stage
template<unsigned N>
struct TreeReduction {
	static float reduce(const hls::vector<float, N>& input) {
#pragma HLS INLINE
	constexpr unsigned M = N/2;
	hls::vector<float, M> sum;

	for(unsigned i=0; i < M; ++i) {
#pragma HLS unroll
		sum[i] = input[2*i] + input[2*i + 1];
	}

	return TreeReduction<M>::reduce(sum);
	}
};

template<>
struct TreeReduction<2> {
	static float reduce(const hls::vector<float, 2>& input) {
#pragma HLS INLINE
		return input[0] + input[1];
	}
};


// Recursive comparison and count (of max)
// Builds a tree to compute the max of a vector 
template<unsigned N, typename T>
struct MaxReduction {

	static T max(const hls::vector<T, N>& input) {
#pragma HLS INLINE
		constexpr unsigned M = N/2;
		hls::vector<T, M> res;

		for(unsigned i=0; i < M; ++i) {
#pragma HLS unroll
			res[i] = input[2*i] > input[2*i + 1] ? input[2*i] : input[2*i + 1];
		}

		return MaxReduction<M,T>::max(res);
	}

};

template<typename T>
struct MaxReduction<2, T> {
	static T max(const hls::vector<T, 2>& input) {
#pragma HLS INLINE
		return (input[0] > input[1]) ? input[0] : input[1];
	}
};



template<unsigned SIMD, typename T>
void smax(
	hls::stream<hls::vector<T,SIMD>> &src,
	hls::stream<hls::vector<float,SIMD>> &dst
) {
#pragma HLS DATAFLOW
	constexpr unsigned SIMD_FOLD = 128/SIMD;
	// Max
	bool first=true;
	T  max;

	// Input pass to determine max
	hls::stream<hls::vector<T, SIMD>>  preproc_buf;
#pragma HLS stream variable=preproc_buf depth=SIMD_FOLD
	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp
		hls::vector<T,SIMD> const x = src.read();
		T res = MaxReduction<SIMD, T>::max(x); 
		if (first) {
			max = res;
			first = false;
		} else {
			max = (max > res) ? max : res;
		}

		preproc_buf.write(x);
	}

	float max_fp = float(max);

	// Pass over the data to calculate the exp and total
	float sum=0.0f;
	hls::stream<hls::vector<float, SIMD>>  exp_buf;
	hls::vector<float, SIMD_FOLD> partial_sum;
#pragma HLS stream variable=exp_buf depth=SIMD_FOLD
	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp
		hls::vector<float, SIMD> x;
		hls::vector<T, SIMD> y = preproc_buf.read();
		for(unsigned j=0; j<SIMD; j++){
			x[j] = hls::exp(float(y[j] - max)); 
		}	
		partial_sum[i] = TreeReduction<SIMD>::reduce(x); 
		exp_buf.write(x);
	}
	sum = TreeReduction<SIMD_FOLD>::reduce(partial_sum);

	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp
		hls::vector<float, SIMD> x = exp_buf.read();
		hls::vector<float, SIMD> y;
		for(unsigned j=0; j<SIMD; j++) {
			y[j] = x[j] / sum; 
		}	
		dst.write(y);
	}

} // smax()