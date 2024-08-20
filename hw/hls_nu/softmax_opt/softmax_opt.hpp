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
// Builds a tree to compute the max of a vector and also the 
// count of how many times it has appeared in the vector
template<typename T>
struct compcount_s {
	T  max;
	unsigned max_count; 
};

template<unsigned N, typename T>
struct CompCountReduction {

	static compcount_s<T> max(const hls::vector<T, N>& input) {
#pragma HLS INLINE
	constexpr unsigned M = N/2;
	hls::vector<compcount_s<T>, M> res;

	for(unsigned i=0; i < M; ++i) {
#pragma HLS unroll
		res[i].max = input[2*i] > input[2*i + 1] ? input[2*i] : input[2*i + 1];
		res[i].max_count = (input[2*i] == input[2*i +1]) ? 2 : 1;
	}

	return CompCountReduction<M,T>::max(res);
	}

	static compcount_s<T> max(const hls::vector<compcount_s<T>, N>& input) {
#pragma HLS INLINE
	constexpr unsigned M = N/2;
	hls::vector<compcount_s<T>, M> res;

	for(unsigned i=0; i < M; ++i) {
#pragma HLS unroll
		if (input[2*i].max > input[2*i+1].max) {
			res[i].max = input[2*i].max;
			res[i].max_count = input[2*i].max_count;
		} else {
			if (input[2*i].max < input[2*i+1].max){
				res[i].max = input[2*i+1].max;
				res[i].max_count = input[2*i+1].max_count;
			} else {
				// they are equal -- add the counts
				res[i].max = input[2*i+1].max;
				res[i].max_count = input[2*i+1].max_count + input[2*i].max_count;
			}
	       }
	}

	return CompCountReduction<M, T>::max(res);
	}
};

template<typename T>
struct CompCountReduction<2, T> {
	static compcount_s<T> max(const hls::vector<compcount_s<T>, 2>& input) {
#pragma HLS INLINE
		compcount_s<T> res;
		if (input[1].max > input[0].max) {
			res.max = input[1].max;
			res.max_count = input[1].max_count;
		} else {
			if (input[1].max < input[0].max){
				res.max = input[0].max;
				res.max_count = input[0].max_count;
			} else {
				// they are equal -- add the counts
				res.max = input[0].max;
				res.max_count = input[0].max_count + input[1].max_count;
			}
	       }
	      return res;
	}
};



template<unsigned SIMD, typename T>
void smax(
	hls::stream<hls::vector<T,SIMD>> &src,
	hls::stream<hls::vector<T,SIMD>> &dst
) {
	constexpr unsigned SIMD_FOLD = 128/SIMD;
	// Max
	T  max_val;
	ap_uint<clog2(SIMD_FOLD*SIMD+1)>  max_cnt = 0;

	// Input pass to determine max_cnt and max_val
	hls::stream<hls::vector<T, SIMD>>  preproc_buf;
#pragma HLS stream variable=preproc_buf depth=SIMD_FOLD
	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp
		hls::vector<T,SIMD> const x = src.read();
		compcount_s<T> res = CompCountReduction<SIMD, T>::max(x); 
		if ((max_cnt == 0) || (max_val < res.max)) {
			max_val = res.max;
			max_cnt = res.max_count;
		} else if(max_val == res.max) {
			max_cnt += res.max_count;
		}
		preproc_buf.write(x);
	}

	// Int buffer
	struct buf_s {
		T      xi;
		float  xx;
	};
	hls::stream<hls::vector<buf_s, SIMD>>  buf;
#pragma HLS stream variable=buf depth=SIMD_FOLD
	float  total = 0.0f;
	hls::vector<float, SIMD> xx;
	hls::vector<buf_s, SIMD> buf_tmp;

	hls::vector<float, SIMD_FOLD> outer_total_partials;
	// Loop input
	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp

		hls::vector<T,SIMD> const x = preproc_buf.read();
		hls::vector<float, SIMD> partial_sums;

		for(std::size_t j = 0; j < SIMD; j++) {
#pragma HLS UNROLL
			xx[j] = hls::exp(float(x[j]));
			buf_tmp[j] = {x[j], xx[j]};
			partial_sums[i] += xx[j];
		}

		outer_total_partials[i] = TreeReduction<SIMD>::reduce(partial_sums);
		buf.write(buf_tmp);
	}

	total = TreeReduction<SIMD_FOLD>::reduce(outer_total_partials);

	// calculate the total

	// Loop output
	bool const  ovf = hls::isinf(total);
	for(unsigned  i = 0; i < SIMD_FOLD; i++) {
#pragma HLS pipeline II=1 style=flp

		hls::vector<buf_s, SIMD> const  x = buf.read();

		float  a;
		float  d;
		hls::vector<float, SIMD> dst_tmp;
		hls::vector<T, SIMD> dst_tmp_int;

		for(std::size_t j = 0; j < SIMD; j++) {
#pragma HLS UNROLL
			if(ovf) {
				a = x[j].xi == max_val? 1.0f : 0.0f;
				d = max_cnt;
			}
			else {
				a = x[j].xx;
				d = total;
			}

			dst_tmp[j] = a/d;
			dst_tmp_int[j] = int8_t(dst_tmp[j]);
		}

		dst.write(dst_tmp_int);
	}

} // smax()
