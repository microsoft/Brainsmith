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
#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include "utils.hpp"

constexpr float epsilon = 1e-5;

// First pipeline stage
//
// Trigger: Data available on src input stream
//
// Desc: Performs a mean calculation across N elements. 
template<unsigned N, unsigned SIMD, typename T>
void mean_stage(
	hls::stream<hls::vector<T, SIMD>> &in_s,
	hls::stream<hls::vector<float, SIMD>> &out_s,
	hls::stream<float> &mean_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N)> count = 0; 
	static float sum = 0.0f;
	static float mean = 0.0f;
#pragma HLS reset variable=count
#pragma HLS reset variable=sum
#pragma HLS reset variable=mean
	
	if (!in_s.empty()) {
		hls::vector<float,SIMD> out;
		hls::vector<T,SIMD> const in = in_s.read();

		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = float(in[i]);	
		}
		out_s.write(out);

		// Mean calc
		sum += TreeReduction<SIMD>::reduce(out);
		count+=SIMD;
		mean = sum / count;

		if (count == N) {
			count = 0;
			mean_s.write(mean); 
			mean = 0.0f;
			sum = 0.0f;
		}
		
	}

}

// For the output of the second stage we are
// calculating the variance but also want to
// pass along the mean value from stage1.
struct varmean_t {
	float mean;
	float var;
};

// Second pipeline stage
//
// Trigger: On data being available on the mean value stream 
//
// Desc: Performs a variance calculation across N elements. 
template<unsigned N, unsigned SIMD>
void var_stage(
	hls::stream<hls::vector<float, SIMD>> &in_s,
	hls::stream<float> &mean_s,

	hls::stream<hls::vector<float,SIMD>> &out_s,
	hls::stream<varmean_t> &varmean_s
) {
#pragma HLS pipeline II=1 style=flp 
	static ap_uint<clog2(N)> count = 0; 
	static float pow_sum = 0.0f;
	static float var = 0.0f;
	static float mean = 0.0f;
	static bool valid = false;
#pragma HLS reset variable=count
#pragma HLS reset variable=pow_sum
#pragma HLS reset variable=var
#pragma HLS reset variable=mean
#pragma HLS reset variable=valid

	if (count == N) {
		count = 0; 
		valid = false;
		varmean_t x = { mean, var };
		varmean_s.write(x);
		pow_sum = 0.0f;
		var = 0.0f;	
		return;
	}

	if (valid && !in_s.empty()) {
		hls::vector<float, SIMD> const in = in_s.read();
		out_s.write(in); // Pass the bulk of the data along

		hls::vector<float, SIMD> pow_res;
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			pow_res[i] = hls::pow((in[i] - mean), 2.0f); 
		}
		pow_sum += TreeReduction<SIMD>::reduce(pow_res);  

		count += SIMD;
		var = pow_sum / count;
	}

	if (!mean_s.empty() && !valid) {
		mean = mean_s.read();
		valid = true;
	}
}


// Third pipeline stage
//
// Trigger: On data being available on the varmean value stream 
//
// Desc: Performs a variance calculation across N elements. 
template<unsigned N, unsigned SIMD>
void inv_sqrt_stage(
	hls::stream<hls::vector<float, SIMD>> &in_s,
	hls::stream<varmean_t> &varmean_s,

	hls::stream<hls::vector<float, SIMD>> &out_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N/SIMD)+1> count = 0; 
	static bool valid = false;
	static varmean_t vm;
#pragma HLS reset variable=count
#pragma HLS reset variable=valid
#pragma HLS reset variable=vm

	if(count == (N/SIMD)) {
		count = 0; 
		valid = false;
		return;
	}

	if (valid && !in_s.empty()) {
		hls::vector<float, SIMD> const in = in_s.read();
		hls::vector<float, SIMD> out;
		for (unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = (in[i] - vm.mean) / hls::sqrt(vm.var + epsilon);  
		}
		out_s.write(out);
		count++;
	}


	if (!varmean_s.empty() && !valid) {
		vm = varmean_s.read();
		valid = true;
	}
}

template<unsigned N, unsigned SIMD, typename T>
void layernorm_pipeline(
	hls::stream<hls::vector<T, SIMD>> &src,
	hls::stream<hls::vector<float, SIMD>> &dst
) {
#pragma HLS DATAFLOW disable_start_propagation

	static hls::stream<hls::vector<float, SIMD>> stage1_s;
#pragma HLS stream variable=stage1_s depth=N
	static hls::stream<float> mean_s;
#pragma HLS stream variable=mean_s depth=2
	static hls::stream<hls::vector<float, SIMD>> stage2_s;
#pragma HLS stream variable=stage2_s depth=N
	static hls::stream<varmean_t> varmean_s; // Stream of the variance and mean combined
#pragma HLS stream variable=varmean_s depth=2

	mean_stage<N, SIMD,T>(src, stage1_s, mean_s);
	var_stage<N, SIMD>(stage1_s, mean_s, stage2_s, varmean_s);
	inv_sqrt_stage<N, SIMD>(stage2_s, varmean_s, dst);
}

#endif
