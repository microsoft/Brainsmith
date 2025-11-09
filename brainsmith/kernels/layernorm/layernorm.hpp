/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT 
 *
 * @author  Shane T. Fleming <shane.fleming@amd.com>
 ****************************************************************************/
#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include "bs_utils.hpp"


// First pipeline stage
//
// Trigger: Data available on src input stream
//
// Desc: Performs a mean calculation across N elements. 
template<typename TI, typename TO, unsigned N, unsigned SIMD>
void mean_stage(
	hls::stream<hls::vector<TI, SIMD>> &in_s,
	hls::stream<hls::vector<TO, SIMD>> &out_s,
	hls::stream<TO> &mean_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N)+1> count = 0;
	static TO sum = TO(0.0f);
#pragma HLS reset variable=count
#pragma HLS reset variable=sum

	if (!in_s.empty()) {
		hls::vector<TO,SIMD> out;
		hls::vector<TI,SIMD> const in = in_s.read();

		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = TO(in[i]);
		}
		out_s.write(out);

		// Accumulate sum
		sum += TreeReduction<TO,SIMD>::reduce(out);
		count += SIMD;

		// Compute mean only at the end (division by constant N, optimized to multiply)
		if (count == N) {
			TO mean = sum / TO(N);  // Constant division - compiler optimizes to multiply
			count = 0;
			mean_s.write(mean);
			sum = TO(0.0f);
		}

	}

}

// For the output of the second stage we are
// calculating the variance but also want to
// pass along the mean value from stage1.
template<typename T>
struct varmean_t {
	T mean;
	T var;
};

// Second pipeline stage
//
// Trigger: On data being available on the mean value stream 
//
// Desc: Performs a variance calculation across N elements. 
template<typename TO, unsigned N, unsigned SIMD>
void var_stage(
	hls::stream<hls::vector<TO, SIMD>> &in_s,
	hls::stream<TO> &mean_s,

	hls::stream<hls::vector<TO,SIMD>> &out_s,
	hls::stream<varmean_t<TO>> &varmean_s
) {
#pragma HLS pipeline II=1 style=flp 
	static ap_uint<clog2(N) + 1> count = 0;
	static TO pow_sum = TO(0.0f);
	static TO mean = TO(0.0f);
	static bool valid = false;
#pragma HLS reset variable=count
#pragma HLS reset variable=pow_sum
#pragma HLS reset variable=mean
#pragma HLS reset variable=valid

	if (count == N) {
		// Compute variance only at the end (division by constant N, optimized to multiply)
		TO var = pow_sum / TO(N);  // Constant division - compiler optimizes to multiply
		count = 0;
		valid = false;
		varmean_t<TO> x = { mean, var };
		varmean_s.write(x);
		pow_sum = TO(0.0f);
		return;
	}

	if (valid && !in_s.empty()) {
		hls::vector<TO, SIMD> const in = in_s.read();
		out_s.write(in); // Pass the bulk of the data along

		hls::vector<TO, SIMD> pow_res;
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			pow_res[i] = hls::pow((in[i] - mean), 2.0f);
		}
		pow_sum += TreeReduction<TO,SIMD>::reduce(pow_res);

		count += SIMD;
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
template<typename TO, unsigned N, unsigned SIMD>
void inv_sqrt_stage(
	const TO epsilon,
	hls::stream<hls::vector<TO, SIMD>> &in_s,
	hls::stream<hls::vector<TO, SIMD>> &out_s,
	hls::stream<varmean_t<TO>> &varmean_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N/SIMD)+1> count = 0; 
	static bool valid = false;
	static varmean_t<TO> vm;
#pragma HLS reset variable=count
#pragma HLS reset variable=valid
#pragma HLS reset variable=vm

	if(count == (N/SIMD)) {
		count = 0; 
		valid = false;
		return;
	}

	if (valid && !in_s.empty()) {
		hls::vector<TO, SIMD> const in = in_s.read();
		hls::vector<TO, SIMD> out;

		// Compute reciprocal of standard deviation once (instead of dividing SIMD times)
		// inv_std = 1 / sqrt(var + epsilon)
		// Then: normalized = (x - mean) * inv_std  (multiply instead of divide)
		TO inv_std = TO(1.0f) / hls::sqrt(vm.var + epsilon);

		for (unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = (in[i] - vm.mean) * inv_std;  // Multiply by reciprocal
		}
		out_s.write(out);
		count++;
	}

	if (!varmean_s.empty() && !valid) {
		vm = varmean_s.read();
		valid = true;
	}
}

template<typename TI, // Input type
       	 typename TO, // Output type 
	     unsigned N, 
	     unsigned SIMD>
void layernorm_pipeline(
	const TO epsilon,
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, SIMD>> &dst
) {
#pragma HLS DATAFLOW disable_start_propagation

	static hls::stream<hls::vector<TO, SIMD>> stage1_s;
#pragma HLS stream variable=stage1_s depth=N
	static hls::stream<TO> mean_s;
#pragma HLS stream variable=mean_s depth=2
	static hls::stream<hls::vector<TO, SIMD>> stage2_s;
#pragma HLS stream variable=stage2_s depth=N
	static hls::stream<varmean_t<TO>> varmean_s; // Stream of the variance and mean combined
#pragma HLS stream variable=varmean_s depth=2

	mean_stage<TI, TO, N, SIMD>(src, stage1_s, mean_s);
	var_stage<TO, N, SIMD>(stage1_s, mean_s, stage2_s, varmean_s);
	inv_sqrt_stage<TO, N, SIMD>(epsilon, stage2_s, dst, varmean_s);
}

#endif
