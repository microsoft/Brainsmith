/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT 
 *
 * @author      Shane T. Fleming <shane.fleming@amd.com>
 ****************************************************************************/

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include <cmath>
#include <climits>
#include <type_traits>
#include "bs_utils.hpp"

// First stage of the pipeline:
//
// Trigger: When a vector of SIMD elements is present in the stream
//
// Desc: Pass over the input N items and calc the max value
template<unsigned N, unsigned SIMD, typename T>
void max_calc_stage(
	hls::stream<hls::vector<T, SIMD>> &ins, 
	hls::stream<hls::vector<T,SIMD>> &outs,
	hls::stream<T> &maxs
) {
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)+1> count = 0;
	static T max = 0;
#pragma HLS reset variable=count
#pragma HLS reset variable=max

	if (count == (N/SIMD)) {
		count = 0;
		maxs.write(max);
		max = 0;
		return;
	}

	if(!ins.empty()){
		hls::vector<T,SIMD> out;
		hls::vector<T,SIMD+1> max_v;
		hls::vector<T,SIMD> const in = ins.read();

		for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL 
			out[i] = in[i]; 
			max_v[i] = in[i];
		}
		outs.write(out);

		max_v[SIMD] = max;
		max = MaxReduction<SIMD+1, T>::max(max_v);

		count++;
	}
}


// Second stage of the pipeline
//
// Trigger: When a max value is sent from the preceeding stage 
//
// Desc: For each item in a N item sequence calc the (exp - max) in float
// track the sum while processing the N items.
template<unsigned N, unsigned SIMD, typename T>
void exp_sum_calc(
	hls::stream<hls::vector<T, SIMD>> &ins, 
	hls::stream<T> &maxs, 
	hls::stream<hls::vector<float, SIMD>> &outs,
	hls::stream<float> &sums
){
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)+1> count = 0;
	static float sum = 0.0f;
	static bool valid = false;
	static float max = 0.0f;
#pragma HLS reset variable=count
#pragma HLS reset variable=sum
#pragma HLS reset variable=valid
#pragma HLS reset variable=max

	if (count == (N/SIMD)) {
		count = 0;
		valid = false;
		sums.write(sum);
		sum = 0.0f;
		return;
	}

	if(valid && !ins.empty()) {
		hls::vector<T, SIMD> const in = ins.read();
		hls::vector<float, SIMD> out;
		for (unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = hls::exp(float(in[i]) - max); 	
		}
		sum += TreeReduction<float,SIMD>::reduce(out); 
		outs.write(out);
		
		count++;
	}

	if (!maxs.empty() && !valid) {
		max = maxs.read();
		valid = true;
	}

}

// Third stage of the pipeline
//
// Trigger: When a sum value is sent from the preceeding stage 
// 
// Desc: For the N items take the input and divide it by the sum 
template<unsigned N, unsigned SIMD>
void div_calc(
	hls::stream<hls::vector<float, SIMD>> &ins, 
	hls::stream<float> &sums,
	hls::stream<hls::vector<float, SIMD>> &outs
){
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)+1> count = 0;
	static bool valid = false;
	static float sum = 0.0f;
#pragma HLS reset variable=count
#pragma HLS reset variable=valid
#pragma HLS reset variable=sum

	if (count == (N/SIMD)) {
		count = 0;
		valid = false;
		return;
	}

	if (valid && !ins.empty()) {
		hls::vector<float, SIMD> const in = ins.read();
		hls::vector<float, SIMD> out;
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS unroll
			out[i] = in[i] / sum;
		}

		outs.write(out);

		count++;
	}

	if(!sums.empty() && !valid ){
		valid = true;
		sum = sums.read();
	}
}


template<unsigned N, unsigned SIMD, typename T>
void smax(
    hls::stream<hls::vector<T, SIMD>> &src,
    hls::stream<hls::vector<float, SIMD>> &dst
) {
#pragma HLS dataflow disable_start_propagation 
    static_assert(N%SIMD == 0, "N must be a multiple of SIMD");

    static hls::stream<hls::vector<T,SIMD>> max_data_s;
#pragma HLS stream variable=max_data_s depth=2*N
    static hls::stream<T> max_s;
#pragma HLS stream variable=max_s depth=4

    static hls::stream<hls::vector<float,SIMD>> exp_data_s;
#pragma HLS stream variable=exp_data_s depth=2*N
    static hls::stream<float> sum_s;
#pragma HLS stream variable=sum_s depth=4

    max_calc_stage<N, SIMD, T>(src, max_data_s, max_s);
    exp_sum_calc<N, SIMD, T>(max_data_s, max_s, exp_data_s, sum_s);
    div_calc<N,SIMD>(exp_data_s, sum_s, dst);

} // smax()


