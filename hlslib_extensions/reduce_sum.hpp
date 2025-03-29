/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	A reduce_sum operator 	
 * @author	Shane Fleming <shane.fleming@amd.com>
 *
 * @description
 * 		Uses a reduction tree to reduce the elements of
 * 		the last vector. 
 *
 ***************************************************************************/
#ifndef REDUCE_SUM_HPP
#define REDUCE_SUM_HPP

#include "bs_utils.hpp"
#include <ap_fixed.h>
#include <ap_float.h>

// TODO: Accumulation datatype awareness to avoid overflow.

template<
	typename T, 
	unsigned SIMD, // Amount of SIMD parallelism on the channel
	unsigned N // How much are we summing from the channel
>
void reduce_sum (
		hls::stream<hls::vector<T,SIMD> &src,
		hls::stream<T> &dst
) {

	static T accumulator = 0;
	static ap_uint<clog2(N/SIMD)+1> count = 0;

	if (!src.empty()) {
		hls::vector<T,SIMD> y = src.read();
		for(unsigned i=0; i<SIMD; i++) {
			accumulator += tree_reduce<SIMD, T, T>(y
					[](T a, T b){ return a+b; });
		}

		count++;
		if (count >= N/SIMD){
			dst.write(accumulator);
			count = 0;
			accumulator = 0;
		}
	}

} // reduce_sum


#endif
