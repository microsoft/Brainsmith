/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT 
 *
 * @author	Shane T. Fleming <shane.fleming@amd.com>
 ****************************************************************************/

#ifndef SM_UTIL_HPP
#define SM_UTIL_HPP
#include "hls_vector.h"

//- Compile-Time Functions --------------------------------------------------

// ceil(log2(x))
//template<typename T>
//constexpr unsigned clog2(T  x) {
//  return  x<2? 0 : 1+clog2((x+1)/2);
//}

//- Streaming Flit with `last` Marking --------------------------------------
template<typename T>
struct flit_t {
	bool  last;
	T     data;

public:
	flit_t(bool  last_, T const &data_) : last(last_), data(data_) {}
	~flit_t() {}
};

//- Streaming Copy ----------------------------------------------------------
template<typename T>
void move(hls::stream<T> &src, hls::stream<T> &dst) {
#pragma HLS pipeline II=1 style=flp
	if(!src.empty())  dst.write(src.read());
}

//- Tree Reduce -------------------------------------------------------------
template< unsigned long  N, typename  TA, typename  TR = TA, typename  F >
TR tree_reduce(hls::stream<TA> &v, F f) {
#pragma HLS inline
#pragma HLS function_instantiate variable=f
        TR  tree[2*N-1];
#pragma HLS array_partition complete dim=1 variable=tree
        for(unsigned  i = N; i-- > 0;) {
#pragma HLS unroll
                tree[N-1 + i] = v.read();
        }
        for(unsigned  i = N-1; i-- > 0;) {
#pragma HLS unroll
                tree[i] = f(tree[2*i+1], tree[2*i+2]);
        }
        return  tree[0];
}

// Recursive comparison and count (of max)
// Builds a tree to compute the max of a vector
template<unsigned N, typename T>
struct MaxReduction {

    static T max(const hls::vector<T, N>& input) {
#pragma HLS INLINE
        constexpr unsigned M = (N + 1) / 2;
        hls::vector<T, M> res;

        for(unsigned i = 0; i < M; ++i) {
#pragma HLS unroll
            if (2*i + 1 < N)
                res[i] = input[2*i] > input[2*i + 1] ? input[2*i] : input[2*i + 1];
            else
                res[i] = input[2*i]; // Handle the case where the input size is odd
        }

        return MaxReduction<M, T>::max(res);
    }

};

template<typename T>
struct MaxReduction<2, T> {
    static T max(const hls::vector<T, 2>& input) {
#pragma HLS INLINE
        return (input[0] > input[1]) ? input[0] : input[1];
    }
};

template<typename T>
struct MaxReduction<1, T> {
    static T max(const hls::vector<T, 1>& input) {
#pragma HLS INLINE
        return input[0];
    }
};

// Recursive reduction tree for the total summation
// Code for the Nth stage
template<typename T, unsigned N>
struct TreeReduction {
    static T reduce(const hls::vector<T, N>& input) {
#pragma HLS INLINE
        constexpr unsigned M = (N + 1) / 2;
        hls::vector<T, M> sum;

        for(unsigned i = 0; i < M; ++i) {
#pragma HLS unroll
            if (2*i + 1 < N)
                sum[i] = input[2*i] + input[2*i + 1];
            else
                sum[i] = input[2*i]; // Handle the case where the input size is odd
        }

        return TreeReduction<T, M>::reduce(sum);
    }
};

template<typename T>
struct TreeReduction<T, 2> {
    static T reduce(const hls::vector<T, 2>& input) {
#pragma HLS INLINE
        return input[0] + input[1];
    }
};

template<typename T>
struct TreeReduction<T, 1> {
    static T reduce(const hls::vector<T, 1>& input) {
#pragma HLS INLINE
        return input[0];
    }
};

#endif
