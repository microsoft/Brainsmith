/****************************************************************************
 * MatMul HLS Header
 * 
 * Example header file for MatMul kernel HLS implementation.
 ****************************************************************************/
#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// Template-based matrix multiplication kernel
template<typename TI, typename TW, typename TO,
         unsigned M, unsigned K, unsigned N,
         unsigned PE, unsigned SIMD>
void matmul_kernel(
    hls::stream<hls::vector<TI, SIMD>> &in_stream,
    hls::stream<hls::vector<TW, SIMD>> &weight_stream,
    hls::stream<hls::vector<TO, PE>> &out_stream
) {
#pragma HLS INLINE off
    // Implementation would go here
    // This is just a stub for the example
}

#endif // MATMUL_HPP