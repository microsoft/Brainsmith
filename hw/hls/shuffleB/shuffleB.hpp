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
#ifndef INPUT_GEN_SHUFFLE_B_HPP
#define INPUT_GEN_SHUFFLE_B_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "utils.hpp"
#include "input_gen.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>

 /* Input generator for BERT shuffleB:
 * @param	T	(inferred) pixel type
 * @param	W	Width of the input	
 * @param	H	Height of the input	
 */
template<unsigned W, unsigned H, typename  T>
void input_gen_bertb(
	hls::stream<hls::vector<T, 4>> &src,
	hls::stream<hls::vector<T, 4>> &dst
) {
#pragma HLS pipeline II=1 style=flp

	constexpr unsigned  ADDR_BITS = clog2((W*H)>>2);
	constexpr unsigned  BUF_SIZE  = 1 << ADDR_BITS;

	// Output buffer
	static hls::vector<T,4> obuf;
	static bool ovld = false;
#pragma HLS reset variable=ovld 
#pragma HLS reset variable=obuf off

	// Buffer memory banks
	static ap_uint<1> pp_wp = 0; // ping-pong write pointer
	static ap_uint<1> pp_rp = 0; // ping-pong read pointer
#pragma HLS reset variable=pp_wp 
#pragma HLS reset variable=pp_rp 
	static T bank0[2][BUF_SIZE];
	static T bank1[2][BUF_SIZE];
	static T bank2[2][BUF_SIZE];
	static T bank3[2][BUF_SIZE];
#pragma HLS reset variable=bank0 off
#pragma HLS reset variable=bank1 off
#pragma HLS reset variable=bank2 off
#pragma HLS reset variable=bank3 off
#pragma HLS dependence variable=bank0 inter false
#pragma HLS dependence variable=bank0 intra false
#pragma HLS dependence variable=bank1 inter false
#pragma HLS dependence variable=bank1 intra false
#pragma HLS dependence variable=bank2 inter false
#pragma HLS dependence variable=bank2 intra false
#pragma HLS dependence variable=bank3 inter false
#pragma HLS dependence variable=bank3 intra false

	// compile time
	constexpr unsigned W_4 =  (unsigned)(W>>2);
	constexpr unsigned W2_4 = (unsigned)((W*2)>>2);
	constexpr unsigned W3_4 = (unsigned)((W*3)>>2);
	constexpr unsigned WH_4 = (unsigned)((W*H)>>2);
	constexpr unsigned H_4 = (unsigned)(H>>2);

	// Read logic variables
	static ap_uint<2> rot_rp = 0; // Read logic rotation
	static ap_uint<3> p_cnt = 0;
	static unsigned h_cnt = 0;
	static ap_uint<clog2(H>>2)> rp_cnt = 0;
	static unsigned rp_cnt_W = 0;
	static bool reset = false;
	static ap_uint<ADDR_BITS> count = 0;
#pragma HLS reset variable=rot_rp 
#pragma HLS reset variable=p_cnt 
#pragma HLS reset variable=h_cnt 
#pragma HLS reset variable=rp_cnt 
#pragma HLS reset variable=rp_cnt_W 
#pragma HLS reset variable=count 
#pragma HLS reset variable=reset 

	constexpr unsigned WP_DELAY=4;
	// -------- Buffer  Write logic ---------
	static ap_uint<2> rot_wp = 0; // controls the write logic rotation through banks 
	static ap_uint<ADDR_BITS> wp_idx_prev =0;
	static ap_uint<clog2(W/4)> rot_wp_cnt = W_4; // Used to control when rotation happens
	static ap_uint<ADDR_BITS> wp[WP_DELAY] = {0,};
#pragma HLS reset variable=rot_wp 
#pragma HLS reset variable=wp_idx_prev 
#pragma HLS reset variable=rot_wp_cnt 
#pragma HLS reset variable=wp 
#pragma HLS array_partition variable=wp complete

	// Reset logic
	if (reset) {
		rot_rp = 0;
		h_cnt = 0;
		count = 0;
		pp_rp = !pp_rp;
	}

	unsigned const idx0 = rp_cnt_W + h_cnt;
	unsigned const idx1 = W_4 + rp_cnt_W + h_cnt;
	unsigned const idx2 = W2_4 + rp_cnt_W + h_cnt;
	unsigned const idx3 = W3_4 + rp_cnt_W + h_cnt;
	unsigned const readOK = (wp[WP_DELAY-1] > idx3) || (pp_wp != pp_rp);
	
	// try and clear the output buffer
	if (ovld)  ovld = !dst.write_nb(obuf);

	// ----- Buffer Read Logic --------------
	reset = false;
	// Try and fill the output buffer
	if (!ovld) {
		if(readOK) {
			switch(rot_rp){
				case 0:
					obuf[0] = bank0[pp_rp][idx0];
					obuf[1] = bank3[pp_rp][idx1];
					obuf[2] = bank2[pp_rp][idx2];
					obuf[3] = bank1[pp_rp][idx3];
					break;
				case 1:
					obuf[0] = bank1[pp_rp][idx0];
					obuf[1] = bank0[pp_rp][idx1];
					obuf[2] = bank3[pp_rp][idx2];
					obuf[3] = bank2[pp_rp][idx3];
					break;
				case 2:
					obuf[0] = bank2[pp_rp][idx0];
					obuf[1] = bank1[pp_rp][idx1];
					obuf[2] = bank0[pp_rp][idx2];
					obuf[3] = bank3[pp_rp][idx3];
					break;
				case 3:
					obuf[0] = bank3[pp_rp][idx0];
					obuf[1] = bank2[pp_rp][idx1];
					obuf[2] = bank1[pp_rp][idx2];
					obuf[3] = bank0[pp_rp][idx3];
					break;

			}
			
			ovld = true;

			// read rotation logic
			if (rp_cnt >= (H_4-1)) {
				rp_cnt = 0; 
				rp_cnt_W = 0; 
				rot_rp++;

				if (p_cnt >= 3) {
					p_cnt = 0;
					if (!h_cnt)
						rot_rp += 3;
					h_cnt++;
				} else {
					p_cnt++;
				}
			} else {
				rp_cnt++;
				rp_cnt_W+=W;
			}

			if (count >= (WH_4-1)) {
				reset=true;
			} else {
				count++;
			}	

		}

	}

	hls::vector<T, 4> x;
	if (wp[0] < WH_4) {
		if (src.read_nb(x)) {

			// Have to be as explicit as possible to instantiate the right memory
			switch(rot_wp) {
				case 0:
					bank0[pp_wp][wp[0]] = x[0];
					bank1[pp_wp][wp[0]] = x[1];
					bank2[pp_wp][wp[0]] = x[2];
					bank3[pp_wp][wp[0]] = x[3];
					break;
				case 1:
					bank3[pp_wp][wp[0]] = x[0];
					bank0[pp_wp][wp[0]] = x[1];
					bank1[pp_wp][wp[0]] = x[2];
					bank2[pp_wp][wp[0]] = x[3];
					break;
				case 2:
					bank2[pp_wp][wp[0]] = x[0];
					bank3[pp_wp][wp[0]] = x[1];
					bank0[pp_wp][wp[0]] = x[2];
					bank1[pp_wp][wp[0]] = x[3];
					break;
				case 3:
					bank1[pp_wp][wp[0]] = x[0];
					bank2[pp_wp][wp[0]] = x[1];
					bank3[pp_wp][wp[0]] = x[2];
					bank0[pp_wp][wp[0]] = x[3];
					break;
			}

			wp[0]++;

			if(rot_wp_cnt >= W_4) {
				rot_wp++;
				rot_wp_cnt = 0;
			}
			rot_wp_cnt++;
		}
	} else {
		if(pp_rp == pp_wp) {
			wp[0] = 0;	
			wp[1] = 0;	
			wp[2] = 0;	
			wp[3] = 0;	
	 		pp_wp = !pp_wp;	
		}
	}

	// Update delay pipeline for wp
	for(unsigned  i = WP_DELAY-1; i > 0; i--)  wp[i] = wp[i-1];

	// --------------------------------------


} // input_gen_bertb()

#endif
