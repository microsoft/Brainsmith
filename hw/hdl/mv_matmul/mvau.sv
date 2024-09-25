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
module mvau #(
	bit IS_MVU,
	parameter COMPUTE_CORE,
	int unsigned MH,
	int unsigned MW,
	int unsigned PE,
	int unsigned SIMD,

	int unsigned ACTIVATION_WIDTH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH = 2*ACTIVATION_WIDTH+$clog2(MH),

	int unsigned SEGMENTLEN = 1,
	bit SIGNED_ACTIVATIONS = 1,
	bit PUMPED_COMPUTE = 0,
	bit FORCE_BEHAVIORAL = 0,
	bit M_REG_LUT = 1,

	// Safely deducible parameters
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH     = SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE * ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN = SIMD % 2
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	// synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_b_tdata,
	input	logic  s_axis_b_tvalid,
	output	logic  s_axis_b_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_a_tdata,
	input	logic  s_axis_a_tvalid,
	output	logic  s_axis_a_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_c_tdata,
	output	logic  m_axis_c_tvalid,
	input	logic  m_axis_c_tready
);

//-------------------- Parameter sanity checks --------------------\\
	initial begin
		if (MH % SIMD != 0) begin
			$error("Matrix width (%0d) is not a multiple of SIMD (%0d).", MH, SIMD);
			$finish;
		end
		if (MW % PE != 0) begin
			$error("Matrix height (%0d) is not a multiple of PE (%0d).", MW, PE);
			$finish;
		end
		if (WEIGHT_WIDTH > 8) begin
			$error("Weight width of %0d-bits exceeds maximum of 8-bits", WEIGHT_WIDTH);
			$finish;
		end
		if (ACTIVATION_WIDTH > 8) begin
			if (!(SIGNED_ACTIVATIONS == 1 && ACTIVATION_WIDTH == 9 && COMPUTE_CORE == "mvu_vvu_8sx9_dsp58")) begin
				$error("Activation width of %0d-bits exceeds maximum of 9-bits for signed numbers on DSP48", ACTIVATION_WIDTH);
				$finish;
			end
		end
		if (COMPUTE_CORE == "mvu_vvu_8sx9_dsp58") begin
			if (SEGMENTLEN == 0) begin
				$error("Segment length of %0d defaults to chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
			end
			if (SEGMENTLEN > (SIMD+2)/3) begin
				$error("Segment length of %0d exceeds chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
				$finish;
			end
		end
		if (!IS_MVU) begin
			if (COMPUTE_CORE != "mvu_vvu_8sx9_dsp58" && COMPUTE_CORE != "mvu_vvu_lut") begin
				$error("VVU only supported on DSP58 or LUT-based implementation");
				$finish;
			end
		end
	end

	uwire  clk = ap_clk;
	uwire  clk2x = ap_clk2x;
	uwire  rst = !ap_rst_n;

	//- Replay to Accommodate Neuron Fold -----------------------------------
	//-----------------------------------------------------------------------
	typedef logic [(IS_MVU? 1:PE)*SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_flatin_t;
	uwire mvu_flatin_t amvau;
	uwire alast;
	uwire afin;
	uwire avld;
	uwire ardy;

	localparam int unsigned  SF = MH/SIMD;
	localparam int unsigned  NF = MW/PE;
	replay_buff_vector #(.LEN(SF), .REP(IS_MVU ? NF : 1), .W($bits(mvu_flatin_t))) activation_replay (
		.clk, .rst,
		.ivld(s_axis_a_tvalid), .irdy(s_axis_a_tready), .idat(mvu_flatin_t'(s_axis_a_tdata)),
		.ovld(avld), .ordy(ardy), .odat(amvau), .olast(alast), .ofin(afin)
	);

	//- Unflatten inputs into structured matrices ---------------------------
	//-----------------------------------------------------------------------
	localparam int unsigned  ACT_PE = IS_MVU? 1 : PE;
	typedef logic [PE    -1:0][WEIGHT_WIDTH    -1:0]  			mvu_b_t;
	typedef logic [PE    -1:0][SIMD-1:0][WEIGHT_WIDTH    -1:0]  mvu_w_t;
	typedef logic [ACT_PE-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0]  mvu_a_t;

	uwire mvu_w_t mvu_w;
	uwire wgt_val;
	uwire wgt_rdy;
	assign wgt_val = s_axis_b_tvalid;
	assign s_axis_b_tready = wgt_rdy;
	assign mvu_w = mvu_w_t'(s_axis_b_tdata);

	//- Flow Control Bracket around Compute Core ----------------------------
	uwire en;
	uwire istb = avld && wgt_val;
	assign ardy = en && wgt_val;
	assign wgt_rdy = en && avld;

	//- Conditional Activations Layout Adjustment for VVU -------------------
	//-----------------------------------------------------------------------
	uwire mvu_a_t  amvau_i;
	if (IS_MVU || (PE == 1)) begin : genMVUInput
		assign  amvau_i = amvau;
	end : genMVUInput
	else begin : genVVUInput
		// The input stream will have the channels interleaved for VVU when PE>1
		// Hence, we need to 'untangle' the input stream, i.e. [..][SIMD*PE][..] --> [..][PE][SIMD][..]
		// Note that for each 'SIMD' (S) and 'PE' (P) element, we have something like:
		// (S_0, P_0), ..., (S_0, P_i), (S_1, P_0), ..., (S_1, P_i), ..., (S_i, P_i) which we need to 'untangle' to
		// (S_0, P_0), ..., (S_i, P_0), (S_0, P_1), ..., (S_i, P_1), ..., (S_i, P_i)
		for(genvar  pe = 0; pe < ACT_PE; pe++) begin
			for(genvar  simd = 0; simd < SIMD; simd++) begin
				assign	amvau_i[pe][simd] = amvau[simd*ACT_PE+pe];
			end
		end
	end : genVVUInput

	//- Conditionally Pumped DSP Compute ------------------------------------
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
	uwire  ovld;
	uwire dsp_p_t  odat;
	if(1) begin : blkDsp
		localparam int unsigned  EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE ? SIMD+1 : SIMD;
		localparam int unsigned  DSP_SIMD = EFFECTIVE_SIMD/(PUMPED_COMPUTE+1);
		typedef logic [PE    -1:0][DSP_SIMD-1:0][WEIGHT_WIDTH    -1:0]  dsp_w_t;
		typedef logic [ACT_PE-1:0][DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  dsp_a_t;

		uwire  dsp_clk;
		uwire  dsp_en;

		uwire  dsp_last;
		uwire  dsp_zero;
		uwire dsp_w_t  dsp_w;
		uwire dsp_a_t  dsp_a;

		uwire  dsp_vld;
		uwire dsp_p_t  dsp_p;

		if(!PUMPED_COMPUTE) begin : genUnpumpedCompute
			assign	dsp_clk = clk;
			assign	dsp_en  = en;

			assign	dsp_last = alast && avld;
			assign	dsp_zero = !istb;
			assign	dsp_w = mvu_w;
			assign	dsp_a = amvau_i;

			assign	ovld = dsp_vld;
			assign	odat = dsp_p;
		end : genUnpumpedCompute
		else begin : genPumpedCompute
			assign	dsp_clk = clk2x;

			// Identify second fast cycle just before active slow clock edge
			logic  Active = 0;
			always_ff @(posedge clk2x) begin
			 if(~ap_rst_n) begin
			     Active <= 0;
			 end
			 else begin
			     Active <= !Active;
			 end
			end
			/*if() begin : blkActive
				uwire  clk_lut[2];	// Put some LUT delay on the input from the fast clock net
				(* DONT_TOUCH = "TRUE", HLUTNM = "CLK_LUT" *) LUT1 #(.INIT(2'b10)) lut0(.O(clk_lut[0]), .I0(clk));
				(* DONT_TOUCH = "TRUE", HLUTNM = "CLK_LUT" *) LUT1 #(.INIT(2'b10)) lut1(.O(clk_lut[1]), .I0(clk_lut[0]));
				always_ff @(posedge clk2x)  Active <= clk_lut[1];
			end : blkActive*/

			// The input for a slow cycle is split across two fast cycles along the SIMD dimension.
			//	- Both fast cycles are controlled by the same enable state.
			//	- A zero cycle is duplicated across both fast cycles.
			//	- The last flag must be restricted to the second fast cycle.

			dsp_w_t  W = 'x;
			for(genvar  pe = 0; pe < PE; pe++) begin : genPERegW

				uwire [2*DSP_SIMD-1:0][WEIGHT_WIDTH-1:0]  w;
				for(genvar  i =    0; i <       SIMD; i++)  assign  w[i] = mvu_w[pe][i];
				for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  w[i] = 0;

				always_ff @(posedge clk2x) begin
					if(rst)      W[pe] <= 'x;
					else if(en)  W[pe] <= w[(Active? DSP_SIMD : 0) +: DSP_SIMD];
				end

			end : genPERegW

			dsp_a_t  A = 'x;
			for(genvar  pe = 0; pe < ACT_PE; pe++) begin : genPERegA

				uwire [2*DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  a;
				for(genvar  i =    0; i <       SIMD; i++)  assign  a[i] = amvau_i[pe][i];
				for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  a[i] = 0;

				always_ff @(posedge clk2x) begin
					if(rst)      A[pe] <= 'x;
					else if(en)  A[pe] <= a[(Active? DSP_SIMD : 0) +: DSP_SIMD];
				end

			end : genPERegA

			logic  Zero = 1;
			logic  Last = 0;
			always_ff @(posedge clk2x) begin
				if(rst) begin
					Zero <= 1;
					Last <= 0;
				end
				else if(en) begin
					Zero <= !istb;
					Last <= alast && avld && Active;
				end
			end

			assign	dsp_en = en;
			assign	dsp_last = Last;
			assign	dsp_zero = Zero;
			assign	dsp_w = W;
			assign	dsp_a = A;

			// Since no two consecutive last cycles will ever be asserted on the input,
			// valid outputs will also always be spaced by, at least, one other cycle.
			// We can always hold a captured output for two cycles to allow the slow
			// clock to pick it up.
			logic    Vld = 0;
			dsp_p_t  P = 'x;
			always_ff @(posedge clk2x) begin
				if(rst) begin
					Vld <= 0;
					P   <= 'x;
				end
				else if(en) begin
					if(dsp_vld)  P <= dsp_p;
					Vld <= dsp_vld || (Vld && !Active);
				end
			end
			assign	ovld = Vld;
			assign	odat = P;

		end : genPumpedCompute

		case(COMPUTE_CORE)
		"mvu_vvu_8sx9_dsp58":
			mvu_vvu_8sx9_dsp58 #(.IS_MVU(IS_MVU), .PE(PE), .SIMD(DSP_SIMD), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
			.ACCU_WIDTH(ACCU_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .SEGMENTLEN(SEGMENTLEN),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk(dsp_clk), .rst, .en(dsp_en),
				.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
				.vld(dsp_vld), .p(dsp_p)
			);
		"mvu_4sx4u":
			mvu_4sx4u #(.PE(PE), .SIMD(DSP_SIMD), .ACCU_WIDTH(ACCU_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), 
			.VERSION(2), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk(dsp_clk), .rst, .en(dsp_en),
				.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
				.vld(dsp_vld), .p(dsp_p)
			);
		"mvu_8sx8u_dsp48":
			mvu_8sx8u_dsp48 #(.PE(PE), .SIMD(DSP_SIMD), .ACCU_WIDTH(ACCU_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk(dsp_clk), .rst, .en(dsp_en),
				.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
				.vld(dsp_vld), .p(dsp_p)
			);
		default: initial begin
			$error("Unrecognized COMPUTE_CORE '%s'", COMPUTE_CORE);
			$finish;
		end
		endcase

	end : blkDsp

//-------------------- Output register slice --------------------\\
	// Make `en`computation independent from external inputs.
	// Drive all outputs from registers.
	struct packed {
		logic rdy;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  A = '{ rdy: 1, default: 'x };	// side-step register used when encountering backpressure
	struct packed {
		logic vld;
		logic [PE-1:0][ACCU_WIDTH-1:0] dat;
	}  B = '{ vld: 0, default: 'x };	// ultimate output register

	assign	en = A.rdy;
	uwire  b_load = !B.vld || m_axis_c_tready;

	always_ff @(posedge clk) begin
		if(rst) begin
			A <= '{ rdy: 1, default: 'x };
			B <= '{ vld: 0, default: 'x };
		end
		else begin
			if(A.rdy)  A.dat <= odat;
			A.rdy <= (A.rdy && !ovld) || b_load;

			if(b_load) begin
				B <= '{
					vld: ovld || !A.rdy,
					dat: A.rdy? odat : A.dat
				};
			end
		end
	end
	assign	m_axis_c_tvalid = B.vld;
	// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
	// These extra bits should never be used. Why not 'x them out?
	assign	m_axis_c_tdata  = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){B.dat[PE-1][ACCU_WIDTH-1]}}, B.dat};

endmodule : mvau