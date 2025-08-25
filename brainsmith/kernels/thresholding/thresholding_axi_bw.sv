/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	All-AXI interface adapter for thresholding module.
 * @author	Thomas B. Preußer <tpreusse@amd.com>
 *
 * @description
 *	This AXI adapter fits the core thresholding functionality:
 *	- with AXI stream data interfaces with flow control
 *	- with implicit round-robin channel rotation as used by FINN, and
 *	- performs aligned byte address to parameter word address translation.
 *****************************************************************************/


// @brainsmith DATATYPE_CONSTRAINT input * 1 32
// @brainsmith DATATYPE_CONSTRAINT output * 1 32
// @brainsmith DATATYPE threshold width T_WIDTH
// @brainsmith BDIM input input_BDIM SHAPE=[CHANNELS]
// @brainsmith SDIM input input_SDIM SHAPE=[PE]
// *NOTE: This PE should really be SIMD
// @brainsmith AXILITE_PARAM USE_AXILITE threshold enable
// @brainsmith WEIGHT threshold

module thresholding_axi #(
    // Interface Parallelism
	int unsigned  input_BDIM = 1, // Channels
	int unsigned  input_SDIM = 1, // Processing Parallelism, requires input BDIM % SDIM = 0

    // Interface Datatype
	int unsigned  input_WIDTH,    // input precision
	int unsigned  output_WIDTH,   // output precision
	int unsigned  T_WIDTH,        // threshold precision
	bit  input_SIGNED = 1,	// signed inputs
	bit  input_FPARG  = 0,	// floating-point inputs: [sign] | exponent | mantissa

	int  BIAS  = 0,	// offsetting the output [0, 2^output_WIDTH-1] -> [BIAS, 2^output_WIDTH-1 + BIAS]

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",

	bit  USE_AXILITE,	// Implement AXI-Lite for threshold read/write

	// Force Use of On-Chip Memory Blocks
	int unsigned  DEPTH_TRIGGER_URAM = 0,	// if non-zero, local mems of this depth or more go into URAM (prio)
	int unsigned  DEPTH_TRIGGER_BRAM = 0,	// if non-zero, local mems of this depth or more go into BRAM
	bit  DEEP_PIPELINE = 0,

	localparam int unsigned  CF = input_BDIM/input_SDIM,	// Channel Fold
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(input_SDIM) + output_WIDTH + 2,
	localparam int unsigned  O_BITS = BIAS >= 0?
		/* unsigned */ $clog2(2**output_WIDTH+BIAS) :
		/* signed */ 1+$clog2(-BIAS >= 2**(output_WIDTH-1)? -BIAS : 2**output_WIDTH+BIAS)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  threshold_AWVALID,
	output	logic                  threshold_AWREADY,
	input	logic [ADDR_BITS-1:0]  threshold_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input	logic         threshold_WVALID,
	output	logic         threshold_WREADY,
	input	logic [31:0]  threshold_WDATA,
	input	logic [ 3:0]  threshold_WSTRB,

	output	logic        threshold_BVALID,
	input	logic        threshold_BREADY,
	output	logic [1:0]  threshold_BRESP,

	// Reading
	input	logic                  threshold_ARVALID,
	output	logic                  threshold_ARREADY,
	input	logic [ADDR_BITS-1:0]  threshold_ARADDR,

	output	logic         threshold_RVALID,
	input	logic         threshold_RREADY,
	output	logic [31:0]  threshold_RDATA,
	output	logic [ 1:0]  threshold_RRESP,

	//- AXI Stream - Input --------------
	output	logic  input_tready,
	input	logic  input_tvalid,
	input	logic [((input_SDIM*input_WIDTH+7)/8)*8-1:0]  input_tdata,

	//- AXI Stream - Output -------------
	input	logic  output_tready,
	output	logic  output_tvalid,
	output	logic [((input_SDIM*O_BITS+7)/8)*8-1:0]  output_tdata
);

	//-----------------------------------------------------------------------
	// AXI-lite Configuration Interface
	uwire  cfg_en;
	uwire  cfg_we;
	uwire [ADDR_BITS-3:0]  cfg_a;
	uwire [T_WIDTH       -1:0]  cfg_d;
	uwire  cfg_rack;
	uwire [T_WIDTH       -1:0]  cfg_q;

	if(USE_AXILITE) begin
		uwire [ADDR_BITS-1:0]  cfg_a0;
		axi4lite_if #(.ADDR_WIDTH(ADDR_BITS), .DATA_WIDTH(32), .IP_DATA_WIDTH(T_WIDTH)) axi (
			.aclk(ap_clk), .aresetn(ap_rst_n),

			.awready(threshold_AWREADY), .awvalid(threshold_AWVALID), .awaddr(threshold_AWADDR), .awprot('x),
			.wready(threshold_WREADY),   .wvalid(threshold_WVALID),   .wdata(threshold_WDATA),   .wstrb(threshold_WSTRB),
			.bready(threshold_BREADY),   .bvalid(threshold_BVALID),   .bresp(threshold_BRESP),

			.arready(threshold_ARREADY), .arvalid(threshold_ARVALID), .araddr(threshold_ARADDR), .arprot('x),
			.rready(threshold_RREADY),   .rvalid(threshold_RVALID),   .rresp(threshold_RRESP),   .rdata(threshold_RDATA),

			.ip_en(cfg_en), .ip_wen(cfg_we), .ip_addr(cfg_a0), .ip_wdata(cfg_d),
			.ip_rack(cfg_rack), .ip_rdata(cfg_q)
		);
		assign	cfg_a = cfg_a0[ADDR_BITS-3:0];
		always_ff @(posedge ap_clk) begin
			assert(!ap_rst_n || !cfg_en || (cfg_a0[ADDR_BITS-2+:2] === 3'h0)) else begin
				$error("%m: Spurious high address bits.");
			end
		end
	end
	else begin
		assign	cfg_en =  0;
		assign	cfg_we = 'x;
		assign	cfg_a  = 'x;
		assign	cfg_d  = 'x;
	end

	//-----------------------------------------------------------------------
	// Cast Inputs into Threshold Data Type
	uwire [input_SDIM-1:0][T_WIDTH-1:0]  idat;
	for(genvar  pe = 0; pe < input_SDIM; pe++) begin
		if(T_WIDTH == input_WIDTH) begin : genCopy
			assign	idat[pe] = input_tdata[pe*input_WIDTH+:input_WIDTH];
		end : genCopy
		else begin
			initial begin
				if(input_FPARG) begin
					$error("%m: Can't cast floating-point type.");
					$finish;
				end
			end

			if(T_WIDTH > input_WIDTH) begin : genWiden
				assign	idat[pe] = { {(T_WIDTH-input_WIDTH){input_SIGNED? input_tdata[(pe+1)*input_WIDTH-1] : 1'b0}}, input_tdata[pe*input_WIDTH+:input_WIDTH] };
			end : genWiden
			else begin : genNarrow
				// Saturate for clipping inputs
				if(!input_SIGNED) begin
					assign	idat[pe] = |input_tdata[pe*input_WIDTH+T_WIDTH+:input_WIDTH-T_WIDTH]? '1 : input_tdata[pe*input_WIDTH+:T_WIDTH];
				end
				else begin
					assign	idat[pe] =
						(input_tdata[pe*input_WIDTH+T_WIDTH+:input_WIDTH-T_WIDTH] == '1) || (input_tdata[pe*input_WIDTH+T_WIDTH+:input_WIDTH-T_WIDTH] == '0)? input_tdata[pe*input_WIDTH+:T_WIDTH] :
						{input_tdata[(pe+1)*input_WIDTH-1], {(T_WIDTH-1){!input_tdata[(pe+1)*input_WIDTH-1]}}};
				end
			end : genNarrow
		end
	end

	//-----------------------------------------------------------------------
	// Kernel Implementation
	thresholding #(
		.output_WIDTH(output_WIDTH), .K(T_WIDTH), .input_BDIM(input_BDIM), .input_SDIM(input_SDIM),
		.input_SIGNED(input_SIGNED), .input_FPARG(input_FPARG), .BIAS(BIAS),
		.THRESHOLDS_PATH(THRESHOLDS_PATH), .USE_CONFIG(USE_AXILITE),
		.DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM), .DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
		.DEEP_PIPELINE(DEEP_PIPELINE)
	) impl (
		.clk(ap_clk), .rst(!ap_rst_n),

		.cfg_en, .cfg_we, .cfg_a, .cfg_d,
		.cfg_rack, .cfg_q,

		.irdy(input_tready), .ivld(input_tvalid), .idat,
		.ordy(output_tready), .ovld(output_tvalid), .odat(output_tdata[input_SDIM*O_BITS-1:0])
	);
	if($bits(output_tdata) > input_SDIM*O_BITS) begin : genPadOut
		assign	output_tdata[$left(output_tdata):input_SDIM*O_BITS] = '0;
	end : genPadOut

endmodule : thresholding_axi
