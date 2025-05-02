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

module thresholding_axi #(
	int unsigned  N,	// output precision
	int unsigned  WI,	// input precision
	int unsigned  WT,	// threshold precision
	int unsigned  C = 1,	// Channels
	int unsigned  PE = 1,	// Processing Parallelism, requires C = k*PE

	bit  SIGNED = 1,	// signed inputs
	bit  FPARG  = 0,	// floating-point inputs: [sign] | exponent | mantissa
	int  BIAS   = 0,	// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",

	bit  USE_AXILITE,	// Implement AXI-Lite for threshold read/write

	// Force Use of On-Chip Memory Blocks
	int unsigned  DEPTH_TRIGGER_URAM = 0,	// if non-zero, local mems of this depth or more go into URAM (prio)
	int unsigned  DEPTH_TRIGGER_BRAM = 0,	// if non-zero, local mems of this depth or more go into BRAM
	bit  DEEP_PIPELINE = 0,

	localparam int unsigned  CF = C/PE,	// Channel Fold
	localparam int unsigned  ADDR_BITS = $clog2(CF) + $clog2(PE) + N + 2,
	localparam int unsigned  O_BITS = BIAS >= 0?
		/* unsigned */ $clog2(2**N+BIAS) :
		/* signed */ 1+$clog2(-BIAS >= 2**(N-1)? -BIAS : 2**N+BIAS)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  config_AWVALID,
	output	logic                  config_AWREADY,
	input	logic [ADDR_BITS-1:0]  config_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input	logic         config_WVALID,
	output	logic         config_WREADY,
	input	logic [31:0]  config_WDATA,
	input	logic [ 3:0]  config_WSTRB,

	output	logic        config_BVALID,
	input	logic        config_BREADY,
	output	logic [1:0]  config_BRESP,

	// Reading
	input	logic                  config_ARVALID,
	output	logic                  config_ARREADY,
	input	logic [ADDR_BITS-1:0]  config_ARADDR,

	output	logic         config_RVALID,
	input	logic         config_RREADY,
	output	logic [31:0]  config_RDATA,
	output	logic [ 1:0]  config_RRESP,

	//- AXI Stream - Input --------------
	output	logic  in0_TREADY,
	input	logic  in0_TVALID,
	input	logic [((PE*WI+7)/8)*8-1:0]  in0_TDATA,

	//- AXI Stream - Output -------------
	input	logic  out0_TREADY,
	output	logic  out0_TVALID,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  out0_TDATA
);

	//-----------------------------------------------------------------------
	// AXI-lite Configuration Interface
	uwire  cfg_en;
	uwire  cfg_we;
	uwire [ADDR_BITS-3:0]  cfg_a;
	uwire [WT       -1:0]  cfg_d;
	uwire  cfg_rack;
	uwire [WT       -1:0]  cfg_q;

	if(USE_AXILITE) begin
		uwire [ADDR_BITS-1:0]  cfg_a0;
		axi4lite_if #(.ADDR_WIDTH(ADDR_BITS), .DATA_WIDTH(32), .IP_DATA_WIDTH(WT)) axi (
			.aclk(ap_clk), .aresetn(ap_rst_n),

			.awready(config_AWREADY), .awvalid(config_AWVALID), .awaddr(config_AWADDR), .awprot('x),
			.wready(config_WREADY),   .wvalid(config_WVALID),   .wdata(config_WDATA),   .wstrb(config_WSTRB),
			.bready(config_BREADY),   .bvalid(config_BVALID),   .bresp(config_BRESP),

			.arready(config_ARREADY), .arvalid(config_ARVALID), .araddr(config_ARADDR), .arprot('x),
			.rready(config_RREADY),   .rvalid(config_RVALID),   .rresp(config_RRESP),   .rdata(config_RDATA),

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
	uwire [PE-1:0][WT-1:0]  idat;
	for(genvar  pe = 0; pe < PE; pe++) begin
		if(WT == WI) begin : genCopy
			assign	idat[pe] = in0_TDATA[pe*WI+:WI];
		end : genCopy
		else begin
			initial begin
				if(FPARG) begin
					$error("%m: Can't cast floating-point type.");
					$finish;
				end
			end

			if(WT > WI) begin : genWiden
				assign	idat[pe] = { {(WT-WI){SIGNED? in0_TDATA[(pe+1)*WI-1] : 1'b0}}, in0_TDATA[pe*WI+:WI] };
			end : genWiden
			else begin : genNarrow
				// Saturate for clipping inputs
				if(!SIGNED) begin
					assign	idat[pe] = |in0_TDATA[pe*WI+WT+:WI-WT]? '1 : in0_TDATA[pe*WI+:WT];
				end
				else begin
					assign	idat[pe] =
						(in0_TDATA[pe*WI+WT+:WI-WT] == '1) || (in0_TDATA[pe*WI+WT+:WI-WT] == '0)? in0_TDATA[pe*WI+:WT] :
						{in0_TDATA[(pe+1)*WI-1], {(WT-1){!in0_TDATA[(pe+1)*WI-1]}}};
				end
			end : genNarrow
		end
	end

	//-----------------------------------------------------------------------
	// Kernel Implementation
	thresholding #(
		.N(N), .K(WT), .C(C), .PE(PE),
		.SIGNED(SIGNED), .FPARG(FPARG), .BIAS(BIAS),
		.THRESHOLDS_PATH(THRESHOLDS_PATH), .USE_CONFIG(USE_AXILITE),
		.DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM), .DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
		.DEEP_PIPELINE(DEEP_PIPELINE)
	) impl (
		.clk(ap_clk), .rst(!ap_rst_n),

		.cfg_en, .cfg_we, .cfg_a, .cfg_d,
		.cfg_rack, .cfg_q,

		.irdy(in0_TREADY), .ivld(in0_TVALID), .idat,
		.ordy(out0_TREADY), .ovld(out0_TVALID), .odat(out0_TDATA[PE*O_BITS-1:0])
	);
	if($bits(out0_TDATA) > PE*O_BITS) begin : genPadOut
		assign	out0_TDATA[$left(out0_TDATA):PE*O_BITS] = '0;
	end : genPadOut

endmodule : thresholding_axi
