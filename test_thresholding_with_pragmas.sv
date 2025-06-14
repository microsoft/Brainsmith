/******************************************************************************
 * Test version of thresholding_axi.sv with Brainsmith pragmas for end-to-end testing
 *****************************************************************************/

// @brainsmith BDIM s_axis -1 [PE]
// @brainsmith BDIM m_axis -1 [PE] 
// @brainsmith DATATYPE s_axis FIXED 8 8
// @brainsmith DATATYPE m_axis FIXED 8 8
// @brainsmith INPUT s_axis
// @brainsmith OUTPUT m_axis
// @brainsmith CONFIG s_axilite

module thresholding_axi #(
	int unsigned  N = 8,	// output precision
	int unsigned  WI = 8,	// input precision
	int unsigned  WT = 8,	// threshold precision
	int unsigned  C = 1,	// Channels
	int unsigned  PE = 1,	// Processing Parallelism, requires C = k*PE

	bit  SIGNED = 1,	// signed inputs
	bit  FPARG  = 0,	// floating-point inputs: [sign] | exponent | mantissa
	int  BIAS   = 0,	// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",

	bit  USE_AXILITE = 1,	// Implement AXI-Lite for threshold read/write

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
	input	logic  ap_start,
	output	logic  ap_done,
	output	logic  ap_idle,
	output	logic  ap_ready,

	//- AXI Lite ------------------------
	// Writing
	input	logic                  s_axilite_AWVALID,
	output	logic                  s_axilite_AWREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,

	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	// Reading
	input	logic                  s_axilite_ARVALID,
	output	logic                  s_axilite_ARREADY,
	input	logic [ADDR_BITS-1:0]  s_axilite_ARADDR,

	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Input --------------
	output	logic  s_axis_TREADY,
	input	logic  s_axis_TVALID,
	input	logic [((PE*WI+7)/8)*8-1:0]  s_axis_TDATA,

	//- AXI Stream - Output -------------
	input	logic  m_axis_TREADY,
	output	logic  m_axis_TVALID,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  m_axis_TDATA
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

			.awready(s_axilite_AWREADY), .awvalid(s_axilite_AWVALID), .awaddr(s_axilite_AWADDR), .awprot('x),
			.wready(s_axilite_WREADY),   .wvalid(s_axilite_WVALID),   .wdata(s_axilite_WDATA),   .wstrb(s_axilite_WSTRB),
			.bready(s_axilite_BREADY),   .bvalid(s_axilite_BVALID),   .bresp(s_axilite_BRESP),

			.arready(s_axilite_ARREADY), .arvalid(s_axilite_ARVALID), .araddr(s_axilite_ARADDR), .arprot('x),
			.rready(s_axilite_RREADY),   .rvalid(s_axilite_RVALID),   .rresp(s_axilite_RRESP),   .rdata(s_axilite_RDATA),

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
			assign	idat[pe] = s_axis_TDATA[pe*WI+:WI];
		end : genCopy
		else begin
			initial begin
				if(FPARG) begin
					$error("%m: Can't cast floating-point type.");
					$finish;
				end
			end

			if(WT > WI) begin : genWiden
				assign	idat[pe] = { {(WT-WI){SIGNED? s_axis_TDATA[(pe+1)*WI-1] : 1'b0}}, s_axis_TDATA[pe*WI+:WI] };
			end : genWiden
			else begin : genNarrow
				// Saturate for clipping inputs
				if(!SIGNED) begin
					assign	idat[pe] = |s_axis_TDATA[pe*WI+WT+:WI-WT]? '1 : s_axis_TDATA[pe*WI+:WT];
				end
				else begin
					assign	idat[pe] =
						(s_axis_TDATA[pe*WI+WT+:WI-WT] == '1) || (s_axis_TDATA[pe*WI+WT+:WI-WT] == '0)? s_axis_TDATA[pe*WI+:WT] :
						{s_axis_TDATA[(pe+1)*WI-1], {(WT-1){!s_axis_TDATA[(pe+1)*WI-1]}}};
				end
			end : genNarrow
		end
	end

	//-----------------------------------------------------------------------
	// Kernel Implementation (simplified for testing)
	
	// Simple pass-through for testing
	assign s_axis_TREADY = m_axis_TREADY;
	assign m_axis_TVALID = s_axis_TVALID;
	assign m_axis_TDATA = s_axis_TDATA;
	
	// Control signals
	assign ap_done = ap_start;
	assign ap_idle = !ap_start;
	assign ap_ready = 1'b1;

endmodule : thresholding_axi