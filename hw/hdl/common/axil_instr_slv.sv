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

module axil_instr_slv (
    input  logic                        aclk,
    input  logic                        aresetn,

    AXI4L.slave                         axi_ctrl,

    // Input
    input  logic                        ready_in,
    input  logic                        ready_out,
    input  logic [63:0]                 interval,
    input  logic [63:0]                 latency,                   
    input  logic [15:0]                 checksum,     
    input  logic                        overflow,

    output logic                        start,
    output logic [31:0]                 n_beats_in,
    output logic [31:0]                 n_runs_in_pow2,              
    output logic [31:0]                 n_beats_out,
    output logic [31:0]                 n_runs_out_pow2,     
    output logic [31:0]                 lfsr_seed
);

// -- Decl ----------------------------------------------------------
// ------------------------------------------------------------------
// Constants
localparam integer N_REGS = 11;
localparam integer AXIL_DATA_BITS = 32;
localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer AXIL_ADDR_BITS = ADDR_LSB + ADDR_MSB;

// Internal registers
logic [AXIL_ADDR_BITS-1:0] axi_awaddr;
logic axi_awready;
logic [AXIL_ADDR_BITS-1:0] axi_araddr;
logic axi_arready;
logic [1:0] axi_bresp;
logic axi_bvalid;
logic axi_wready;
logic [AXIL_DATA_BITS-1:0] axi_rdata;
logic [1:0] axi_rresp;
logic axi_rvalid;

// Registers
logic [N_REGS-1:0][AXIL_DATA_BITS-1:0] slv_reg;
logic slv_reg_rden;
logic slv_reg_wren;
logic aw_en;

// -- Def -----------------------------------------------------------
// ------------------------------------------------------------------

// -- Register map ----------------------------------------------------------------------- 
// 0 (W1S/R)  : Control/Status (
localparam integer CTRL_STAT_REG = 0;
    // CTRL
    localparam integer CTRL_START = 0; 
    // STAT
    localparam integer STAT_READY_IN = 0;
    localparam integer STAT_READY_OUT = 1;
    localparam integer STAT_OVERFLOW = 2;
// 1 (WR)   : Number of beats in
localparam integer N_BEATS_IN_REG = 1;
// 2 (WR)   : Number of beats out
localparam integer N_BEATS_OUT_REG = 2;
// 3 (WR)   : Number of runs in
localparam integer N_RUNS_IN_REG = 3;
// 4 (WR)   : Number of runs out
localparam integer N_RUNS_OUT_REG = 4;
// 5 (WR)   : LFSR seed
localparam integer SEED_REG = 5;
// 6 (RO)   : Interval low
localparam integer INT_LOW_REG = 6;
// 7 (RO)   : Interval high
localparam integer INT_HIGH_REG = 7;
// 8 (RO)   : Latency low
localparam integer LAT_LOW_REG = 8;
// 9 (RO)   : Latency high
localparam integer LAT_HIGH_REG = 9;
// 10 (RO)   : Checksum
localparam integer CSUM_REG = 10;

// Write process
assign slv_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 ) begin
    slv_reg <= 0;
  end
  else begin
    // Control
    start <= 1'b0;

    if(slv_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB])
        CTRL_STAT_REG:
            if(axi_ctrl.wstrb[0]) begin
                start <= 1'b1;
            end
        N_BEATS_IN_REG:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[N_BEATS_IN_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        N_BEATS_OUT_REG:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[N_BEATS_IN_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        N_RUNS_IN_REG:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[N_RUNS_IN_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        N_RUNS_OUT_REG:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[N_RUNS_OUT_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        SEED_REG:
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[SEED_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        
        default : ;
      endcase
    end
  end
end    

// Read process
assign slv_reg_rden = axi_arready & axi_ctrl.arvalid & ~axi_rvalid;

always_ff @(posedge aclk) begin
  if( aresetn == 1'b0 ) begin
    axi_rdata <= 0;
  end
  else begin
    if(slv_reg_rden) begin
      axi_rdata <= 0;
      
      case (axi_araddr[ADDR_LSB+:ADDR_MSB])
        CTRL_STAT_REG: begin
            axi_rdata[0] <= ready_in;
            axi_rdata[1] <= ready_out;
            axi_rdata[2] <= overflow;
        end
        N_BEATS_IN_REG:
            axi_rdata <= slv_reg[N_BEATS_IN_REG];
        N_BEATS_OUT_REG:
            axi_rdata <= slv_reg[N_BEATS_OUT_REG];
        N_RUNS_IN_REG:
            axi_rdata <= slv_reg[N_RUNS_IN_REG];
        N_RUNS_OUT_REG:
            axi_rdata <= slv_reg[N_RUNS_OUT_REG];
        SEED_REG:
            axi_rdata <= slv_reg[SEED_REG];
        INT_LOW_REG:
            axi_rdata <= interval[31:0];
        INT_HIGH_REG:
            axi_rdata <= interval[63:32];
        LAT_LOW_REG:
            axi_rdata <= latency[31:0];
        LAT_HIGH_REG:
            axi_rdata <= latency[63:32];
        CSUM_REG:
            axi_rdata[15:0] <= checksum;
        default: ;
      endcase
    end
  end 
end

// Output
always_comb begin
    n_beats_in = slv_reg[N_BEATS_IN_REG];
    n_beats_out = slv_reg[N_BEATS_OUT_REG];
    n_runs_in_pow2 = slv_reg[N_RUNS_IN_REG];
    n_runs_out_pow2 = slv_reg[N_RUNS_OUT_REG];
    lfsr_seed = slv_reg[SEED_REG];
end

// --------------------------------------------------------------------------------------
// AXI CTRL  
// -------------------------------------------------------------------------------------- 
// Don't edit

// I/O
assign axi_ctrl.awready = axi_awready;
assign axi_ctrl.arready = axi_arready;
assign axi_ctrl.bresp = axi_bresp;
assign axi_ctrl.bvalid = axi_bvalid;
assign axi_ctrl.wready = axi_wready;
assign axi_ctrl.rdata = axi_rdata;
assign axi_ctrl.rresp = axi_rresp;
assign axi_ctrl.rvalid = axi_rvalid;

// awready and awaddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_awready <= 1'b0;
      axi_awaddr <= 0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && axi_ctrl.awvalid && axi_ctrl.wvalid && aw_en)
        begin
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
          axi_awaddr <= axi_ctrl.awaddr;
        end
      else if (axi_ctrl.bready && axi_bvalid)
        begin
          aw_en <= 1'b1;
          axi_awready <= 1'b0;
        end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end  

// arready and araddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_arready <= 1'b0;
      axi_araddr  <= 0;
    end 
  else
    begin    
      if (~axi_arready && axi_ctrl.arvalid)
        begin
          axi_arready <= 1'b1;
          axi_araddr  <= axi_ctrl.araddr;
        end
      else
        begin
          axi_arready <= 1'b0;
        end
    end 
end    

// bvalid and bresp
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && axi_ctrl.awvalid && ~axi_bvalid && axi_wready && axi_ctrl.wvalid)
        begin
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0;
        end                   
      else
        begin
          if (axi_ctrl.bready && axi_bvalid) 
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end

// wready
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && axi_ctrl.wvalid && axi_ctrl.awvalid && aw_en )
        begin
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end  

// rvalid and rresp (1Del?)
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_rvalid <= 0;
      axi_rresp  <= 0;
    end 
  else
    begin    
      if (axi_arready && axi_ctrl.arvalid && ~axi_rvalid)
        begin
          axi_rvalid <= 1'b1;
          axi_rresp  <= 2'b0;
        end   
      else if (axi_rvalid && axi_ctrl.rready)
        begin
          axi_rvalid <= 1'b0;
        end                
    end
end    

endmodule // gbm slave