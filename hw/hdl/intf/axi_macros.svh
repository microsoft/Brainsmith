/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

`ifndef AXI_MACROS_SVH_
`define AXI_MACROS_SVH_

`define AXIS_ASSIGN(s, m)              	            \
	assign m.tdata      = s.tdata;     	            \
	assign m.tvalid     = s.tvalid;    	            \
	assign s.tready     = m.tready;

`define AXIS_TIE_OFF_M(m)				            \
	assign m.tvalid		= 1'b0;			            \
	assign m.tdata		= 0;			            

`define AXIS_TIE_OFF_S(s)				            \
	assign s.tready		= 1'b1;			

`define AXIS_ASSIGN_S2I(s, m)                       \
    assign ``m``.tdata    = ``s``_tdata;            \
	assign ``m``.tvalid   = ``s``_tvalid;           \
	assign ``s``_tready   = ``m``.tready;

`define AXIS_ASSIGN_I2S(s, m)                       \
    assign ``m``_tdata    = ``s``.tdata;            \
	assign ``m``_tvalid   = ``s``.tvalid;           \
	assign ``s``.tready   = ``m``_tready;

`define AXIL_ASSIGN(s, m)              	            \
	assign m.araddr 	= s.araddr;		            \
	assign m.arvalid 	= s.arvalid;	            \
	assign m.awaddr		= s.awaddr;		            \
	assign m.awvalid	= s.awvalid;	            \
	assign m.bready 	= s.bready;		            \
	assign m.rready 	= s.rready; 	            \
	assign m.wdata		= s.wdata;		            \
	assign m.wstrb		= s.wstrb;		            \
	assign m.wvalid 	= s.wvalid;		            \
	assign s.arready 	= m.arready;	            \
	assign s.awready	= m.awready; 	            \
	assign s.bresp		= m.bresp;		            \
	assign s.bvalid 	= m.bvalid;		            \
	assign s.rdata		= m.rdata;		            \
	assign s.rresp		= m.rresp;		            \
	assign s.rvalid		= m.rvalid;		            \
	assign s.wready 	= m.wready;

`define AXIL_TIE_OFF_M(m)				            \
	assign m.araddr		= 0;			            \
	assign m.arvalid 	= 1'b0;			            \
	assign m.awaddr		= 0;			            \
	assign m.awvalid 	= 1'b0;			            \
	assign m.rready 	= 1'b1;			            \
	assign m.wdata 		= 0;			            \
	assign m.wstrb 		= 0;			            \
	assign m.valid 		= 1'b0;			            \
	assign m.bready 	= 1'b1;

`define AXIL_TIE_OFF_S(s)				            \
	assign s.arready	= 1'b1;			            \
	assign s.awready  	= 1'b1;			            \
	assign s.rdata 		= 0;			            \
	assign s.rresp 		= 0;			            \
	assign s.rvalid 	= 1'b0;			            \
	assign s.wready 	= 1'b0;			            \
	assign s.bresp 		= 0;			            \
	assign s.bvalid		= 1'b0;		

`define AXIL_ASSIGN_S2I(s, m)              	        \
	assign ``m``.araddr 	= ``s``_araddr;		    \
	assign ``m``.arvalid 	= ``s``_arvalid;	    \
	assign ``m``.awaddr		= ``s``_awaddr;		    \
	assign ``m``.awvalid	= ``s``_awvalid;	    \
	assign ``m``.bready 	= ``s``_bready;		    \
	assign ``m``.rready 	= ``s``_rready; 	    \
	assign ``m``.wdata		= ``s``_wdata;		    \
	assign ``m``.wstrb		= ``s``_wstrb;		    \
	assign ``m``.wvalid 	= ``s``_wvalid;		    \
	assign ``s``_arready 	= ``m``.arready;	    \
	assign ``s``_awready	= ``m``.awready; 	    \
	assign ``s``_bresp		= ``m``.bresp;		    \
	assign ``s``_bvalid 	= ``m``.bvalid;		    \
	assign ``s``_rdata		= ``m``.rdata;		    \
	assign ``s``_rresp		= ``m``.rresp;		    \
	assign ``s``_rvalid		= ``m``.rvalid;		    \
	assign ``s``_wready 	= ``m``.wready;	

`define AXIL_ASSIGN_I2S(s, m)              	        \
	assign ``m``_araddr 	= ``s``.araddr;		    \
	assign ``m``_arvalid 	= ``s``.arvalid;	    \
	assign ``m``_awaddr		= ``s``.awaddr;		    \
	assign ``m``_awvalid	= ``s``.awvalid;	    \
	assign ``m``_bready 	= ``s``.bready;		    \
	assign ``m``_rready 	= ``s``.rready; 	    \
	assign ``m``_wdata		= ``s``.wdata;		    \
	assign ``m``_wstrb		= ``s``.wstrb;		    \
	assign ``m``_wvalid 	= ``s``.wvalid;		    \
	assign ``s``.arready 	= ``m``_arready;	    \
	assign ``s``.awready	= ``m``_awready; 	    \
	assign ``s``.bresp		= ``m``_bresp;		    \
	assign ``s``.bvalid 	= ``m``_bvalid;		    \
	assign ``s``.rdata		= ``m``_rdata;		    \
	assign ``s``.rresp		= ``m``_rresp;		    \
	assign ``s``.rvalid		= ``m``_rvalid;		    \
	assign ``s``.wready 	= ``m``_wready;	

`endif