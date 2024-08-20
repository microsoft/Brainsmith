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

`ifndef AXI_INTF_SV_
`define AXI_INTF_SV_

import cTypes::*;

// ----------------------------------------------------------------------------
// AXI4 stream 
// ----------------------------------------------------------------------------
interface AXI4S #(
	parameter AXI4S_DATA_BITS = AXI_DATA_BITS
) (
    input  logic aclk
);

typedef logic [AXI4S_DATA_BITS-1:0] data_t;

data_t          tdata;
logic           tready;
logic           tvalid;

// Tie off unused master signals
task tie_off_m ();
    tdata      = 0;
    tvalid     = 1'b0;
endtask

// Tie off unused slave signals
task tie_off_s ();
    tready     = 1'b0;
endtask

// Master
modport m (
	import tie_off_m,
	input tready,
	output tdata, tvalid
);

// Slave
modport s (
    import tie_off_s,
    input tdata, tvalid,
    output tready
);

endinterface

`endif