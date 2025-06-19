############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""
BrainSmith HLS Templates

This module provides HLS code generation templates for BrainSmith custom operations.
Templates are compatible with FINN's template system for seamless integration.
"""

# Template for C++ simulation (cppsim) code generation
docompute_template = """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}
"""

# Template for IP generation TCL script
ipgentcl_template = """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"
set config_bnnlibdir "$::env(FINN_DEPS_DIR)/finn-hlslib"
puts "finn-hlslib dir: $config_bnnlibdir"
set config_customhlsdir "$::env(BSMITH_DIR)/brainsmith/hw_kernels/hls"
puts "BrainSmith custom HLS dir: $config_customhlsdir"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

$DEFAULT_DIRECTIVES$
$EXTRA_DIRECTIVES$

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
"""

# Template for IP generation with timeout support
docompute_template_timeout = """
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

unsigned timeout = 0;
while(timeout < $TIMEOUT_VALUE$){

$DOCOMPUTE$

if($TIMEOUT_CONDITION$){
timeout++;
}

else{
$TIMEOUT_READ_STREAM$
timeout = 0;
}
}

$DATAOUTSTREAM$

$SAVEASCNPY$

}
"""

# Template for IP generation C++ code
ipgen_template = """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$

#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
"""
