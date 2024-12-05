# flake8: noqa
# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# tcl script for IP generation
ipgentcl_template = """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"
set config_bnnlibdir "$::env(FINN_ROOT)/deps/finn-hlslib"
puts "finn-hlslib dir: $config_bnnlibdir"
set config_customhlsdir "$::env(FINN_ROOT)/custom_hls"
puts "custom HLS dir: $config_customhlsdir"
set config_bshlsdir "$::env(FINN_ROOT)/deps/finnbrainsmith/hlslib_extensions"
puts "BrainSmith HLS dir: $config_bshlsdir"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir -I$config_bshlsdir"

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
