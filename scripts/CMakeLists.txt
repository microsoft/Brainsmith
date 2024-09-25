# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is subject to the Xilinx Design License Agreement located
# in the LICENSE.md file in the root directory of this repository.
#
# This file contains confidential and proprietary information of Xilinx, Inc.
# and is protected under U.S. and international copyright and other
# intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in
# any application requiring failsafe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
cmake_minimum_required(VERSION 3.0)
project(finn_v80)

set(IPREPO_DIR ${CMAKE_BINARY_DIR}/iprepo)
file(MAKE_DIRECTORY ${IPREPO_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

#
# Check Python Version
#
#find_package(Python3 3.8 REQUIRED)
#message(STATUS "Found Python Version: ${Python3_VERSION}")

#
# FINN
#
set(FINN_PATH "${CMAKE_SOURCE_DIR}/scripts_finn/" CACHE STRING "FINN path.")
set(GEN_FINN_PATH "" CACHE STRING "Generated FINN partitions path.")
set(SHELL_XRT 0 CACHE STRING ".master level only.")

#
# Build optimizations
#
set(BUILD_OPT 1 CACHE STRING "Build optimizations (increases compilation time).")
set(SIM_RUN 0 CACHE STRING "Run command line simulation during `make sim`.")

#
# Device
#
#set(FPGA_PART xcv80-lsva4737-2MHP-e-S CACHE STRING "FPGA device.")
set(FPGA_PART xcu250-figd2104-2L-e CACHE STRING "FPGA device.")
set(ACLK_F 250 CACHE STRING "Clock frequency.")
set(EN_XTERM 1 CACHE STRING "Terminal.")
set(TB_NAME "tb_partitions_012" CACHE STRING "testbench dir.")

function(period_calc expr out)
    execute_process(COMMAND awk "BEGIN {printf ${expr}}" OUTPUT_VARIABLE __out)
    set(${out} ${__out} PARENT_SCOPE)
endfunction()

period_calc("1000.0 / ${ACLK_F}" ACLK_P)
MATH(EXPR ACLK_F_DP "${ACLK_F} * 2")

#
# Read config
#

# Config files
set(JSON_FILES
    "${CMAKE_SOURCE_DIR}/config/folding_partition_0.json"
    "${CMAKE_SOURCE_DIR}/config/folding_partition_1.json"
    "${CMAKE_SOURCE_DIR}/config/folding_partition_2.json"
    "${CMAKE_SOURCE_DIR}/config/top.json"
)

# Read JSON
set(JSON_FILES_COMMAND)
foreach(JSON_FILE IN LISTS JSON_FILES)
    list(APPEND JSON_FILES_COMMAND ${JSON_FILE})
endforeach()

execute_process(
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/read_json.py ${JSON_FILES_COMMAND}
    OUTPUT_VARIABLE JSON_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "\n" ";" JSON_LINES ${JSON_OUTPUT})

# Set vars
function(set_cmake_variable VAR_NAME VAR_VALUE)
    set(${VAR_NAME} ${VAR_VALUE} CACHE INTERNAL "")
endfunction()

foreach(line ${JSON_LINES})
    string(REGEX MATCH "([^=]+)=(.*)" _ ${line})
    set_cmake_variable(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
endforeach()

#
# FINN gen
#

set(GEN_FINN 0)
if(GEN_FINN_PATH STREQUAL "") 
  set(GEN_FINN 1)
  set(GEN_FINN_PATH ${CMAKE_BINARY_DIR})
else()
  set(GEN_FINN 0)
endif()

#
# Find Vivado
#

find_package(Vivado REQUIRED)
if (NOT VIVADO_FOUND)
   message(FATAL_ERROR "Vivado not found.")
endif()

find_package(VivadoHLS REQUIRED)
if (NOT VIVADO_HLS_FOUND)
  message(FATAL_ERROR "Vivado HLS not found.")
endif()

#
# Build dirs
#

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/checkpoints")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/reports")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bitstreams")

#
# Targets
#

# Package
configure_file(${CMAKE_SOURCE_DIR}/hw/hdl/intf/pkt_types.sv.in ${CMAKE_BINARY_DIR}/pkt_types.sv)

# HLS cores
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/hls")
set(HLS_CORES "shuffleA" "shuffleB" "shuffleC" "softmaxquant" "layernorm" )
foreach(HLS_CORE ${HLS_CORES})
  configure_file(${CMAKE_SOURCE_DIR}/hw/hls/${HLS_CORE}/${HLS_CORE}.cpp.in ${CMAKE_BINARY_DIR}/hls/${HLS_CORE}.cpp)
  if(EXISTS ${CMAKE_SOURCE_DIR}/hw/hls/${HLS_CORE}/${HLS_CORE}_tb.cpp.in)
    configure_file(${CMAKE_SOURCE_DIR}/hw/hls/${HLS_CORE}/${HLS_CORE}_tb.cpp.in ${CMAKE_BINARY_DIR}/hls/${HLS_CORE}_tb.cpp)
  endif()
endforeach()

# Scripts
configure_file(${CMAKE_SOURCE_DIR}/scripts_finn/build_all.py.in ${CMAKE_BINARY_DIR}/build_all.py)
configure_file(${CMAKE_SOURCE_DIR}/scripts/bd_gen.py.in ${CMAKE_BINARY_DIR}/bd_gen.py)
configure_file(${CMAKE_SOURCE_DIR}/scripts/base.tcl.in ${CMAKE_BINARY_DIR}/base.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/comp_hls.tcl.in ${CMAKE_BINARY_DIR}/comp_hls.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/cr_bd.tcl.in ${CMAKE_BINARY_DIR}/cr_bd.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/inst_ip.tcl.in ${CMAKE_BINARY_DIR}/inst_ip.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/create_project.tcl.in ${CMAKE_BINARY_DIR}/create_project.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/cr_sim.tcl.in ${CMAKE_BINARY_DIR}/cr_sim.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/compile.tcl.in ${CMAKE_BINARY_DIR}/compile.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/create_export.tcl.in ${CMAKE_BINARY_DIR}/create_export.tcl)


# Commands
set(DOCKER_SCRIPT_PATH "${FINN_PATH}finn")

set(FINN_GEN_CMD COMMAND ${CMAKE_COMMAND} -E chdir ${DOCKER_SCRIPT_PATH} ${CMAKE_COMMAND} -E env bash run-docker.sh build_custom ${CMAKE_BINARY_DIR} build_all)
set(HLS_SYNTH_CMD COMMAND ${VIVADO_HLS_BINARY} -f comp_hls.tcl -tclargs ${target} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set(BUILD_PRJCT COMMAND ${VIVADO_BINARY} -mode tcl -source ${CMAKE_BINARY_DIR}/create_project.tcl -notrace WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set(COMP_CMD COMMAND ${VIVADO_BINARY} -mode tcl -source ${CMAKE_BINARY_DIR}/compile.tcl -notrace WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set(BUILD_SIM COMMAND ${VIVADO_BINARY} -mode tcl -source ${CMAKE_BINARY_DIR}/cr_sim.tcl -notrace WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set(BUILD_EXPORT COMMAND ${VIVADO_BINARY} -mode tcl -source ${CMAKE_BINARY_DIR}/create_export.tcl -notrace WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

if(VITIS_HLS) 
  message("** Vitis toolchain")
else()
  message("** Vivado toolchain")
endif()

#
# RUN
#

if(GEN_FINN) 
  add_custom_target(project
    ${FINN_GEN_CMD}  
    ${HLS_SYNTH_CMD}
    ${BUILD_PRJCT}
  )

  add_custom_target(sim
    ${FINN_GEN_CMD} 
    ${HLS_SYNTH_CMD}
    ${BUILD_SIM}
  )

  add_custom_target(export
    ${FINN_GEN_CMD}
    ${HLS_SYNTH_CMD}
    ${BUILD_EXPORT}
  )
else()
  add_custom_target(project
    ${BUILD_PRJCT}
  )

  add_custom_target(sim 
    ${BUILD_SIM}
  )
endif()

add_custom_target(compile
    ${COMP_CMD}
  )
