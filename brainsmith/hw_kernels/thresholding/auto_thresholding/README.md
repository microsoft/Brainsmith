# Auto-Generated Thresholding HWCustomOp

This directory contains the FINN registration modules for the auto-generated thresholding hardware kernel.

## Structure

- `hls.py` - Registers the HLS implementation with domain `brainsmith.hw_kernels.thresholding.auto_thresholding.hls`
- `rtl.py` - Registers the RTL implementation with domain `brainsmith.hw_kernels.thresholding.auto_thresholding.rtl`

## Usage

The modules are automatically imported when needed. FINN will recognize operations with these domains and use the registered classes.

## Implementation

The actual implementation is in:
- HWCustomOp: `../bsmith/thresholding_axi_hw_custom_op.py`
- RTL Backend: `../bsmith/thresholding_axi_rtl.py`

These were auto-generated from the SystemVerilog RTL using the Hardware Kernel Generator.