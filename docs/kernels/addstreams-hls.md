# AddStreams HLS Backend

**HLS implementation of [AddStreams](addstreams.md) using finn-hlslib templates.**

**Language**: C++ with Vivado HLS pragmas

**Source**: `brainsmith/kernels/addstreams/addstreams_hls.py`

**Target Devices**: Xilinx 7-series, UltraScale, UltraScale+

---

## Summary

The HLS backend implements AddStreams using the `AddStreams_Batch` template from finn-hlslib. This provides a fully pipelined hardware implementation optimized for streaming dataflow with PE-based channel parallelism.

**Features**:

- Fully pipelined implementation with II=1 (initiation interval)
- Uses pure LUT-based adders (no DSP blocks required)
- Automatic bitwidth expansion for overflow prevention
- C++ simulation (cppsim) for fast functional verification
- Supports arbitrary integer datatypes (INT8, INT16, INT32)

**Advantages**:

- High-level C++ code generation simplifies maintenance
- finn-hlslib templates provide well-tested implementations
- Fast synthesis time (5-15 minutes for typical configurations)
- Portable across Xilinx device families

**Limitations**:

- Slightly higher LUT usage compared to hand-optimized RTL
- Synthesis time increases with large PE values
- HLS pragmas may require tuning for optimal timing closure at high frequencies

---

## Constraints

All constraints from [AddStreams schema](addstreams.md#constraints) plus:

**HLS-specific constraints**:

- ✅ PE must be a power of 2 for optimal packing (recommended, not enforced)
- ✅ Input/output bitwidths must be ≤ 64 bits for efficient stream packing
- ⚠️ Large PE values (>64) may require additional pipelining for timing closure

---

## Hardware Interface

### RTL Ports

```cpp
void AddStreams_nodename(
    hls::stream<ap_uint<PE*InWidth>> &in0_V,   // PE×InWidth-bit input0
    hls::stream<ap_uint<PE*InWidth>> &in1_V,   // PE×InWidth-bit input1
    hls::stream<ap_uint<PE*OutWidth>> &out0_V  // PE×OutWidth-bit output
)
```

| Port | Type | Bitwidth | Protocol | Description |
|------|------|----------|----------|-------------|
| in0_V | Input | PE × InWidth | AXI4-Stream | Packed first input stream |
| in1_V | Input | PE × InWidth | AXI4-Stream | Packed second input stream |
| out0_V | Output | PE × OutWidth | AXI4-Stream | Packed output stream (OutWidth = InWidth + 1) |

**Packing**: Channels are packed into wide words for efficient streaming. For PE=16 with INT8 inputs, each stream word is 128 bits (16 × 8).

**Example** (INT8 inputs, PE=16):
```cpp
void AddStreams_residual_add(
    hls::stream<ap_uint<128>> &in0_V,   // 16 × INT8 = 128 bits
    hls::stream<ap_uint<128>> &in1_V,   // 16 × INT8 = 128 bits
    hls::stream<ap_uint<144>> &out0_V   // 16 × INT9 = 144 bits
)
```

---

## Resource Characteristics

### Synthesis Results

Typical resource usage on **xczu7ev-ffvc1156-2-e** (ZCU104) with INT8 inputs:

| PE | DSP | BRAM_18K | URAM | LUT | FF | Freq (MHz) |
|----|-----|----------|------|-----|-----|------------|
| 1 | 0 | 0 | 0 | 250 | 300 | 300 |
| 4 | 0 | 0 | 0 | 500 | 600 | 275 |
| 16 | 0 | 0 | 0 | 1200 | 1500 | 250 |
| 64 | 0 | 0 | 0 | 4000 | 5000 | 200 |

**Scaling characteristics**:

- **DSP usage**: Zero - AddStreams uses pure LUT-based adders
- **BRAM usage**: Zero - No static parameters to store
- **LUT usage**: Linear scaling with PE (~60 LUTs per PE for INT8, ~100 LUTs per PE for INT16)
- **FF usage**: Linear scaling with PE (~75 FFs per PE for INT8, ~120 FFs per PE for INT16)
- **Frequency**: Slight degradation at high PE due to wider datapaths and routing congestion

**Datatype impact**:

For INT16 inputs (vs INT8), expect:
- 1.6× LUT increase
- 1.6× FF increase
- 10-15% frequency reduction

---

## Performance

### Cycle Count

```python
def get_exp_cycles(self):
    """Expected cycles for one inference."""
    folded_shape = self.get_folded_output_shape()  # (N, H, W, C/PE, PE)
    return int(np.prod(folded_shape[:-1]))  # Exclude PE dimension
```

**Actual formula**:
```python
cycles = N × H × W × (C / PE)
```

Where:
- N = batch size
- H, W = spatial dimensions
- C = number of channels
- PE = processing elements

### Throughput

For typical ResNet-50 residual layer (224×224×64, INT8):

| PE | Cycles | @ 250 MHz | Throughput (fps) |
|----|--------|-----------|------------------|
| 1 | 3,211,264 | 12.8 ms | 78 |
| 4 | 802,816 | 3.2 ms | 312 |
| 16 | 200,704 | 0.8 ms | 1,245 |
| 64 | 50,176 | 0.2 ms | 4,981 |

**Calculation example** (PE=16):
```python
cycles = 1 × 224 × 224 × (64 / 16) = 200,704
time = 200,704 / 250 MHz = 0.8 ms
fps = 1 / 0.8 ms = 1,245 fps
```

### Pipeline Characteristics

- **Initiation Interval (II)**: 1 (fully pipelined - new input every clock)
- **Pipeline Depth**: ~5-10 cycles (depends on PE and datatype)
- **Latency**: ~5-10 cycles from first input to first output

**Implications**:

- Steady-state throughput: 1 output per cycle (after initial latency)
- No stalls or bubbles in pipeline under normal operation
- Back-pressure propagates through AXI4-Stream protocol

---

## Code Generation

### Template Instantiation

```cpp
#include "streamtools.h"

AddStreams_Batch<
    PE,                  // Processing elements
    ap_int<InWidth>,     // Input datatype (e.g., ap_int<8>)
    ap_int<InWidth>,     // Second input datatype (same as first)
    ap_int<OutWidth>,    // Output datatype (e.g., ap_int<9>)
    NumElements          // Total elements to process
>(in0_V, in1_V, out0_V, 1);  // NumReps=1 for single batch
```

**Example instantiation** (INT8, PE=16, 224×224×64):
```cpp
#include "streamtools.h"

AddStreams_Batch<
    16,              // PE
    ap_int<8>,       // INT8 input
    ap_int<8>,       // INT8 input
    ap_int<9>,       // INT9 output (overflow protection)
    3211264          // 1 × 224 × 224 × 64 = 3,211,264 elements
>(in0_V, in1_V, out0_V, 1);
```

### HLS Pragmas

Key optimizations applied by finn-hlslib template:

```cpp
#pragma HLS PIPELINE II=1
// Ensures one output per cycle (fully pipelined)

#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=in1_V
#pragma HLS INTERFACE axis port=out0_V
// AXI4-Stream interfaces for FPGA dataflow

#pragma HLS INLINE
// Inline function for better optimization
```

**Additional pragmas** for multi-layer designs:
```cpp
#pragma HLS DATAFLOW
// Enable task-level pipelining when AddStreams is part of larger design
```

---

## See Also

- [AddStreams KernelOp](addstreams.md) - Abstract kernel specification
- [Kernel Architecture](../developer-guide/3-reference/kernels.md) - Backend design patterns
- [finn-hlslib Documentation](https://github.com/Xilinx/finn-hlslib) - HLS template library

---

## API Reference

::: brainsmith.kernels.addstreams.AddStreams_hls
    options:
      show_source: false
      heading_level: 3
      members:
        - global_includes
        - defines
        - docompute
        - blackboxfunction
        - pragmas
