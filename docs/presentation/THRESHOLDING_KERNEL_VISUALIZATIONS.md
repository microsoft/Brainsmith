# Thresholding Kernel Visualizations

## Overview: Thresholding Operation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       THRESHOLDING KERNEL                               │
│                   Element-wise Comparison Operation                     │
└─────────────────────────────────────────────────────────────────────────┘

                Input                     Threshold                Output
                Value                      Array                   Index
                  │                          │                       │
                  ▼                          ▼                       ▼
            ┌─────────┐              ┌─────────────┐          ┌─────────┐
            │   42    │              │ T[0] = 10   │          │         │
            │         │   compare    │ T[1] = 25   │   ───►   │    2    │
            │  INT8   │     with     │ T[2] = 50   │          │  UINT2  │
            └─────────┘              │ T[3] = 100  │          └─────────┘
                                     └─────────────┘
                                           ▲
                                           │
                              42 > 25 && 42 < 50
                              Therefore: output = 2

Operation: For each input, find the threshold interval it falls into
```

## Dataflow Architecture from Pragmas

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLDING_AXI MODULE                              │
│                  (Based on pragma annotations)                          │
└─────────────────────────────────────────────────────────────────────────┘

Pragma Analysis:
─────────────────
@brainsmith DATATYPE input * 1 32      → Input: 1-32 bit flexible width
@brainsmith DATATYPE output * 1 32     → Output: 1-32 bit flexible width
@brainsmith DATATYPE_PARAM threshold width T_WIDTH → Threshold configurable
@brainsmith BDIM input input_BDIM SHAPE=[CHANNELS]  → Block = CHANNELS
@brainsmith SDIM input input_SDIM SHAPE=[PE]        → Stream = PE parallel
@brainsmith AXILITE_PARAM threshold USE_AXILITE     → Runtime config

                          ┌───────────────────────┐
                          │   USE_AXILITE=1       │
                          │  ┌────────────────┐   │
    AXI-Lite Interface ───┼─▶│   Threshold    │   │
    (Runtime Config)      │  │    Memory      │   │
                          │  │  T[0]...T[N]   │   │
                          │  └────────┬───────┘   │
                          │           │           │
                          │           ▼           │
Input Stream              │  ┌────────────────┐   │              Output Stream
[CHANNELS total]          │  │   Comparison   │   │              [CHANNELS total]
[PE parallel] ────────────┼─▶│     Logic      │───┼───────────▶ [PE parallel]
TDATA: PE×WIDTH bits      │  │  PE parallel   │   │              TDATA: PE×O_BITS
                          │  └────────────────┘   │
                          └───────────────────────┘

Key Parameters:
- input_BDIM = CHANNELS (total channels to process)
- input_SDIM = PE (parallelism factor)
- Channel Fold (CF) = CHANNELS/PE (cycles to process all channels)
```

## Data Hierarchy for Thresholding

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLDING DATA HIERARCHY                          │
└─────────────────────────────────────────────────────────────────────────┘

TENSOR LEVEL (Full inference data)
┌─────────────────────────────────────────────────────────────────────────┐
│  Example: 256 channels of activation data                               │
│  Shape: [1, 256] for element-wise operation                            │
│  Total elements: 256                                                    │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ Block Tiling
                            ▼
BLOCK LEVEL (Processing unit)
┌─────────────────────────────────────────────────────────────────────────┐
│  Block shape: [1, CHANNELS]                                             │
│  Example: [1, 256] - Process all channels as one block                 │
│  Note: For thresholding, typically BLOCK = TENSOR (no spatial tiling)  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ Stream Tiling
                            ▼
STREAM LEVEL (Per-clock processing)
┌─────────────────────────────────────────────────────────────────────────┐
│  Stream shape: [1, PE]                                                  │
│  Example: [1, 8] - Process 8 channels per clock cycle                  │
│  Cycles needed: CHANNELS/PE = 256/8 = 32 cycles                        │
│                                                                         │
│  Cycle 0: ch[0:7]    Cycle 1: ch[8:15]    ...    Cycle 31: ch[248:255]│
│    ┌─┬─┬─┬─┬─┬─┬─┬─┐   ┌─┬─┬─┬─┬─┬─┬─┬─┐        ┌─┬─┬─┬─┬─┬─┬─┬─┐  │
│    │0│1│2│3│4│5│6│7│   │8│9│A│B│C│D│E│F│  ...   │█│█│█│█│█│█│█│█│  │
│    └─┴─┴─┴─┴─┴─┴─┴─┘   └─┴─┴─┴─┴─┴─┴─┴─┘        └─┴─┴─┴─┴─┴─┴─┴─┘  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ Element Level
                            ▼
ELEMENT LEVEL (Individual values)
┌─────────────────────────────────────────────────────────────────────────┐
│  Input datatype: Configurable 1-32 bits (e.g., INT8)                   │
│  Output datatype: Configurable 1-32 bits (e.g., UINT2 for 4 classes)   │
│  Threshold datatype: T_WIDTH bits                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Kernel Model Representation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                THRESHOLDING KERNEL MODEL                                │
└─────────────────────────────────────────────────────────────────────────┘

KernelDefinition("thresholding")
│
├─ InputDefinition("input")
│  ├─ datatype_constraints: [INT/UINT, 1-32 bits]
│  ├─ block_tiling: [1, "CHANNELS"]
│  ├─ stream_tiling: [1, "PE"]
│  └─ is_weight: false
│
├─ InputDefinition("threshold")
│  ├─ datatype_constraints: [INT/UINT, T_WIDTH bits]
│  ├─ block_tiling: ["LEVELS", "CHANNELS"]
│  ├─ is_weight: true
│  └─ axilite_configurable: true
│
└─ OutputDefinition("output")
   ├─ datatype_constraints: [UINT, 1-32 bits]
   ├─ block_tiling: [1, "CHANNELS"]
   └─ computed_rate: PE elements/cycle

Relationships:
- input.dim[1] EQUAL output.dim[1]  (Same number of channels)
- threshold.dim[1] EQUAL input.dim[1] (Thresholds for each channel)
```

## Streaming Architecture Detail

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STREAMING DATAPATH (PE=8)                            │
└─────────────────────────────────────────────────────────────────────────┘

Clock Cycle N:
                          Threshold Memory
                         (Per-channel arrays)
                    ┌──────────────────────────┐
                    │ Ch0: [10, 25, 50, 100]  │
                    │ Ch1: [15, 30, 55, 105]  │
                    │  ⋮                       │
                    │ Ch7: [12, 27, 52, 102]  │
                    └──────────┬───────────────┘
                               │ 8 arrays
                               ▼
Input Stream                                              Output Stream
┌─────────────┐         ┌─────────────┐                 ┌─────────────┐
│ Ch0: 42     │         │ Comparator  │                 │ Ch0: 2      │
│ Ch1: 73     │  ────▶  │ Array (×8)  │  ────▶         │ Ch1: 3      │
│ Ch2: 18     │         │             │                 │ Ch2: 1      │
│ Ch3: 91     │         │  Parallel   │                 │ Ch3: 3      │
│ Ch4: 5      │         │  Threshold  │                 │ Ch4: 0      │
│ Ch5: 56     │         │  Checking   │                 │ Ch5: 3      │
│ Ch6: 33     │         │             │                 │ Ch6: 2      │
│ Ch7: 45     │         │ PE units    │                 │ Ch7: 2      │
└─────────────┘         └─────────────┘                 └─────────────┘
8×input_WIDTH           8 parallel units                8×output_WIDTH

Processing: Each PE unit independently compares its input against
           its channel's threshold array to find the interval
```

## AXI Interface Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AXI INTERFACE MAPPING                            │
└─────────────────────────────────────────────────────────────────────────┘

AXI-Stream Input (input_*)
┌────────────────────────────────────────┐
│ TDATA: ((PE×input_WIDTH+7)/8)×8 bits   │
│ ┌────┬────┬────┬────┬────┬────┬───┐   │
│ │ PE0│ PE1│ PE2│ PE3│ PE4│ PE5│...│   │
│ └────┴────┴────┴────┴────┴────┴───┘   │
│ TVALID: Data valid signal              │
│ TREADY: Backpressure signal            │
└────────────────────────────────────────┘

AXI-Lite Configuration (threshold_*)
┌────────────────────────────────────────┐
│ Write Channel:                         │
│ - AWADDR: Threshold memory address     │
│ - WDATA: 32-bit threshold value        │
│ - WSTRB: Byte write enables            │
│                                        │
│ Read Channel:                          │
│ - ARADDR: Threshold memory address     │
│ - RDATA: 32-bit threshold value        │
│                                        │
│ Address Mapping:                       │
│ Addr = CH×LEVELS×4 + LEVEL×4          │
└────────────────────────────────────────┘

AXI-Stream Output (output_*)
┌────────────────────────────────────────┐
│ TDATA: ((PE×O_BITS+7)/8)×8 bits       │
│ ┌───┬───┬───┬───┬───┬───┬───┬───┐    │
│ │Out│Out│Out│Out│Out│Out│Out│Out│    │
│ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │    │
│ └───┴───┴───┴───┴───┴───┴───┴───┘    │
│ TVALID/TREADY: Flow control            │
└────────────────────────────────────────┘

O_BITS calculation includes BIAS offset handling
```

## Performance Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE CHARACTERISTICS                          │
└─────────────────────────────────────────────────────────────────────────┘

Given Configuration:
- CHANNELS = 256
- PE = 8
- input_WIDTH = 8 bits
- output_WIDTH = 2 bits (4 classes)
- Clock = 100 MHz

Performance Metrics:
┌─────────────────────────────────────────────────────────────────────────┐
│ Metric                    │ Calculation              │ Value            │
├───────────────────────────┼──────────────────────────┼──────────────────┤
│ Channel Fold (CF)         │ CHANNELS / PE            │ 32               │
│ Initiation Interval       │ CF cycles                │ 32 cycles        │
│ Input Bandwidth           │ PE × input_WIDTH         │ 64 bits/cycle    │
│ Output Bandwidth          │ PE × output_WIDTH        │ 16 bits/cycle    │
│ Throughput @ 100MHz       │ 100M / 32                │ 3.125M inf/sec   │
│ Input Data Rate           │ 64 × 100                 │ 6.4 Gbps         │
│ Output Data Rate          │ 16 × 100                 │ 1.6 Gbps         │
└───────────────────────────┴──────────────────────────┴──────────────────┘

Resource Usage Factors:
- Threshold Storage: CHANNELS × LEVELS × T_WIDTH bits
- Comparator Logic: PE × LEVELS comparators
- Memory Type: Controlled by DEPTH_TRIGGER_URAM/BRAM parameters
```

## Tiling Strategy Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TILING STRATEGY FOR THRESHOLDING                     │
└─────────────────────────────────────────────────────────────────────────┘

Typical Configuration (Element-wise operation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tensor Shape: [1, 256]  (Batch=1, Channels=256)
                ↓
Block Tiling: [1, "CHANNELS"]
Result: [1, 256]  (Process entire tensor as one block)
                ↓
Stream Tiling: [1, "PE"]
Result: [1, 8]  (Stream 8 channels per cycle)

Visual Representation:
                    Tensor/Block
        ┌─────────────────────────────────┐
        │  All 256 Channels              │
        │ ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─ │
        │ │0│1│2│3│4│5│6│7│8│9│A│B│C│... │
        │ └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─ │
        └─────────────────────────────────┘
                    ↓ Stream
        ┌─────────┐ ┌─────────┐ ┌─────────┐
Cycle 0:│ Ch 0-7  │ │ Ch 8-15 │ │Ch 16-23 │ ...
        └─────────┘ └─────────┘ └─────────┘
          PE=8        PE=8        PE=8

Alternative: Multi-Sample Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tensor Shape: [32, 256]  (Batch=32, Channels=256)
                ↓
Block Tiling: ["BATCH_TILE", "CHANNELS"]
Result: [8, 256]  (Process 8 samples at a time)
                ↓
Stream Tiling: [1, "PE"]
Result: [1, 8]  (Still stream 8 channels per cycle)

This would process 8 samples × 256 channels per block
requiring 32 cycles per block (256/8)
```

## Kernel Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KERNEL INTEGRATION FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: RTL with Pragmas (thresholding_axi_bw.sv)
┌─────────────────────────────────────────┐
│ // @brainsmith DATATYPE input * 1 32   │
│ // @brainsmith BDIM input input_BDIM   │
│ // @brainsmith SDIM input input_SDIM   │
│ module thresholding_axi #(              │
│   int unsigned input_BDIM = 1,         │
│   int unsigned input_SDIM = 1,         │
│   ...                                   │
└────────────────┬────────────────────────┘
                 │ RTL Parser
                 ▼
Step 2: Extracted Metadata
┌─────────────────────────────────────────┐
│ {                                       │
│   "interfaces": {                       │
│     "input": {                          │
│       "bdim": "input_BDIM",            │
│       "sdim": "input_SDIM",            │
│       "shape": ["CHANNELS", "PE"]      │
│     }                                   │
│   }                                     │
│ }                                       │
└────────────────┬────────────────────────┘
                 │ Template Engine
                 ▼
Step 3: Generated FINN HWCustomOp
┌─────────────────────────────────────────┐
│ class ThresholdingAXI(AutoHWCustomOp):  │
│   def __init__(self):                   │
│     self.kernel_def = self._create_def()│
│                                         │
│   def get_nodeattr_types(self):         │
│     return {                            │
│       "CHANNELS": ("i", False, 1),     │
│       "PE": ("i", False, 1),           │
│       "input_WIDTH": ("i", False, 8),  │
│       ...                              │
│     }                                   │
└─────────────────────────────────────────┘
```

## Summary

The thresholding kernel is a perfect example of an element-wise operation in the Kernel Modeling system:

1. **Simple Data Flow**: One input stream → threshold comparison → one output stream
2. **Channel Parallelism**: PE parameter controls how many channels process in parallel
3. **No Spatial Tiling**: Unlike convolution, thresholding typically processes entire spatial dimensions
4. **Runtime Configuration**: AXI-Lite interface allows updating thresholds without recompilation
5. **Flexible Datatypes**: Both input and output widths are parameterizable (1-32 bits)

The kernel demonstrates clean separation between:
- **Definition** (what interfaces and parameters exist)
- **Model** (concrete dimensions and types)
- **Implementation** (actual RTL logic)

This makes it easy to explore different parallelism levels (PE values) and bit widths without modifying the core algorithm.