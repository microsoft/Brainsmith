# Brainsmith System Overview

## High-Level Architecture (ASCII Art)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               BRAINSMITH PLATFORM                                    │
│                    Open-Source FPGA AI Accelerator Framework                         │
│                         Microsoft & AMD Collaboration                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  PyTorch Model  │ ───► │    Brevitas      │ ───► │   ONNX Model     │
│                 │      │  Quantization    │      │                  │
└─────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                  BRAINSMITH DSE v3                                     │
│ ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐         │
│ │      PHASE 1        │   │      PHASE 2        │   │      PHASE 3        │         │
│ │  Design Space       │──►│  Design Space       │──►│   Build Runner      │         │
│ │   Constructor       │   │     Explorer        │   │                     │         │
│ │                     │   │                     │   │                     │         │
│ │ • Parse Blueprint   │   │ • Generate Configs  │   │ • Select Backend    │         │
│ │ • Validate Schema   │   │ • Rank by Pareto   │   │ • Execute Build     │         │
│ │ • Expand Variants   │   │ • Execute Builds    │   │ • Collect Metrics   │         │
│ └─────────────────────┘   └─────────────────────┘   └─────────────────────┘         │
│              │                      │                          │                       │
│              ▼                      ▼                          ▼                       │
│       Design Space           Build Configs              Build Results                  │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────┐
                          │   FPGA Implementation    │
                          │   • RTL Generation       │
                          │   • Bitstream Creation   │
                          │   • Performance Metrics  │
                          └──────────────────────────┘
```

## Key Components

### Input Layer
- **PyTorch Model**: Original neural network model
- **Brevitas Quantization**: Reduces precision for hardware efficiency
- **ONNX Model**: Standardized model representation

### DSE v3 Core
- **Phase 1 - Design Space Constructor**: Parses blueprints and creates valid design space
- **Phase 2 - Design Space Explorer**: Explores configurations and ranks by Pareto optimality
- **Phase 3 - Build Runner**: Executes hardware builds and collects metrics

### Output Layer
- **RTL Generation**: Hardware description in Verilog/VHDL
- **Bitstream Creation**: FPGA programming file
- **Performance Metrics**: Throughput, latency, resource utilization

## Design Philosophy

Brainsmith's three-phase approach separates concerns:
1. **Construction** - What configurations are possible?
2. **Exploration** - Which configurations are optimal?
3. **Execution** - How to build the selected configurations?

This separation enables:
- Modular development and testing
- Parallel exploration of design space
- Easy integration of new backends and optimizations