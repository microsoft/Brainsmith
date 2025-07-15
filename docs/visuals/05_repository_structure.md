# Repository Structure

## Directory Organization (ASCII Art)

```
brainsmith/
├── 🔧 core/
│   ├── phase1/                     ← Design Space Constructor
│   ├── phase2/                     ← Design Space Explorer
│   ├── phase3/                     ← Build Runner
│   └── plugins/                    ← Plugin Registry System
│
├── 🧩 kernels/                     [Hardware Operators]
│   ├── layernorm/                  ← Layer normalization
│   │   ├── layernorm.py            ← Kernel HW operator
│   │   ├── layernorm_hls.py        ← HLS backend
│   │   ├── layernorm_rtl.py        ← RTL backend
│   │   └── infer_layernorm.py      ← Infer transform
│   └── .../                        ← More kernels...
│
├── 🔄 transforms/                  [Graph Transformations]
│   ├── cleanup/                    ← RemoveIdentity
│   ├── optimization/               ← Streamline, Tiling
│   ├── dataflow_opt/               ← Hardware optimizations
│   └── .../                        ← More transforms...
│
├── 📋 steps/                       [Build Pipeline Steps]
│   └── bert_steps.py               ← BERT-specific steps
│
└── 📘 blueprints/                  [Blueprint Templates]
    ├── legacy.yaml                 ← Traditional single-point
    └── modern.yaml                 ← DSE v3 multi-config
```

## Core Module Details

### brainsmith/core/
The heart of the DSE v3 system:

**phase1/** - Design Space Constructor
- `forge.py`: Main API entry point
- `parser.py`: Blueprint YAML parsing
- `validator.py`: Schema and constraint validation
- `data_structures.py`: Core data types

**phase2/** - Design Space Explorer
- `explorer.py`: Exploration orchestration
- `combination_generator.py`: Config generation
- `ranker.py`: Pareto optimization
- `hooks.py`: Extension system

**phase3/** - Build Runner
- `build_runner.py`: Abstract interface
- `factory.py`: Backend selection
- `legacy_finn_backend.py`: Current FINN integration
- `future_brainsmith_backend.py`: Plugin-based backend

**plugins/** - Plugin System
- `registry.py`: Central registration
- `decorators.py`: Plugin decorators
- `plugin_collections.py`: Organized access
- `framework_adapters.py`: QONNX/FINN integration

### Transform Organization

**cleanup/** - Model cleaning
- Remove redundant operations
- Fold constants
- Simplify graph structure

**optimization/** - Performance optimization
- Streamlining for hardware
- Tiling and parallelization
- Resource balancing

**dataflow_opt/** - Hardware-specific
- FINN-specific transforms
- Dataflow optimizations
- Memory layout transforms

### Kernel Implementations

Each kernel provides:
- ONNX operator definition
- Hardware attributes
- Backend compatibility
- Resource estimation

### Backend System

Backends are now integrated into kernel implementations:

**High-Level Synthesis (HLS)**
- C++ code generation via `*_hls.py` files
- Vivado HLS integration
- Automated optimization

**Register Transfer Level (RTL)**
- Direct Verilog generation via `*_rtl.py` files
- Hand-optimized implementations
- Maximum performance