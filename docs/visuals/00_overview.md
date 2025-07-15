# Brainsmith Visual Documentation - Overview

This directory contains visual diagrams explaining the Brainsmith platform architecture and key features.

## Document Index

1. **[01_system_overview.md](01_system_overview.md)** - High-level system architecture
2. **[02_three_phase_pipeline.md](02_three_phase_pipeline.md)** - Detailed three-phase DSE pipeline
3. **[03_plugin_system.md](03_plugin_system.md)** - Plugin registry and architecture
4. **[04_design_space_exploration.md](04_design_space_exploration.md)** - DSE flow state machine
5. **[05_repository_structure.md](05_repository_structure.md)** - Repository organization
6. **[06_backend_system.md](06_backend_system.md)** - Backend factory and implementations
7. **[07_blueprint_configuration.md](07_blueprint_configuration.md)** - Blueprint YAML structure
8. **[08_pareto_optimization.md](08_pareto_optimization.md)** - Multi-objective optimization
9. **[09_data_flow.md](09_data_flow.md)** - Data flow through the system
10. **[10_plugin_discovery.md](10_plugin_discovery.md)** - Plugin registration sequence

## About Brainsmith

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It converts PyTorch models to RTL implementations for FPGA deployment using a sophisticated Blueprint-based design space exploration (DSE) approach.

### Core Pipeline
```
PyTorch Model → Brevitas Quantization → ONNX → FINN/Brainsmith → RTL Synthesis
Blueprint YAML → DSE v3 → Hardware Implementation → FPGA Deployment
```

### Key Innovation
Brainsmith introduces a **Blueprint-driven DSE system** that systematically explores hardware design spaces to find optimal FPGA configurations. Unlike traditional single-point solutions, it explores multiple kernel backends, transform combinations, and build configurations to maximize performance while meeting resource constraints.