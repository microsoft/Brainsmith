# Analysis Libraries

This directory contains analysis, profiling, and benchmarking tools for performance evaluation and optimization guidance.

## Structure

### Profiling Tools
- **`profiling/`** - Performance profiling and benchmarking
  - `model_profiling.py` - Model-level performance analysis
  - `roofline_runner.py` - Roofline model execution
  - `roofline.py` - Roofline analysis implementation

### Analysis Tools
- **`tools/`** - General analysis and utility tools
  - `gen_kernel.py` - Kernel generation utilities
  - `hw_kernel_gen/` - Hardware kernel generation framework
    - `hkg.py` - Hardware kernel generator main interface
    - `data.py` - Data structures and parsing
    - `generators/` - Code generation modules
    - `rtl_parser/` - RTL parsing and analysis
    - `templates/` - Code generation templates

### Extension Points
- **`contrib/`** - Stakeholder-contributed analysis tools

## Usage

### Performance Analysis
```python
from brainsmith.libraries.analysis.profiling import roofline_analysis
from brainsmith.libraries.analysis.profiling import RooflineProfiler

# Run roofline analysis
analysis = roofline_analysis(model, platform_config)

# Use profiler directly
profiler = RooflineProfiler(platform="zynq_ultrascale")
results = profiler.profile(model)
```

### Hardware Kernel Generation
```python
from brainsmith.libraries.analysis.tools.hw_kernel_gen import hkg

# Generate hardware kernel from RTL
kernel_config = hkg.generate_kernel(
    rtl_source="my_kernel.sv",
    output_dir="generated_kernels/"
)
```

### Integration with DSE
```python
from brainsmith.core.api import forge

# Analysis tools are used during design space exploration
result = forge(
    model=model,
    blueprint="high_performance",
    analysis=["roofline", "resource_utilization"]
)
```

## Features
- **Roofline Analysis**: Performance bound analysis for FPGA platforms
- **Resource Modeling**: FPGA resource utilization prediction
- **Kernel Generation**: Automated hardware kernel creation from RTL
- **Performance Profiling**: Comprehensive benchmarking capabilities
- **Platform Support**: Multi-platform analysis (Zynq, Versal, etc.)

## Integration
- Seamless integration with DSE engine
- Support for multiple FPGA platforms
- Automatic result visualization and reporting
- Integration with blueprint system for analysis recipes