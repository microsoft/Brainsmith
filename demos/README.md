# Kernel Integrator & Kernel Modeling System Demos

This directory contains interactive demonstrations showcasing the main innovations of the Kernel Integrator and Kernel Modeling System.

## 🎯 Overview

These demos visually demonstrate how the Kernel Integrator revolutionizes FPGA accelerator development by:
- Converting SystemVerilog RTL to FINN HWCustomOp in seconds
- Providing mathematically rigorous hardware abstractions
- Eliminating 95% of boilerplate code
- Reducing integration time from weeks to minutes

## 📋 Available Demos

### Demo 1: RTL to FINN in 30 Seconds
**File:** `demo_01_rtl_to_finn.py`

Shows the complete conversion pipeline from SystemVerilog RTL to FINN-compatible HWCustomOp in real-time.

```bash
python demos/demo_01_rtl_to_finn.py
```

**Features:**
- Live progress tracking
- Side-by-side RTL and generated code
- Performance metrics comparison
- Time savings visualization

### Demo 2: RTL Parser Interactive Explorer  
**File:** `demo_02_rtl_parser.py`

Interactive exploration of SystemVerilog parsing with rich visualizations.

```bash
python demos/demo_02_rtl_parser.py
```

**Features:**
- File browser with 13 example RTL files
- Pragma highlighting and statistics
- Multi-format export (JSON, Markdown)
- Complexity analysis dashboard

### Future Demos (Coming Soon)

The following demos are planned for future development:

- **Progressive Complexity** - Evolution from simple to complex RTL modules
- **AutoHWCustomOp Magic** - Before/after comparison of boilerplate elimination  
- **Template Generation** - CodegenBinding and parameter flow visualization
- **End-to-End Testing** - Complete test pipeline visualization
- **Performance Metrics** - Impact visualization with charts
- **Pragma System Power** - Advanced pragma features
- **Real Hardware Integration** - FPGA synthesis results

## 🚀 Running the Demos

### Prerequisites

Install required Python packages:
```bash
pip install rich questionary matplotlib networkx
```

### Running the Demos

All demos must be run inside the Brainsmith Docker container:

```bash
# Start the container and get a shell
./smithy shell

# Then run demos directly
python demos/demo_01_rtl_to_finn.py
python demos/demo_02_rtl_parser.py
python demos/run_all_demos.py
```

Or run from outside the container:
```bash
./smithy exec "python demos/demo_01_rtl_to_finn.py"
./smithy exec "python demos/run_all_demos.py"
```

### Running Individual Demos

Each demo can be run independently:

```bash
# Interactive mode (default)
python demos/demo_01_rtl_to_finn.py

# Non-interactive mode
python demos/demo_01_rtl_to_finn.py --non-interactive

# With custom RTL file
python demos/demo_01_rtl_to_finn.py --rtl-file path/to/file.sv
```

### Running All Demos

Use the master runner for presentations:

```bash
# Interactive menu
python demos/run_all_demos.py

# Presentation mode (auto-advance)
python demos/run_all_demos.py --presentation

# Generate report
python demos/run_all_demos.py --report
```

## 📁 Output Files

Demo outputs are saved to `demo_outputs/` with subdirectories for each demo:

```
demo_outputs/
├── rtl_to_finn/         # Generated FINN files
├── rtl_parser/          # Exported metadata
├── kernel_modeling/     # Visualizations
└── *.json              # Demo results
```

## 🎨 Visualization Examples

The demos generate various visualizations:

1. **Data Hierarchy Diagrams** - Shows Tensor → Block → Stream → Element flow
2. **Interface Diagrams** - Visualizes kernel interfaces and connections
3. **Performance Charts** - Compares manual vs automated metrics
4. **Pragma Visualizations** - Highlights RTL annotations

## 💡 Tips for Presenters

1. **Start with Demo 1** for immediate impact (30-second conversion)
2. **Show Demo 2** for technical deep-dive (RTL Parser)
3. **Keep each demo to 2-3 minutes** for engagement
4. **Have terminal and visualizations ready** before starting

## 🔧 Customization

### Adding New Examples

1. Add RTL files to `tests/tools/kernel_integrator/rtl_parser/demo_rtl/`
2. Update examples in demo scripts
3. Add new kernel configurations to Demo 4

### Creating New Demos

1. Copy a demo template from existing files
2. Import common utilities from `demos.common`
3. Follow the established structure
4. Update this README and `run_all_demos.py`

## 📚 Further Documentation

- [Kernel Integrator Architecture](../docs/KI_DESIGN_DOCUMENT.md)
- [RTL Parser Guide](../brainsmith/tools/kernel_integrator/rtl_parser/README.md)
- [Kernel Modeling System](../brainsmith/core/dataflow/README.md)
- [FINN Integration](../brainsmith/core/finn/README.md)

## 🐛 Troubleshooting

### Import Errors
- Ensure you're running inside the Docker container
- Check that dependencies are installed

### Visualization Issues
- Install matplotlib: `pip install matplotlib`
- For headless systems, use `Agg` backend

### File Not Found
- Run from the repository root
- Check file paths in demo scripts

### Permission Errors
- Ensure `demo_outputs/` directory is writable
- Run with appropriate permissions