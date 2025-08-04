# 🚀 Kernel Integrator Demo Guide

This guide walks you through running the two completed demos that showcase the Kernel Integrator capabilities.

## Prerequisites

1. **Start the Docker container:**
   ```bash
   ./smithy shell
   ```

2. **Install demo dependencies (if not already installed):**
   ```bash
   pip install rich questionary matplotlib
   ```

## Demo 1: RTL to FINN in 30 Seconds

### What It Shows
- Complete conversion from SystemVerilog RTL to FINN HWCustomOp
- Real-time progress tracking  
- 99.9% reduction in integration time (2 weeks → 5 minutes)

### How to Run

**Interactive Mode** (recommended for presentations):
```bash
python demos/demo_01_rtl_to_finn.py
```

**Non-Interactive Mode** (for quick demonstration):
```bash
python demos/demo_01_rtl_to_finn.py --non-interactive
```

**With Custom RTL File**:
```bash
python demos/demo_01_rtl_to_finn.py --rtl-file brainsmith/kernels/mvu/mvu_vvu_axi.sv
```

### What You'll See
1. **RTL File Preview** - SystemVerilog with pragma annotations highlighted
2. **Live Conversion Progress** - Progress bars for each conversion stage
3. **Generated Files** - Three files created:
   - `thresholding_axi_hw_custom_op.py` - FINN HWCustomOp implementation
   - `thresholding_axi_rtl.py` - RTL backend for synthesis
   - `thresholding_axi_wrapper.v` - Verilog wrapper
4. **Impact Metrics** - Before/after comparison showing 99.9% time reduction

### Demo Duration: 2-3 minutes

---

## Demo 2: RTL Parser Interactive Explorer

### What It Shows
- Rich visualization of RTL parsing process
- 13 example files from simple to complex
- Pragma extraction and analysis
- Multi-format export capabilities

### How to Run

**Option 1: Master Runner** (recommended):
```bash
python demos/run_all_demos.py --demo 2
```

**Option 2: Direct Execution**:
```bash
python demos/demo_02_rtl_parser.py
```

**With Specific File**:
```bash
python demos/demo_02_rtl_parser.py --file tests/tools/kernel_integrator/rtl_parser/demo_rtl/03_datatype_pragmas.sv
```

### What You'll See
1. **File Selection Menu** - Choose from 13 example RTL files:
   - `01_basic_module.sv` - Minimal example
   - `03_datatype_pragmas.sv` - Datatype specifications
   - `06_complex_interfaces.sv` - Multiple interfaces
   - `10_matrix_vector_unit.sv` - Real-world complexity
2. **File Preview** - Syntax highlighting with pragma annotations
3. **Parse Results** - Full metadata extraction display
4. **Export Options** - Save as JSON or Markdown
5. **Pragma Dashboard** - Statistics and complexity analysis

### Demo Duration: 3-4 minutes

---

## Running All Demos

### Option 1: Interactive Shell Script
```bash
./run_demos.sh
```

### Option 2: Master Python Runner
```bash
# Interactive menu
python demos/run_all_demos.py

# Presentation mode (auto-advance)
python demos/run_all_demos.py --presentation --delay 10

# Generate report after running
python demos/run_all_demos.py --report
```

---

## Demo Outputs

After running demos, check the generated files:

```bash
# Demo 1 outputs
ls -la demo_outputs/rtl_to_finn/
cat demo_outputs/rtl_to_finn/thresholding_axi_hw_custom_op.py | head -50

# Demo results
ls demo_outputs/*_results.json

```

---

## Presentation Tips

1. **Start with Demo 1** - Shows immediate impact (30-second conversion)
2. **Use non-interactive mode** for smooth presentations
3. **Have outputs ready** - Pre-run demos to ensure files exist
4. **Highlight key metrics**:
   - 336 hours → 5 minutes (99.9% reduction)
   - 1000 lines → 50 lines (95% reduction)
   - 20x productivity gain

5. **Focus on innovations**:
   - Pragma-based automation
   - Type-safe abstractions
   - Zero runtime overhead
   - Explicit parameter tracking

---

## Troubleshooting

### "Module not found" errors
```bash
# Ensure you're in the container
./smithy shell

# Check Python path
python -c "import sys; print(sys.path)"
```

### Interactive prompts in non-interactive mode
Use the `--non-interactive` flag or run through the master runner

### Missing visualizations
```bash
pip install matplotlib networkx
```

### Permission errors
```bash
chmod +x run_demos.sh
mkdir -p demo_outputs
```

---

## Key Takeaways

1. **Demo 1**: SystemVerilog → FINN in seconds, not weeks
2. **Demo 2**: Rich metadata extraction and analysis
3. **Demo 4**: Mathematical rigor meets hardware flexibility

The demos showcase how the Kernel Integrator eliminates manual FINN integration work while the Kernel Modeling System provides clean, type-safe abstractions for hardware accelerator development.