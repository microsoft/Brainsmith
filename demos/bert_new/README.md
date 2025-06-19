# 🚀 BERT Accelerator Demo - Powered by brainsmith.forge()

**See how one function call creates FPGA accelerators from BERT models!**

## 30-Second Quick Start

### Option 1: Using Docker (Recommended)

```bash
# From the brainsmith root directory
./run-docker.sh python demos/bert_new/end2end_bert.py --output-dir ./bert_demo_output
```

### Option 2: Direct Python (if environment is set up)

```bash
# From the brainsmith root directory
python demos/bert_new/end2end_bert.py --output-dir ./bert_demo_output
```

**What just happened?** You created a complete FPGA accelerator for BERT inference using the `bert_minimal` blueprint. No complex configuration, no hardware expertise required.

## 🎯 What This Demo Shows

- **The Power of One Function**: `brainsmith.forge()` handles everything
- **Blueprint Magic**: YAML blueprints make optimization automatic
- **Model → Accelerator**: Complete end-to-end transformation
- **FPGA Made Simple**: Hardware acceleration without the complexity

## ✨ Customization Options

```bash
# Custom BERT size
./run-docker.sh python demos/bert_new/end2end_bert.py \
    --output-dir ./large_bert \
    --hidden-size 512 \
    --num-layers 6

# Different FPGA board
./run-docker.sh python demos/bert_new/end2end_bert.py \
    --output-dir ./versal_bert \
    --board "Versal_VCK190"

# Longer sequences
./run-docker.sh python demos/bert_new/end2end_bert.py \
    --output-dir ./long_seq_bert \
    --sequence-length 512
```

## 🎉 What You Get

After successful completion, your output directory contains:
- `bert_model.onnx` - Your generated BERT model
- `accelerator.zip` - Complete FPGA accelerator core
- `bert_accelerator_info.json` - Build information
- Performance metrics and resource utilization

## Expected Output

You should see:
```
🚀 BERT Accelerator Demo - Powered by brainsmith.forge()
📦 Generating BERT model: 3 layers, 384D
✨ Watch one function call create an FPGA accelerator!
BERT model generated: ./bert_demo_output/bert_model.onnx
📋 Using blueprint: [blueprint_path]
🎯 Target board: V80
🚀 Generating BERT accelerator with brainsmith.forge()...
📦 Processing results...
🎉 SUCCESS! BERT accelerator generated!
📁 Your accelerator is ready in: ./bert_demo_output
⚡ Throughput: [X] operations/second
🏗️  Resource usage: [Y]% LUTs
🚀 That's it! One function call created your FPGA accelerator.
🎯 Model: BERT 3 layers, 384 hidden size
💡 Ready to deploy on V80
```

## 🔧 Under the Hood

The demo uses modern BrainSmith architecture:
- **`brainsmith.forge()`** - Single function for complete workflow
- **Blueprint system** - YAML-driven optimization
- **Automatic parameter selection** - No manual tuning needed
- **Clean error handling** - Clear success/failure feedback

## 📁 Files

- `end2end_bert.py` - Main demo (showcases forge() simplicity)
- `gen_initial_folding.py` - Legacy reference (preserved for experts)
- `Makefile` - Advanced build recipes (for power users)

## 🔧 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'brainsmith'"**
A: Use the Docker option - it has all dependencies pre-installed.

**Q: "Blueprint not found"**
A: Ensure you're running from the brainsmith root directory.

**Q: Demo takes a long time**
A: Normal! FPGA compilation can take 10-30 minutes depending on model size.

## 🎯 The Big Picture

This demo proves that FPGA acceleration doesn't have to be complex:

**Before**: Weeks of manual optimization, complex toolchains, hardware expertise required

**After**: One function call, automatic optimization, accelerator ready to deploy

## Next Steps

- Try different BERT configurations
- Experiment with other FPGA boards
- Deploy your accelerator to actual hardware
- Explore the generated files to understand the output

---

**Ready to see the magic? Run the 30-second demo above! 🚀**