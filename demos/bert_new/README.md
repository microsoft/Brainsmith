# 🚀 BERT Accelerator Demo - Powered by brainsmith.forge()

**See how one function call creates FPGA accelerators from BERT models!**

## 30-Second Quick Start

```bash
# Generate a BERT accelerator - that's it!
python end2end_bert.py --output-dir ./my_bert_accelerator
```

**What just happened?** You created a complete FPGA accelerator for BERT inference using the `bert_minimal` blueprint. No complex configuration, no hardware expertise required.

## 🎯 What This Demo Shows

- **The Power of One Function**: `brainsmith.forge()` handles everything
- **Blueprint Magic**: YAML blueprints make optimization automatic
- **Model → Accelerator**: Complete end-to-end transformation
- **FPGA Made Simple**: Hardware acceleration without the complexity

## ✨ Customization Options

```bash
# Customize your BERT model
python end2end_bert.py \
    --output-dir ./custom_bert \
    --hidden-size 512 \
    --num-layers 6 \
    --sequence-length 256

# Target different FPGA boards
python end2end_bert.py \
    --output-dir ./versal_bert \
    --board "Versal_VCK190"
```

## 🎉 What You Get

After running the demo, your output directory contains:
- `bert_model.onnx` - Your generated BERT model
- `accelerator.zip` - Complete FPGA accelerator core
- `bert_accelerator_info.json` - Build information
- Performance metrics and resource utilization

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

## 🎯 The Big Picture

This demo proves that FPGA acceleration doesn't have to be complex:

**Before**: Weeks of manual optimization, complex toolchains, hardware expertise required

**After**: One function call, automatic optimization, accelerator ready to deploy

---

**Ready to see the magic? Run the 30-second demo above! 🚀**