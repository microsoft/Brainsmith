# 🚀 BERT Demo Quick Start Guide

## How to Run the Demo

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

## Customization Examples

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

## What You Get

After successful completion, `./bert_demo_output/` contains:
- `bert_model.onnx` - Your BERT model
- `accelerator.zip` - Complete FPGA accelerator
- `bert_accelerator_info.json` - Build metadata

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'brainsmith'"**  
A: Use the Docker option - it has all dependencies pre-installed.

**Q: "Blueprint not found"**  
A: Ensure you're running from the brainsmith root directory.

**Q: Demo takes a long time**  
A: Normal! FPGA compilation can take 10-30 minutes depending on model size.

## Next Steps

- Try different BERT configurations
- Experiment with other FPGA boards
- Deploy your accelerator to actual hardware
- Explore the generated files to understand the output