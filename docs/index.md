---
hide:
  - toc
---

# Brainsmith

## Open-source AI acceleration on FPGA: from ONNX to RTL

Brainsmith compiles ONNX models to optimized dataflow accelerator designs for FPGAs, intelligently exploring hardware configurations to find designs that maximize performance within your resource constraints.

<div class="grid" markdown style="display: grid; grid-template-columns: 1fr auto 2fr; gap: 2rem; align-items: center; margin-top: 0.5rem;">

<div markdown>

<img src="images/mha_onnx.png" alt="ONNX Graph Structure" style="width: 100%; height: 500px; object-fit: contain;">

</div>

<div style="font-size: 3rem; color: #666; text-align: center;">
→
</div>

<div markdown>

<img src="images/bert_dfc.png" alt="Generated Dataflow Accelerator" style="width: 100%; height: 500px; object-fit: contain;">

</div>

</div>


---

## Built With

Brainsmith builds upon:

- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library

Developed through collaboration between **Microsoft** and **AMD**.

**License**: MIT - see [LICENSE](https://github.com/microsoft/brainsmith/blob/main/LICENSE)
