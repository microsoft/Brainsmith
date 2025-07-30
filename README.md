## Brainsmith

Brainsmith is a platform with the goal of fully automated design space exploration (DSE) and implementation of neural networks on FPGA, from PyTorch to RTL.

## Pre-Release

**This repository is in a pre-release state and under active co-development by Microsoft and AMD.**.

### Pre-release features:
- **Component library** - Register hardware designs and algorithms as components in the Brainsmith library for use in DSE
- **Blueprint interface** - Define the components, algorithms, and parameters for a single build
- **BERT demo** - Example end-to-end demo (PyTorch to stitched-IP RTL accelerator)

### Planned major features:
- **Automated Design Space Exploration (DSE)** - Iteratively run builds across a design space, evaluating performance to converge on the optimal design for given search objectives and constraints
- **Parallelized tree execution** - Execute multiple builds in parallel, intelligently re-using build artefacts
- **Automated Kernel Integrator** - Easy integration of new hardware kernels, generate full compiler integration python code from RTL or HLS code alone
- **FINN Kernel backend rework** - Flexible backends for FINN kernels, currently you can only select between HLS or RTL backend, in the future releases multiple RTL or HLS backends will be supported to allow for more optimization
- **Accelerated FIFO sizing** - The FIFO sizing phase of Brainsmith builds currently represent +90% of runtime (not including Vivado Synthesis + Implementation). This will be significantly accelerated in future releases.

## Quick Start

### Dependencies
1. Ubuntu 22.04+
2. Vivado Design Suite 2024.2 (migration to 2025.1 in process)
3. Docker with [non-root permissions](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)


### 1. Set key environment variables

```bash
# Brainsmith env vars with example paths
export BSMITH_ROOT=/home/user/brainsmith/
export BSMITH_BUILD_DIR=/home/user/builds/brainsmith
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

### 2. Run end-to-end test to validate environment 

```bash
# Start environment container
./smithy daemon

# Attach shell to container 
./smithy shell
# Run example
./demos/bert/scripts/quicktest.sh

# OR execute one-off command 
./smithy exec ./demos/bert/scripts/quicktest.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Brainsmith is developed through a collaboration between Microsoft and AMD.

The project builds upon:
- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks on FPGAs
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX model representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library
