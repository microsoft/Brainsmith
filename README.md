## Brainsmith

Brainsmith is an open-source platform for FPGA AI accelerators.
This repository is in a pre-release state and under active co-devlopment by Microsoft and AMD.

### Quick start

1. Set environment variables (separate from FINN variables), example below:
```
export BSMITH_ROOT="~/brainsmith"
export BSMITH_HOST_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

2. Clone this repo (SSH cloning is currently required):
```bash
git clone git@github.com:microsoft/Brainsmith.git
```

3. (Optional) Dependencies are specified in `docker/hw_compilers/finn/fetch-repos.sh` which lists specific hashes/branches to pull during docker build. Feel free to adjust these if you work off a different feature fork/branch of key dependencies like FINN or QONNX.

4. Launch the docker container. Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc. and run the changes in docker without rebuilding the container.
```
./run-docker.sh
```

5. Validate with a 1 layer end-to-end build (generates DCP image, multi-hour build):
```
cd tests/end2end/bert
make single_layer
```

6. Alternatively, run a simplified test skipping DCP gen:
```
cd brainsmith/jobs/bert
python scripts/gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json
python endtoend.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False
```

7. Alternatively, you can also run a suite of tests on the finnbrainsmith repository which will check:
 
* Shuffle hardware generation and correctness
* QuantSoftMax hardware generation and correctness
* EndtoEnd flow

```
cd tests
pytest ./
```
