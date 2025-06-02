## Brainsmith

Brainsmith is an open-source platform for FPGA AI accelerators.
This repository is in a pre-release state and under active co-devlopment by Microsoft and AMD.

### Quick start

1. Set environment variables (separate from FINN variables), example below:
```bash
export BSMITH_ROOT="~/brainsmith"
export BSMITH_BUILD_DIR="~/builds/brainsmith"
export BSMITH_XILINX_PATH="/tools/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export BSMITH_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
```

2. Clone this repo (SSH cloning is currently required):
```bash
git clone git@github.com:microsoft/Brainsmith.git
```

3. (Optional) Dependencies are specified in `docker/fetch-repos.sh` which lists specific hashes/branches to pull during docker build. Feel free to adjust these if you work off a different feature fork/branch of key dependencies like FINN or QONNX.

4. Launch the docker container. Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc. and run the changes in docker without rebuilding the container.

```bash
# Start persistent container (one-time setup)
./brainsmith-container start daemon

# Get instant shell access anytime
./brainsmith-container shell

# Or execute commands quickly
./brainsmith-container exec "python script.py"

# Check status
./brainsmith-container status

# Stop when done
./brainsmith-container stop
```

5. Validate with a 1 layer end-to-end build (generates DCP image, multi-hour build):
```bash
cd tests/end2end/bert
make single_layer
```

6. Alternatively, run a simplified test skipping DCP gen:
```bash
cd demos/bert
python gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json
python end2end_bert.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json -d False
```

7. Alternatively, you can also run a suite of tests on the brainsmith repository which will check:
 
* Shuffle hardware generation and correctness
* QuantSoftMax hardware generation and correctness
* EndtoEnd flow

```bash
cd tests
pytest ./
```