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

2. Clone this repo with submodules (SSH cloning is currently required):
```bash
git clone --recurse-submodules git@github.com:microsoft/Brainsmith.git
```

   If you already cloned without submodules, initialize them:
```bash
git submodule update --init --recursive
```

3. **Dependencies**: This repository uses Git submodules for major dependencies:
   - **FINN** (at repository root): Submodule pointing to the `custom/transformer` branch
   - **Other dependencies**: Specified in `docker/fetch-repos.sh` for additional components pulled during docker build
   
   To update FINN to a newer commit:
```bash
cd finn
git fetch origin
git checkout origin/custom/transformer  # or specific commit hash
cd ..
git add finn
git commit -m "Update FINN submodule"
```

4. Launch the docker container. Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc. and run the changes in docker without rebuilding the container.

```bash
# Start persistent container (one-time setup)
./smithy daemon

# Get instant shell access anytime
./smithy shell

# Or execute commands quickly
./smithy exec "python script.py"

# Check status
./smithy status

# Stop when done
./smithy stop
```

> **Note for existing users**: If you previously used `./run-docker.sh`, it now automatically redirects to `smithy` for compatibility. The new `smithy` tool provides 73% faster container operations with persistent containers. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details.

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
