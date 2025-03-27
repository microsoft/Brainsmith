## Brainsmith FINN Plugin repo

This repo contains a plugin for the FINN dataflow compiler as part of the Microsoft/AMD BrainSmith project.
This repo is a collection of operators and transformations that FINN can pick up and load into the FINN docker.

### Quick start

1. Set environment variables (seperate from FINN variables), example below:
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

4. Launch the docker container:
```
./run-docker.sh
```

5. Navigate to the job for BERT models. Generate a pre-made configuration file for rapid
testing:
```
cd brainsmith/jobs/bert
python scripts/gen_initial_folding.py --simd 12 --pe 8 --num_layers 1 -t 1 -o ./configs/l1_simd12_pe8.json
```
6. You can then try and build a BERT model in brevitas, extract the BERT encoder potion of the design, and push it through the build flow with the following script. 
```
python endtoend.py -o l1_simd12_pe8 -n 12 -l 1 -z 384 -i 1536 -x True -p ./configs/l1_simd12_pe8.json
```

7. Alternatively, you can also run a suite of tests on the finnbrainsmith repository which will check:
 
* Shuffle hardware generation and correctness
* QuantSoftMax hardware generation and correctness
* EndtoEnd flow

```
cd tests
pytest ./
```

Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc. and run the changes in docker without rebuilding the container 
