# FINN Flows for simplified bert model

This folder contains Python scripts to split the simplified bert model into three parts and build flows to generate IPs for two of the partitions using the [FINN Compiler](https://github.com/Xilinx/finn).

## Setup

1. Ensure you have a working [Docker installation](https://docs.docker.com/engine/install/)
2. Run the `get-finn.sh` script in this folder to get a copy of the FINN compiler at the correct commit hash for rebuilding this example. You may have to re-run this script in the future when this repo gets updated, and more updates/fixes are made to the FINN compiler.
3. Set up the environment variables, see https://finn-dev.readthedocs.io/en/latest/getting_started.html for more. Some example values:
```
# parallelize jobs across 4 workers
export NUM_DEFAULT_WORKERS=4
# specify storage location for intermediate build files, recommended to have at least 10GB free
export FINN_HOST_BUILD_DIR=/scratch/finn-build
# specify location for Vitis install
# assuming Vitis 2024.1 is installed under /opt/Xilinx/Vitis/2024.1
export FINN_XILINX_PATH="/opt/Xilinx"
export FINN_XILINX_VERSION="2024.1"
# mount the licenses (e.g. if license are in /opt/Xilinx/licenses)
export FINN_DOCKER_EXTRA=" -v /proj/xbuilds/licenses:/proj/xbuilds/licenses -e XILINXD_LICENSE_FILE=/proj/xbuilds/licenses -v ${CMAKE_SOURCE_DIR}:${CMAKE_SOURCE_DIR} "


```
4. Place the relevant ONNX model file under `${CMAKE_SOURCE_DIR}/models/`.

## Flow files and example models

All relevant files for running the FINN compiler flows are contained under the [scripts_finn](scripts_finn/) folder.

## Running the flows

The table below lists the Python scripts.

| Script name                | Description                                                                                               |
|----------------------------|-----------------------------------------------------------------------------------------------------------|
| `create_partitions`        | Tidy up and partitioning of the model                                                                     |
| `generate_ip`              | Generate FINN IP for one partition, the partition is selected by argument; e.g., --model_name partition_0 |
| `custom_steps`             | Contains custom steps for the FINN flows                                                                  |


Note that all outputs from these flows are placed under a subdirectory.
Launch the build as follows, fixing the path to the repository for your system as needed:

```
cd /path/to/repository/scripts_finn/finn
./run-docker.sh build_custom /path/to/repository/scripts_finn create_partitions
```

This will tidy up the model and create 3 partitions in the subdirectory `output_create_partitions/intermediate_models/partitions/`:

* partition_0.onnx
* partition_1.onnx
* partition_2.onnx

We can use `generate_ip.py` to generate the FINN IP for partition 0 or partition 2, e.g.:

```
cd /path/to/repository/scripts_finn/finn
./run-docker.sh build_custom /path/to/repository/scripts_finn generate_ip --model_name partition_0
```
