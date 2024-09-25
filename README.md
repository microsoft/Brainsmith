# FINN-Brainwave

## Setup

1. Ensure you have a working [Docker installation](https://docs.docker.com/engine/install/)
2. Tools: `CMake` (3.16), `Vivado` (2024.1)
3. Get FINN:
```
cd scripts_finn
bash get-finn.sh
cd ../
```
3. Set up the environment variables, see https://finn-dev.readthedocs.io/en/latest/getting_started.html for more. Some example values:
```
# parallelize jobs across 4 workers
export NUM_DEFAULT_WORKERS=4
# specify storage location for intermediate build files, recommended to have at least 10GB free
export FINN_HOST_BUILD_DIR=<repo_path>/scripts_finn/finn/finn_temp_files
# specify location for Vitis install
# assuming Vitis 2024.1 is installed under /opt/Xilinx/Vitis/2024.1
export FINN_XILINX_PATH="/opt/Xilinx"
export FINN_XILINX_VERSION="2024.1"
# mount the licenses (e.g. if license are in /opt/Xilinx/licenses)
export FINN_DOCKER_EXTRA=" -v /opt/Xilinx/licenses:/opt/Xilinx/licenses -e XILINXD_LICENSE_FILE=/opt/Xilinx/licenses -v <repo_path>:<repo_path> "
```
4. Place the relevant ONNX model files under `models/`.

## Build

Configuration of the design is handled within .json files that are under the `config` directory (all partitions).

Once the desired configuration is set, stitched project can be built.
Importantly, this project will contain the instrumentation shell for the v80 as well, so full bitstreams can be generated straight away.

This can be done by running the following set of commands:

```
mkdir build && cd build
cmake ../
make project
```

To speed up builds you can import partitions and IP cores from previous builds. This can be done simply by providing the path to the existing build.
Instead of the previous `cmake ../` command, run the following:

```
cmake ../ -DGEN_FINN_PATH=<path_to_existing_build>
```

Once the project is built, you can open it in `Vivado` for further modifications. The project files are contained in `<path to repo>/build/finn_bwave/`. To open the project in Vivado you can run the following command:
```
vivado <path to repo>/build/finn_bwave/finn_v80.xpr
```

Bitstream can be loaded through Vivado JTAG programmer (GUI or script).

Additionally, compilation reports and design checkpoints will be generated under `reports` and `checkpoints` respectively.

## Simulation

Within the default project, simulation of the whole design (all partitions) can be done through regular Vivado simulation flow (GUI).   

In the Vivado GUI the behavioural simulation can then be run by navigating, on the left command panel, to:
```
Run Simulation -> Run Behavioral Simulation 
```

This should then launch a running simulation where signals can be added to a waveform and inspected. 

### Simulating a portion of the design

In many cases, it is beneficial to simulate only portions of the design (even individual partitions can take long time to simulate!). 

To do this, steps remain similar:

```
mkdir build && cd build
cmake ../ -DGEN_FINN_PATH=<path_to_existing_build> -DTB_NAME=<desired_core_to_simulate> -DSIM_RUN=<run_cmd_line_sim>
make project
```

The added arguments here are the `TB_NAME` and `SIM_RUN`. The first represents the name of the directory where simulation particular simulation files are placed under `tb` directory.
The second runs the simulation during the build in the command line.


Some examples of commands for simulating portions of the design are given below:

#### Simulate partition\_0
```
mkdir build && cd build
cmake ../ -DTB_NAME=tb_partition_0
make project
vivado <path to repo>/build/sim/bwave_tb.xpr
```

#### Simulate partition\_1
```
mkdir build && cd build
cmake ../ -DTB_NAME=tb_partition_1
make project
vivado <path to repo>/build/sim/bwave_tb.xpr
```

#### Simulate partition\_2
```
mkdir build && cd build
cmake ../ -DTB_NAME=tb_partition_2
make project
vivado <path to repo>/build/sim/bwave_tb.xpr
```


