## BrainSmith FINN Plugin repo

This repo contains a plugin for the FINN dataflow compiler as part of the Microsoft/AMD BrainSmith project.
This repo is a collection of operators and transformations that FINN can pick up and load into the FINN docker.

### Quick start

1. To use the repo requires a specific FINN branch. Please clone the following:
```bash
git clone https://github.com/Xilinx/finn.git -b custom/transformer
```

2. Within this branch, you should see a `python_repos.txt` file; these are Python repositories that are pulled in and installed during the build-up of the docker container.
We need to add _this_ repo to this file to install it as a plugin. Add the following to the bottom of the `python_repos.txt` file:
```
dir,url,commit_hash
qonnx,https://github.com/fastmachinelearning/qonnx.git,c1c12d2549c5de4478371d9999db991691007c10
finn-experimental,https://github.com/Xilinx/finn-experimental.git,0724be21111a21f0d81a072fccc1c446e053f851
brevitas,https://github.com/Xilinx/brevitas.git,d4834bd2a0fad3c1fbc0ff7e1346d5dcb3797ea4
pyverilator,https://github.com/maltanar/pyverilator.git,ce0a08c20cb8c1d1e84181d6f392390f846adbd1
finnbrainsmith,git@github.com:Xilinx-Projects/finn_brainwave.git,plugin
```
Feel free to adjust this if you work off a different feature fork/branch.

3. Launch the docker container:
```
./run-docker.sh
```

4. Within the docker container, navigate to the plugin directory:
```
cd deps/finnbrainsmith
```

5. You can then try and push the latest ONNX version of the tiny BERT model through the design
```
cd bert_build
python build.py -i bert-tiny-1layer_relu_scale_1_fp16_quant_qonnx.onnx
```

Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc.. from the files in the `dips/finnbrainsmith` directory and run the changes in the docker container.
