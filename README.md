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
qonnx,https://github.com/fastmachinelearning/qonnx.git,ca91dbe24e8d0122ba981070b918be31fb60750e
finn-experimental,https://github.com/Xilinx/finn-experimental.git,0724be21111a21f0d81a072fccc1c446e053f851
brevitas,https://github.com/Xilinx/brevitas.git,0ea7bac8f7d7b687c1ac0c8cb4712ad9885645c5
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

5. You can then try and build a BERT model in brevitas, extract the BERT encoder potion of the design, and push it through the build flow with the following script. 
```
cd bert_build
python endtoend.py -o finnbrainsmith_bert.onnx
```

6. You can also run a suite of tests on the finnbrainsmith repository which will check:
 
* Shuffle hardware generation and correctness
* QuantSoftMax hardware generation and correctness
* EndtoEnd flow

To run the tests
```
cd tests
pytest ./
```

Since the Python repo is installed in developer mode in the docker container, you can edit the files, push to git, etc.. from the files in the `deps/finnbrainsmith` directory and run the changes in the docker container.
