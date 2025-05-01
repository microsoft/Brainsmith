############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Daniel Penrose <daniel.penrose@amd.com>
############################################################################

import pytest
import onnx
import os
import numpy as np
import finn.core.onnx_exec as oxe
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
from onnx import helper, numpy_helper, OperatorSetIdProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.builder.build_dataflow_steps import step_specialize_layers
from finn.builder.build_dataflow_config import DataflowBuildConfig

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_output")


@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
class OpTest(ABC):
    """A class used to test FINN custom operators."""

    ##########################################
    #                Fixtures                #
    ##########################################

    @pytest.fixture(autouse=True)
    @abstractmethod
    def model(self) -> ModelWrapper:
        """An abstract fixture that generates the QONNX ModelWrapper to be tested (when
        implemented). Each test MUST override this fixture, otherwise any PyTests
        will result in a NotImplementedError.

        Helper functions such as create_model() and run_transforms() may be useful in
        reducing boilerplate when implementing this fixture."""

        raise NotImplementedError("This OpTest's model() fixture is unimplemented.")

    @pytest.fixture(autouse=True)
    def model_specialised(
        self,
        model: ModelWrapper,
        target_fpga: str,
        save_intermediate_models: bool,
    ) -> ModelWrapper:
        """A fixture that applys layer specialisation to the 'model' fixture, then returns it.
        The model is specialised differently depending on which execution mode is used (cppsim
        or rtlsim)."""

        specialised_model: ModelWrapper = self.apply_builder_step(
            model,
            step_specialize_layers,
            dict(fpga_part=target_fpga)
        )

        return specialised_model

    @pytest.fixture
    def target_fpga(self) -> str:
        """The fpga we're targeting for testing. Can be overridden by test classes."""

        return "xcv80-lsva4737-2MHP-e-S"

    @pytest.fixture
    def target_node(self) -> int:
        """The index of the node in the model we're focusing on. Allows for multiple nodes to be present,
        with tests that only target a specific node. Defaults to the first node. Can be overridden."""

        return 0

    @pytest.fixture
    def input_tensors(self, model: ModelWrapper) -> dict:
        """Creates the tensor(s) passed to the model, to be used by the simulation during
        testing. This fixture creates a tensor with random values, but can be overriden
        by subclasses to pass specific values."""

        input_t = {}
        for input in model.graph.input:
            input_value = gen_finn_dt_tensor(
                model.get_tensor_datatype(input.name),
                model.get_tensor_shape(input.name),
            )
            input_t[input.name] = input_value
        return input_t

    @pytest.fixture
    def save_intermediate_models(self) -> dict:
        """TODO: Comment"""

        return True

    @pytest.fixture(autouse=True)
    def auto_saver(
        self,
        request: pytest.FixtureRequest,
        save_intermediate_models: bool
    ):
        """TODO: Comment"""

        if save_intermediate_models:

            # Attempt to make the directory we'll store our intermediate models in.
            models_directory = os.path.join(OUTPUT_DIR, request.module.__name__)
            try:
                os.mkdir(models_directory)
            except FileExistsError:
                warn(f"Overwriting intermediate models in existing directory {models_directory}.")

            # Save the output of each ModelWrapper fixture that starts with "model"
            # (i.e. model, model_specialised). Assumes these fixtures return ModelWrappers.
            for fixture in filter(lambda x: x.startswith("model"), request.fixturenames):

                filename = os.path.join(models_directory, fixture) + ".onnx"
                output = request.getfixturevalue(fixture)

                if isinstance(output, ModelWrapper):
                    output.save(filename)
                else:
                    raise TypeError(f"Model fixture {fixture} has return type {type(output)}. Expected ModelWrapper.")

        return

    @pytest.fixture(autouse=True)
    def create_output_dir(self):
        """Automatically makes a test output directory if none exists. This fixture auto-runs
        before all test functions. If the directory exists, this fixture does nothing."""

        try:
            os.mkdir(OUTPUT_DIR)
        except FileExistsError:
            pass

    ##########################################
    #                  Tests                 #
    ##########################################

    # Ensure the number of cycles the layer takes to run in rtlsim
    # aligns with the expected number of cycles.
    def test_cycles(
        self, model_specialised: ModelWrapper, target_node: int, exec_mode: str
    ) -> None:

        if exec_mode == "rtlsim":
            op_type = model_specialised.graph.node[target_node].op_type
            node = model_specialised.get_nodes_by_op_type(op_type)[0]
            inst = getCustomOp(node)
            cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
            exp_cycles_dict = model_specialised.analysis(exp_cycles_per_layer)
            exp_cycles = exp_cycles_dict[node.name]
            assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
            assert exp_cycles != 0

    ##########################################
    #            Helper Functions            #
    ##########################################

    def create_model(
        self,
        inputs: List[tuple[dict[str, any], str]],  # (tensor_params, finn_dt)
        outputs: List[tuple[dict[str, any], str]],  # (tensor_params, finn_dt)
        inits: List[dict[str, any]],  # (tensor_params)
        nodes: List[dict[str, any]],  # (node_params)
        opset: int = 17,
        name: str = "OpTest_Graph",
    ) -> ModelWrapper:
        """Creates a model using standard ONNX helper functions."""

        # Inputs
        input_protos: List[onnx.ValueInfoProto] = []
        for input in inputs:
            input_protos.append(helper.make_tensor_value_info(**input[0]))

        # Initialisers
        init_protos: List[onnx.TensorProto] = []
        for init in inits:
            init_protos.append(numpy_helper.from_array(**init))

        # Outputs
        output_protos: List[onnx.ValueInfoProto] = []
        for output in outputs:
            output_protos.append(helper.make_tensor_value_info(**output[0]))

        # Nodes
        node_protos: List[onnx.NodeProto] = []
        for node in nodes:
            node_protos.append(helper.make_node(**node))

        # Model
        model: onnx.ModelProto = helper.make_model(
            helper.make_graph(
                node_protos, name, input_protos, output_protos, init_protos
            ),
            opset_imports=[OperatorSetIdProto(version=opset)],
        )

        # Wrap the ONNX model in a QONNX model wrapper
        model_wrapper = ModelWrapper(model)

        # Annotate the model's input/output to the QONNX datatypes.
        for input in inputs:
            model_wrapper.set_tensor_datatype(input[0]["name"], DataType[input[1]])
        for output in outputs:
            model_wrapper.set_tensor_datatype(output[0]["name"], DataType[output[1]])

        return model_wrapper

    def apply_transforms(
        self,
        model: ModelWrapper,
        transform_list: List[Transformation],
        validate: bool = False,
        input_tensors: dict = None,
        tolerance: float = 1e-5,
    ) -> ModelWrapper:
        """Applies a list of QONNX transformations to a given model. If 'validate' is enabled,
        the function compares the output from model before and after the transforms were
        applied, to ensure the functionality of the model hasn't changed."""

        if validate:
            out_name = model.graph.output[0].name
            ref_output = oxe.execute_onnx(model, input_tensors)[out_name]

        for transformation in transform_list:
            model = model.transform(transformation)

        if validate:
            t_output = oxe.execute_onnx(model, input_tensors)[out_name]
            if not np.allclose(ref_output, t_output, atol=tolerance):
                raise RuntimeError(
                    f"Transformation {transformation} failed expected {ref_output=} but got {t_output=}"
                )

        return model

    def apply_builder_step(
        self,
        model: ModelWrapper,
        step: callable, 
        cfg_settings: dict = {}
    ) -> ModelWrapper:
        """Apply a FINN Builder step to a QONNX ModelWrapper. Takes in the Model,
        the step function to be executed, and any named parameters of that need to be
        used in the step's DataflowBuildConfig. These named parameters are passed via a
        dictionary, with the name of each parameter as its key."""
        
        # Default non-optional parameters for the DataflowBuildConfig class
        if 'output_dir' not in cfg_settings:
            cfg_settings['output_dir'] = os.path.join(OUTPUT_DIR, model.model.graph.name)
        if 'synth_clk_period_ns' not in cfg_settings:
            cfg_settings['synth_clk_period_ns'] = 4.0
        if 'generate_outputs' not in cfg_settings:
            cfg_settings['generate_outputs'] = []

        # Create a dummy config so we can call the step correctly.
        config: DataflowBuildConfig = DataflowBuildConfig(**cfg_settings)

        return step(model, config)
