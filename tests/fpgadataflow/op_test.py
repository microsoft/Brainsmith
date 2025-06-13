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
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.builder.build_dataflow_steps import step_specialize_layers
from finn.builder.build_dataflow_config import DataflowBuildConfig


class OpTest(ABC):
    """An abstract class which uses PyTest's functionality to make writing tests for Brainsmith operators easier.
    \ \ 
        This class contains tests and fixtures, which are features of PyTest.
        - All methods beginning with ``f_`` are fixtures, i.e. :func:`OpTest.f_model`. See `PyTest's fixture documentation <https://docs.pytest.org/en/7.1.x/how-to/fixtures.html>`_ for more info.
        - All methods beginning with ``test_`` are tests, i.e. :func:`OpTest.test_cycles`. See `PyTest's getting started page <https://docs.pytest.org/en/stable/>`_ for more info.
        - All methods that do not have these prefixes are helper functions, to be used to aid in writing your own tests."""

    ##########################################
    #                Fixtures                #
    ##########################################

    @pytest.fixture()
    @abstractmethod
    def f_model(self) -> ModelWrapper:
        """An abstract fixture that generates the QONNX ModelWrapper to be tested (when
        implemented). Each test MUST override this fixture, otherwise any PyTests
        will result in a NotImplementedError. Helper functions such as create_model()
        and run_transforms() may be useful in reducing boilerplate when implementing this
        fixture.

        :return: A :class:`ModelWrapper` containing the ONNX graph we'll use for testing
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

        raise NotImplementedError("This OpTest's f_model() fixture is unimplemented.")

    @pytest.fixture()
    def f_model_hw(
        self,
        f_model: ModelWrapper,
        f_infer_hw_transform: Transformation,
        f_input_tensors,
    ) -> ModelWrapper:
        """Converts all ONNX layers of a specific type to hardware layers,
        using a given inference function. If that function does not exist
        (i.e. f_infer_hw_transform == 'None'), then the model is directly
        passed through. All fixtures reliant on hardware inference should
        check if f_infer_hw_transform == 'None' before using this fixture.

        :param f_model: Auto-populated by the :func:`OpTest.f_model` fixture's return value
        :type f_model: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_infer_hw_transform: Auto-populated by the :func:`OpTest.f_infer_hw_transform` fixture's return value
        :type minfer_hw_transformodel: :class:`qonnx.transformation.base.Transformation`

        :return: A :class:`ModelWrapper` containing the converted model
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

        if f_infer_hw_transform is not None:
            return self.apply_transforms(
                model=f_model,
                transform_list=[f_infer_hw_transform],
                input_tensors=f_input_tensors,
            )
        else:
            warn("skipped f_model_hw step, as no f_infer_hw_transform was provided.")
            return f_model

    @pytest.fixture()
    def f_model_specialised(
        self,
        f_model_hw: ModelWrapper,
        f_target_fpga: str,
        f_output_dir: str,
    ) -> ModelWrapper:
        """A fixture that applies layer specialisation to the 'f_model_hw'
        fixture, then returns the resulting ModelWrapper.;

        :param f_model_hw: Auto-populated by the :func:`OpTest.f_model_hw` fixture's return value
        :type f_model_hw: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_target_fpga: Auto-populated by the :func:`OpTest.f_target_fpga` fixture's return value
        :type f_target_fpga: str

        :param f_output_dir: Auto-populated by the :func:`OpTest.f_output_dir` fixture's return value
        :type f_output_dir: str

        :return: A :class:`ModelWrapper` containing the specialised model
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

        specialised_model: ModelWrapper = self.apply_builder_step(
            f_model_hw,
            step_specialize_layers,
            f_output_dir,
            dict(fpga_part=f_target_fpga),
        )

        return specialised_model

    @pytest.fixture
    def f_infer_hw_transform(self) -> Transformation:
        """The transformation to infer a hardware layer from a standard ONNX layer.
        If this fixture returns 'None', OpTest assumes to skip hardware inference.

        :return: The :class:`Transformation` we'll apply when inferring hardware layers
                 (Default: ``None``)
        :rtype: :class:`qonnx.transformation.base.Transformation`"""

        return None

    @pytest.fixture
    def f_target_fpga(self) -> str:
        """The fpga we're targeting for testing. Can be overridden by test classes.

        :return: The name of the fpga we're targeting for testing
                 (Default: ``"xcv80-lsva4737-2MHP-e-S"``)
        :rtype: str"""

        return "xcv80-lsva4737-2MHP-e-S"

    @pytest.fixture
    def f_target_node(self) -> int:
        """The index of the node in the model we're focusing on. Allows for multiple
        nodes to be present, with tests that only target a specific node. Defaults to
        the first node. Can be overridden.

        :return: The index of the node we wish to focus our tests on.
                 (Default: ``0``)
        :rtype: int"""

        return 0

    @pytest.fixture
    def f_input_tensors(self, f_model: ModelWrapper) -> dict[str, any]:
        """Creates the tensor(s) passed to the model, to be used by the simulation during
        testing. By default, this fixture creates a tensor with random values, but can be
        overridden by tests to pass specific values.

        :param f_model: Auto-populated by the :func:`OpTest.f_model` fixture's return value
        :type f_model: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :return: A dictionary. Each entry in the dictionary contains an input tensor's name as its key, and the data we wish to pass to it as its value.
        :rtype: :class:`qonnx.transformation.base.Transformation`"""

        input_t = {}
        for input in f_model.graph.input:
            input_value = gen_finn_dt_tensor(
                f_model.get_tensor_datatype(input.name),
                f_model.get_tensor_shape(input.name),
            )
            input_t[input.name] = input_value
        return input_t

    @pytest.fixture
    def f_save_models(self) -> bool:
        """If this fixture is overridden to return True, the 'f_auto_saver' fixture
        will automatically save the outputs of all 'f_model' fixtures to .onnx files.


        :return: Whether or not the auto-saver should save intermediate models.
                 (Default: ``False``)
        :rtype: bool"""

        return False

    @pytest.fixture(autouse=True)
    def f_auto_saver(
        self,
        request: pytest.FixtureRequest,
        f_save_models: bool,
        f_output_dir: str,
    ) -> None:
        """Saves the output of each intermediate model step to a .onnx file.

        If the 'f_save_models' fixture evaluates to true, this fixture
        automatically scrapes all fixtures with names that start with 'f_model'
        (i.e. 'f_model', 'f_model_specialised') and saves their output to a .onnx file
        (if the fixtures return a ModelWrapper).

        :param request: Auto-populated by the :func:`pytest.fixtures.FixtureRequest()` fixture's return value
        :type request: :class:`pytest.fixtures.FixtureRequest`

        :param f_save_models: Auto-populated by the :func:`OpTest.f_save_models` fixture's return value
        :type f_save_models: bool

        :param f_output_dir: Auto-populated by the :func:`OpTest.f_output_dir` fixture's return value
        :type f_output_dir: str"""

        if f_save_models:

            # Attempt to make the directory we'll store our intermediate models in.
            models_directory = os.path.join(f_output_dir, request.module.__name__)
            try:
                os.mkdir(models_directory)
            except FileExistsError:
                warn(
                    f"Overwriting saved models in existing directory {models_directory}."
                )

            # Save the output of each ModelWrapper fixture that starts with "f_model"
            # (i.e. f_model, f_model_specialised). Assumes these fixtures return ModelWrappers.
            for fixture in filter(
                lambda x: x.startswith("f_model"), request.fixturenames
            ):

                filename = os.path.join(models_directory, fixture) + ".onnx"
                output = request.getfixturevalue(fixture)

                if isinstance(output, ModelWrapper):
                    output.save(filename)
                else:
                    raise TypeError(
                        f"Model fixture {fixture} has return type {type(output)}. Expected ModelWrapper."
                    )

        return

    @pytest.fixture
    def f_output_dir(self) -> str:
        """The directory we'll save the output of our tests to. By default,
        OpTest saves to a directory called "test_output", which is created
        in the same directory as the python file containing the test.

        :return: The output directory that all test output will be saved to.
                 (Default: ``"__file__/test_output"``)
        :rtype: str"""

        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_output")

    @pytest.fixture(autouse=True)
    def f_create_output_dir(self, f_output_dir: str) -> None:
        """Automatically makes a test output directory if none exists. This
        fixture auto-runs before all test functions. If the directory exists,
        this fixture does nothing.

        :param f_output_dir: Auto-populated by the :func:`OpTest.f_output_dir` fixture's return value
        :type f_output_dir: str"""

        try:
            os.mkdir(f_output_dir)
        except FileExistsError:
            pass

    @pytest.fixture(params=["cppsim", "rtlsim"])
    def f_exec_mode(self2, request) -> str:
        """Parameterised fixture. Provides the exec mode parameter ("cppsim or rtlsim")

        :return: The exec mode
                 (Default: ``"cppsim"`` or ``"rtlsim"``)
        :rtype: str"""
        return request.param

    ##########################################
    #                  Tests                 #
    ##########################################

    def test_cycles(
        self, f_model_specialised: ModelWrapper, f_target_node: int, f_exec_mode: str
    ) -> None:
        """Ensure the number of cycles the layer takes to run in rtlsim aligns
        with the expected number of cycles.

        :param f_model_specialised: Auto-populated by the :func:`OpTest.f_model_specialised` fixture's return value
        :type f_model_specialised: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_target_node: Auto-populated by the :func:`OpTest.f_target_node` fixture's return value
        :type f_target_node: int

        :param f_exec_mode: Auto-populated by OpTest's :func:`OpTest.f_exec_mode` PyTest parameter.
            Check the fixture for all possible parameterisations.
        :type f_exec_mode: str"""

        if f_exec_mode == "rtlsim":
            op_type = f_model_specialised.graph.node[f_target_node].op_type
            node = f_model_specialised.get_nodes_by_op_type(op_type)[0]
            inst = getCustomOp(node)
            cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
            exp_cycles_dict = f_model_specialised.analysis(exp_cycles_per_layer)
            exp_cycles = exp_cycles_dict[node.name]
            assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
            assert exp_cycles != 0

    def test_conv_to_hardware(self, f_model, f_model_hw, f_input_tensors):
        """Compare the outputs of 'f_model' and 'f_model_hw', when executed using
        ONNX runtime. Ensure that they are functionally identical.

        :param f_model: Auto-populated by the :func:`OpTest.f_model` fixture's return value
        :type f_model: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_model_hw: Auto-populated by the :func:`OpTest.f_model_hw` fixture's return value
        :type f_model_hw: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_input_tensors: Auto-populated by the :func:`OpTest.f_input_tensors` fixture's return value
        :type f_input_tensors: dict"""
        out_name = f_model.graph.output[0].name

        # Generate our reference and our compared outputs
        ref_output = oxe.execute_onnx(f_model, f_input_tensors)[out_name]
        cmp_output = oxe.execute_onnx(f_model_hw, f_input_tensors)[out_name]
        
        # Compare f_model_hw's output to f_model's output.
        assert np.allclose(ref_output, cmp_output, atol=1e-5)
    
    def test_specialise_layers(self, f_model_hw, f_model_specialised, f_input_tensors):
        """Compare the outputs of 'f_model_hw' and 'f_model_specialised', when executed using
        ONNX runtime. Ensure that they are functionally identical.

        :param f_model_hw: Auto-populated by the :func:`OpTest.f_model_hw` fixture's return value
        :type f_model_hw: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_model_specialised: Auto-populated by the :func:`OpTest.f_model_specialised` fixture's return value
        :type f_model_specialised: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param f_input_tensors: Auto-populated by the :func:`OpTest.f_input_tensors` fixture's return value
        :type f_input_tensors: dict"""
        out_name = f_model_hw.graph.output[0].name

        # Generate our reference and our compared outputs
        ref_output = oxe.execute_onnx(f_model_hw, f_input_tensors)[out_name]
        cmp_output = oxe.execute_onnx(f_model_specialised, f_input_tensors)[out_name]
        
        # Compare f_model_specialised's output to f_model_hw's output.
        assert np.allclose(ref_output, cmp_output, atol=1e-5)


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
        """Creates a model using ONNX.helper and QONNX.ModelWrapper. Removes the need for
        additional boilerplate code.

        :param inputs: A list of tuples, with each tuple representing an input of the
            model. These tuples have two elements: the first element in each tuple is a dictionary
            containing the named parameters you're passing to ``onnx.helper.make_tensor_value_info()``.
            The second element is the ``QONNX.core.datatype`` string that that input will be set to.
        :type inputs: List(tuple(dict, str))

        :param outputs: A list of tuples, with each tuple representing an output of the
            model. These tuples have two elements: the first element in each tuple is a dictionary
            containing  any named parameters you're passing to ``onnx.helper.make_tensor_value_info()``.
            The second element is the ``QONNX.core.datatype`` string that that input will be set to.
        :type outputs: List(tuple(dict, str))

        :param inits: A list of dictionaries, each dictionary representing an initialiser of the model. These should named parameters of ```onnx.numpy_helper.from_array`` <https://onnx.ai/onnx/api/numpy_helper.html#onnx.numpy_helper.to_array>`_.
        :type inits: List(dict)

        :param nodes: A list of dictionaries, each dictionary representing a node of the model. These should named parameters of ```onnx.numpy_helper.from_array`` <https://onnx.ai/onnx/api/helper.html#onnx.helper.make_node>`_.
        :type nodes: List(dict)

        :param opset: The opset of the generated model. Defaults to 17.
                      (Default: ``17``)
        :type opset: int, optional

        :param name: The name of the generated model.
                     (Default: ``"OpTest_Graph"``)
        :type name: str, optional

        :return: A :class:`ModelWrapper` containing the generated model
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

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
        transform_list: List[Transformation] | List[List[Transformation]],
        input_tensors: dict = None,
        tolerance: float = 1e-5,
        output_dir: str = None,
    ) -> ModelWrapper:
        """Applies a list of QONNX transformations to a given model.

        If 'input_tensors' are provided, the function compares the model's output pre
        and post-transformation. 'transform_list' can accept either a list of transforms,
        or a nested list of transforms. This affects how model validation is performed.
        Regular transform-lists are validated after every transform. Nested transform-
        lists are validated after every sub-list, so transforms [[1,2,3],[4,5]] would
        be validated between 3-4, and after 5.

        If an 'output_dir' is provided, the model will be saved to that directory after
        ANY transform is applied.

        :param model: The :class:`ModelWrapper` we'll apply our transform list to
        :type model: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param transform_list: A list (or nested list) containing each :class:`Transformation`
            we'll apply to the model
        :type transform_list: List(Transformation) or List(List(Transformation))

        :param input_tensors: The dictionary containing the names of tensors (key)
            and their inputs as numpy arrays (value) which we'd use to validate our model.
            Providing an input dictionary enables model validation.
            (Default: ``None``)
        :type input_tensors: dict

        :param tolerance: The acceptable tolerance that model outputs can differ during validation.
            (Default: ``1e-5``)
        :type tolerance: float

        :param output_dir: The output directory used to save between every transformation step.
            Providing an input directory enables intermediate model saving.
            (Default: ``None``)
        :type output_dir: str

        :return: A :class:`ModelWrapper` containing the transformed model
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

        if input_tensors is not None:
            # Generate reference model output to compare our transformed output to.
            out_name = model.graph.output[0].name
            ref_output = oxe.execute_onnx(model, input_tensors)[out_name]

        for index, transform in enumerate(transform_list):

            if isinstance(transform, list):
                # If 'transform' is a list, we apply each sub-transform in the list, then validate.
                for subindex, i in enumerate(transform):
                    model = model.transform(i)
                    if output_dir is not None:
                        model.save(
                            os.path.join(
                                output_dir,
                                f"{str(index)}_{str(subindex)}_{i.__class__.__name__}.onnx",
                            )
                        )
            else:
                # If 'transform' is an single transform, we validate between each transform.
                model = model.transform(transform)
                if output_dir is not None:
                    model.save(
                        os.path.join(
                            output_dir,
                            f"{str(index)}_{transform.__class__.__name__}.onnx",
                        )
                    )

            if input_tensors is not None:
                # Compare the transformed model's output to the reference output.
                t_output = oxe.execute_onnx(model, input_tensors)[out_name]
                if not np.allclose(ref_output, t_output, atol=tolerance):
                    raise RuntimeError(
                        f"Transformation step {transform} failed:\n"
                        + f"expected {ref_output=} but got {t_output=}"
                    )

        return model

    def apply_builder_step(
        self,
        model: ModelWrapper,
        step: callable,
        output_dir: str,
        cfg_settings: dict = {},
    ) -> ModelWrapper:
        """Apply a FINN Builder step to a QONNX ModelWrapper. Takes in the Model,
        the step function to be executed, and any named parameters of that need to be
        used in the step's DataflowBuildConfig. These named parameters are passed via a
        dictionary, with the name of each parameter as its key.

        :param model: The :class:`ModelWrapper` we'll apply our builder step to
        :type model: :class:`qonnx.core.modelwrapper.ModelWrapper`

        :param step: A list (or nested list) containing each :class:`Transformation` we'll
            apply to the model
        :type step: callable

        :param output_dir: The directory that the builder step will use, if the step
            involves saving intermediate models
        :type output_dir: str

        :param cfg_settings: A dictionary containing named parameters to pass to the step's
            :class:`finn.builder.build_dataflow_config.DataFlowBuildConfig`
        :type cfg_settings: dict, optional

        :return: A :class:`ModelWrapper` containing the transformed model
        :rtype: :class:`qonnx.core.modelwrapper.ModelWrapper`"""

        # Default non-optional parameters for the DataflowBuildConfig class
        if "output_dir" not in cfg_settings:
            cfg_settings["output_dir"] = os.path.join(
                output_dir, model.model.graph.name
            )
        if "synth_clk_period_ns" not in cfg_settings:
            cfg_settings["synth_clk_period_ns"] = 4.0
        if "generate_outputs" not in cfg_settings:
            cfg_settings["generate_outputs"] = []
        if "board" not in cfg_settings:
            cfg_settings["board"] = "ZCU104"

        # Create a dummy config so we can call the step correctly.
        config: DataflowBuildConfig = DataflowBuildConfig(**cfg_settings)

        return step(model, config)
