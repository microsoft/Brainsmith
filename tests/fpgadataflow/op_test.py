import pytest
import onnx
import numpy as np
import finn.core.onnx_exec as oxe
from abc import ABC, abstractmethod
from typing import List
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
           reducing boilerplate when implementing this fixture.
        """
        raise NotImplementedError("This OpTest's model() fixture is unimplemented.")
    

    @pytest.fixture(autouse=True)
    def model_specialised(
            self,
            model:ModelWrapper,
            input_tensors:dict,
            exec_mode:str,
        ) -> ModelWrapper:
        """A fixture that applys layer specialisation to the 'model' fixture, then returns it.
           The model is specialised differently depending on which execution mode is used (cppsim#
           or rtlsim).
        """

        # May parameterise these in the future.
        target_fpga = "xcv80-lsva4737-2MHP-e-S"
        target_clk_ns = 5

        transform_list = [
            SpecializeLayers(target_fpga),
            GiveUniqueNodeNames(),
            SetExecMode(exec_mode),
        ]
        
        if exec_mode is "cppsim":
            transform_list.append(PrepareCppSim())
            transform_list.append(CompileCppSim())
        if exec_mode is "rtlsim":
            transform_list.append(PrepareIP(target_fpga, target_clk_ns))
            transform_list.append(HLSSynthIP())
            transform_list.append(PrepareRTLSim())

        return self.apply_transforms(
            model=model,
            input_tensors=input_tensors,
            transform_list=transform_list,
            validate=True
        )
    
    @pytest.fixture
    def input_tensors(self, model) -> dict:
        """Creates the tensor(s) passed to the model, to be used by the simulation during
           testing. This fixture creates a tensor with random values, but can be overriden
           by subclasses to pass specific values."""
        input_t = {}
        for input in model.graph.input:
            input_value = gen_finn_dt_tensor(
                model.get_tensor_datatype(input.name),
                model.get_tensor_shape(input.name)
            )
            input_t[input.name] = input_value
        print(input_t)
        return input_t
    

    ##########################################
    #                  Tests                 #
    ##########################################

    # Ensure the number of cycles the layer takes to run in rtlsim
    # aligns with the expected number of cycles.
    def test_cycles(
            self, 
            model_specialised:ModelWrapper,
            exec_mode:str
        ) -> None:

        if exec_mode == "rtlsim":
            op_type = model_specialised.graph.node[0].op_type  # TODO: what if the node we're testing isn't the first node?
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
            inputs:List[tuple[dict[str,any], str]],     # (tensor_params, finn_dt)
            outputs:List[tuple[dict[str,any], str]],    # (tensor_params, finn_dt)
            inits:List[dict[str,any]],                  # (tensor_params)
            nodes:List[dict[str,any]],                  # (node_params)
            opset:int = 17,
            name:str = "OpTest_Graph",
        ) -> ModelWrapper:
        """Creates a model using standard ONNX helper functions. """

        # Inputs
        input_protos:List[onnx.ValueInfoProto] = []
        for input in inputs:
            input_protos.append(helper.make_tensor_value_info(**input[0]))

        # Initialisers
        init_protos:List[onnx.TensorProto] = []
        for init in inits:
            init_protos.append(numpy_helper.from_array(**init))
        
        # Outputs
        output_protos:List[onnx.ValueInfoProto] = []
        for output in outputs:
            output_protos.append(helper.make_tensor_value_info(**output[0]))

        # Nodes
        node_protos:List[onnx.NodeProto] = []
        for node in nodes:
            node_protos.append(helper.make_node(**node))
        
        # Model
        model:onnx.ModelProto = helper.make_model(
            helper.make_graph(node_protos,name,input_protos,output_protos,init_protos),
            opset_imports=[OperatorSetIdProto(version=opset)]
        )
        
        # Wrap the ONNX model in a QONNX model wrapper
        model_wrapper = ModelWrapper(model)

        # Annotate the LayerNorm's input/output to the QONNX datatypes.
        for input in inputs:
            model_wrapper.set_tensor_datatype(input[0]["name"], DataType[input[1]])
        for output in outputs:
            model_wrapper.set_tensor_datatype(output[0]["name"], DataType[output[1]])
        
        return model_wrapper
    
    def apply_transforms(
            self,
            model:ModelWrapper,
            transform_list:List[Transformation],
            validate:bool=False,
            validation_input:dict=None,
            tolerance=1e-5,
        ) -> ModelWrapper:
        """Applies a list of QONNX transformations to a given model. If 'validate' is enabled,
           the function compares the output from model before and after the transforms were
           applied, to ensure the functionality of the model hasn't changed."""

        if validate:
            out_name = model.graph.output[0].name
            ref_output = oxe.execute_onnx(model, validation_input)[out_name]

        for transformation in transform_list:
            model = model.transform(transformation)

        if validate:
            t_output = oxe.execute_onnx(model, validation_input)[out_name]
            assert np.allclose(ref_output, t_output, atol=tolerance)
        
        return model