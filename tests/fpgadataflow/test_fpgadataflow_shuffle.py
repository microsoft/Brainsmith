import pytest
import torch
import torch.onnx
from torch import nn
import onnx
import tempfile

from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.core.modelwrapper import ModelWrapper
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

from finnbrainsmith.transformation.shuffle_helpers import shuffle_perfect_loopnest_coeffs
from finnbrainsmith.transformation.convert_to_hw_layers import InferShuffle

test_fpga_part:str = "xcv80-lsva4737-2MHP-e-S"
test_synth_clk_period_ns:int = 5

class PytorchShuffle(nn.Module):
    """ From pytorch create a reshape and transpose combination
    that can be used for testing """

    def __init__(self, transpose_perm:tuple[int], 
            reshape1_shape:tuple[int]=None, 
            reshape2_shape:tuple[int]=None
        )->None:
        super(PytorchShuffle, self).__init__()
        self.transpose_perm = transpose_perm
        self.reshape1_shape = reshape1_shape
        self.reshape2_shape = reshape2_shape

    def forward(self, x):
        if self.reshape1_shape is not None:
            x = x.view(*self.reshape1_shape)
        x = x.permute(*self.transpose_perm)
        if self.reshape2_shape is not None:
            x = x.view(*self.reshape2_shape)
        return x

def construct_onnx_model(
        input_shape:tuple[int],
        transpose_perm:tuple[int],
        reshape1_shape:tuple[int],
        reshape2_shape:tuple[int],
        dt:DataType,
    )->ModelWrapper:

    """ Creates an ONNX model that can be used for testing
    the shuffle operation compiler integration. Uses the 
    pytorch methods in PytorchShuffle to generate the model. """

    dummy_input = torch.randn(*input_shape)
    model = PytorchShuffle(
            transpose_perm=transpose_perm,
            reshape1_shape=reshape1_shape,
            reshape2_shape=reshape2_shape
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
        model_input = torch.rand(input_shape)
        export_qonnx(model, model_input, temp_file.name, opset_version=17) 
        qonnx_cleanup(temp_file.name, out_file=temp_file.name)

        new_model = ModelWrapper(temp_file.name)
        new_model.set_tensor_datatype(new_model.graph.input[0].name, dt)
        new_model.set_tensor_datatype(new_model.graph.output[0].name, dt)
        new_model.transform(InferShapes())
        new_model.transform(InferDataTypes())
        return new_model
    raise RuntimeError(f"Error unable to export the ONNX file to the temporary location")


@pytest.mark.fpgadataflow
def test_pytorch_to_specialised():
    model = construct_onnx_model(
                input_shape=(1,128,384),
                transpose_perm=(0,2,1,3),
                reshape1_shape=(1,128,12,32),
                reshape2_shape=None,
                dt=DataType["INT8"]
            ) 

    # Attempt to build the HLS for this
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, test_synth_clk_period_ns))


def make_single_shuffle_modelwrapper(
        simd:int=1,
        data_type=DataType["UINT8"],
        in_reshaped:tuple[int]=(1, 128, 12, 32),
        in_shape:tuple[int]=(1, 128, 384),
        inner_moves:bool=False,
        loop_coeffs:tuple[int]=(49152, 49152, 4096, 32),
        out_reshaped:tuple[int]=(1, 12, 128, 32),
        out_shape:tuple[int]=(1, 12, 128, 32),
        perm:tuple[int]=(0, 2, 1, 3),
        name:str="Shuffle_0"
        ):
    ''' Create a single Shuffle node '''
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(in_shape))
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, list(out_reshaped))

    new_node = helper.make_node(
            "Shuffle",
            ["global_in"],
            ["global_out"],
            domain="finnbrainsmith.custom_op.fpgadataflow",
            backend="fpgadataflow",
            name=name,
            simd=simd,
            data_type=data_type.name,
            in_reshaped=in_reshaped,
            in_shape=in_shape,
            inner_moves=inner_moves,
            loop_coeffs=loop_coeffs,
            out_reshaped=out_reshaped,
            out_shape=out_shape,
            perm=perm
    )

    graph = helper.make_graph(
            [new_node],
            "shuffle_graph",
            inputs=[inp],
            outputs=[outp]
    )

    model = qonnx_make_model(graph)
    model = ModelWrapper(model)

    model.set_tensor_datatype("global_in", data_type)
    model.set_tensor_datatype("global_out", data_type)

    return model


@pytest.mark.fpgadataflow
def test_convert_to_hw_shuffle_layer():
    ''' Checks the conversion of a shuffle layer into hardware '''
    model = make_single_shuffle_modelwrapper() 

    # Attempt to build the HLS for this
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, test_synth_clk_period_ns))
    

@pytest.mark.parametrize("in_shape", [(1, 128, 384)])
@pytest.mark.parametrize("in_reshaped", [(1, 128, 12, 32)])
@pytest.mark.parametrize("out_shape", [(1, 12, 128, 32)])
@pytest.mark.parametrize("out_reshaped", [(1, 12, 128, 32)])
@pytest.mark.parametrize("perm", [(0, 2, 1, 3)])
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd1"])
@pytest.mark.fpgadataflow
def test_cppsim_shuffle_layer(in_shape, in_reshaped, out_shape, out_reshaped, perm, datatype, simd):
    ''' Checks cppsim of the shuffle_hls layer '''
    dt = DataType[datatype]
    simd = int(simd[-1])

    model = make_single_shuffle_modelwrapper(
                simd=simd,
                data_type=dt,
                in_reshaped=in_reshaped,
                in_shape=in_shape,
                inner_moves=False,
                loop_coeffs=shuffle_perfect_loopnest_coeffs(in_reshaped,perm),
                out_reshaped=out_reshaped,
                out_shape=out_shape,
                perm=perm
            )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name : input}

    # Attempt to build the HLS for this
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    

