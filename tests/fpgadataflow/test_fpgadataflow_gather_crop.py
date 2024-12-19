import pytest
import onnxruntime as ort
import numpy as np
import os

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames


from finnbrainsmith.transformation.convert_to_hw_layers import InferCropFromGather

from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

def make_gather_node(axis):
    return helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["output"],
        axis=axis
    )

def make_gather_graph(indices, axis):

    size = indices.shape[0]

    i_shape = [1, 128,  384]
    o_shape = [1, size, 384]

    # Define the input tensor
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, i_shape)

    # Define the output tensor
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, o_shape)

    indices = helper.make_tensor('indices', TensorProto.INT64, [len(indices)], indices)

    # Create the graph
    graph = helper.make_graph(
        nodes = [],
        name = 'GatherGraph',
        inputs = [data],
        outputs = [output],
        initializer = [
            indices,
        ]
    )

    # Create the QONNX model
    model = qonnx_make_model(graph, producer_name="com.brainsmith")
    model = ModelWrapper(model, fix_missing_initializer_valueinfo=True)

    model.graph.node.append(make_gather_node(axis))
    model.save("gather_crop.onnx")
    return model

@pytest.mark.parametrize("simd", [1, 2, 32])
@pytest.mark.parametrize("indices", [[0], [1], [4, 5, 6], [126], [127]])
def test_fpgadataflow_gather_crop(simd, indices, axis=1):
    test_fpga_part = "xczu3eg-sbva484-1-e"

    indices = np.array(indices)
    model = make_gather_graph(indices, axis=axis)

    # Run the model using the onnx runtime
    ort_session = ort.InferenceSession("gather_crop.onnx")
    ort_inputs = {
        "data": np.random.rand(1, 128, 384).astype(np.float32),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    expected = np.take(ort_inputs["data"], indices, axis=axis)

    # Check the output shape
    assert ort_outs[0].shape == expected.shape

    # Check the output values
    assert np.allclose(ort_outs[0], expected)

    model = model.transform(InferCropFromGather(simd))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    output = oxe.execute_onnx(model, {"data": ort_inputs["data"]})

    assert np.allclose(output['output'], ort_outs[0])

    test_synth_clk_period_ns = 10
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    model.save("gather_crop_infered.onnx")
    os.environ["LIVENESS_THRESHOLD"] = str(1000000000)
    output = oxe.execute_onnx(model, ort_inputs)
    assert np.allclose(output['output'], ort_outs[0])


