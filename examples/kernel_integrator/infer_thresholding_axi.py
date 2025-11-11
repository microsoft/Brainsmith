############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Transform to convert MultiThreshold nodes to ThresholdingAxi KernelOp.

Matches the behavior of InferThresholdingLayer but targets the auto-generated
ThresholdingAxi RTL implementation.
"""

import qonnx.core.data_layout as DataLayout
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.onnx import nchw_to_nhwc


# QONNX wrapper to ONNX model graphs
class InferThresholdingAxi(Transformation):
    """Convert any MultiThreshold into a standalone thresholding HLS layer."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "MultiThreshold":
                thl_input = node.input[0]
                thl_threshold = node.input[1]
                thl_output = node.output[0]
                thl_in_shape = model.get_tensor_shape(thl_input)
                model.get_tensor_shape(thl_threshold)
                idt = model.get_tensor_datatype(thl_input)
                tdt = model.get_tensor_datatype(thl_threshold)

                # check layout of inputs/outputs, and convert if needed
                # check layout and convert if necessary
                thl_in_layout = model.get_tensor_layout(thl_input)
                if thl_in_layout == DataLayout.NCHW:
                    thl_input = nchw_to_nhwc(thl_input, model, node_ind)
                    node_ind += 1
                    thl_in_shape = model.get_tensor_shape(thl_input)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                thl_output_layout = model.get_tensor_layout(thl_output)
                if thl_output_layout == DataLayout.NCHW:
                    thl_output = nchw_to_nhwc(thl_output, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume number of channels is in last dimension
                ifc = int(thl_in_shape[-1])
                # create node with no parallelization first
                pe = 1

                odt = model.get_tensor_datatype(thl_output)
                scale = getCustomOp(node).get_nodeattr("out_scale")
                assert scale == 1.0, (
                    node.name + ": MultiThreshold out_scale must be 1 for HLS conversion."
                )
                actval = getCustomOp(node).get_nodeattr("out_bias")
                assert int(actval) == actval, (
                    node.name + ": MultiThreshold out_bias must be integer for HLS conversion."
                )
                actval = int(actval)

                # a signed activation should always have a negative bias,
                # but BIPOLAR uses the -1 as 0 encoding so the assert does not apply
                if odt != DataType["BIPOLAR"]:
                    assert (not odt.signed()) or (actval < 0), (
                        node.name + ": Signed output requires actval < 0"
                    )

                new_node = helper.make_node(
                    "ThresholdingAxi",
                    [thl_input, thl_threshold],
                    [thl_output],
                    domain="brainsmith.examples.kernel_integrator.kernel",
                    backend="RTL",
                    CHANNELS=ifc,
                    PE=pe,
                    BIAS=actval,
                    inputDataType=idt.name,
                    weightDataType=tdt.name,
                    outputDataType=odt.name,
                    name="AutoThresholdingAxi_" + node.name,
                )

                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        return (model, graph_modified)

