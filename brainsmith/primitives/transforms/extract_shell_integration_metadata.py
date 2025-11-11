# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shell integration metadata extraction transform."""

import json

import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation


class ExtractShellIntegrationMetadata(Transformation):
    """Walks the ONNX graph and extracts all relevant metadata for shell integration
    handover."""

    def __init__(self, metadata_file: str):
        super().__init__()
        self.metadata_file: str = metadata_file
        self.md = {}

    def apply(self, model):
        graph = model.graph

        instreams = {}
        for input_tensor in graph.input:
            consumer = model.find_consumer(input_tensor.name)
            inst = registry.getCustomOp(consumer)
            instreams[input_tensor.name] = {
                "width": inst.get_instream_width(),
                "shape": inst.get_normal_input_shape(),
            }
        self.md["instreams"] = instreams

        outstreams = {}
        for output_tensor in graph.output:
            producer = model.find_producer(output_tensor.name)
            inst = registry.getCustomOp(producer)
            outstreams[output_tensor.name] = {
                "width": inst.get_outstream_width(),
                "shape": inst.get_normal_output_shape(),
            }
        self.md["outstreams"] = outstreams

        static_matmuls = {}
        for node in graph.node:
            if node.op_type == "MVAU_rtl":
                inst = registry.getCustomOp(node)
                mm = {}
                mm["MH"] = inst.get_nodeattr("MH")
                mm["MW"] = inst.get_nodeattr("MW")
                mm["SIMD"] = inst.get_nodeattr("SIMD")
                mm["PE"] = inst.get_nodeattr("PE")
                static_matmuls[node.name] = mm
        self.md["static_matmuls"] = static_matmuls

        with open(self.metadata_file, "w") as fp:
            json.dump(self.md, fp, indent=4)

        return (model, False)
