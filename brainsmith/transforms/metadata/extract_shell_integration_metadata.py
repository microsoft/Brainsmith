"""Shell integration metadata extraction transform."""

import json
from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry
from brainsmith.plugin.core import transform


@transform(
    name="ExtractShellIntegrationMetadata",
    stage="cleanup",
    description="Extract metadata for shell integration handover",
    author="shane.fleming",
    version="1.0.0",
    requires=["qonnx"]
)
class ExtractShellIntegrationMetadata(Transformation):
    """Walks the ONNX graph and extracts all relevant metadata for shell integration
    handover."""
    def __init__(self, metadata_file: str):
        super().__init__()
        self.metadata_file: str = metadata_file
        self.md = {}

    def apply(self, model):
        graph = model.graph

        # Extract instream widths
        instreams = {}
        for input_tensor in graph.input:
            consumer = model.find_consumer(input_tensor.name)
            inst = registry.getCustomOp(consumer)
            instream = {}
            instream['width'] = inst.get_instream_width() 
            instreams[input_tensor.name] = instream
            instream['shape'] = inst.get_normal_input_shape() 
        self.md['insteams'] = instreams

        # Extract outstream widths
        outstreams = {}
        for output_tensor in graph.output:
            producer = model.find_producer(output_tensor.name)
            inst = registry.getCustomOp(producer)
            outstream = {}
            outstream['width'] = inst.get_outstream_width() 
            outstreams[output_tensor.name] = outstream
            outstream['shape'] = inst.get_normal_output_shape()
        self.md['outsteams'] = outstreams
    
        static_matmuls = {}
        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                mm = {}
                mm['MH'] = inst.get_nodeattr("MH")
                mm['MW'] = inst.get_nodeattr("MW")
                mm['SIMD'] = inst.get_nodeattr("SIMD")
                mm['PE'] = inst.get_nodeattr("PE")
                static_matmuls[node.name] = mm
        self.md["static_matmuls"] = static_matmuls

        with open(self.metadata_file, "w") as fp:
            json.dump(self.md, fp, indent=4)

        return(model, False)