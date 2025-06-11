"""Metadata extraction and shell integration operations."""

import os
import shutil
import json
from finn.builder.build_dataflow_config import DataflowOutputType
from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry


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


def shell_metadata_handover_step(model, cfg):
    """
    Extract metadata for shell integration process.
    
    Category: metadata
    Dependencies: []
    Description: Extracts metadata for shell integration handover such as for the v80
    
    This information is stored in a json file that is passed to the build process.
    It adds this to the stitched_ip output directory and checks it exists ahead of time.
    """
    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        if os.path.isdir(cfg.output_dir + '/stitched_ip'):
            model = model.transform(ExtractShellIntegrationMetadata(cfg.output_dir + "/stitched_ip/shell_handover.json"))
            # copy over the ref IO *.npy files into the stitched_ip for handover
            shutil.copy(cfg.verify_input_npy, cfg.output_dir + '/stitched_ip')
            shutil.copy(cfg.verify_expected_output_npy, cfg.output_dir + '/stitched_ip')
            return model
        else:
            raise RuntimeError(f"Error: could not find stitched IP directory so unable to create metadata. Please ensure this is called after the create_stitched_ip step")
    return model