############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""InferDataflowGraph: Meta-transform orchestrating topology and pattern matching.

Two-phase workflow:
1. Topology analysis: Insert infrastructure kernels based on graph structure
2. Pattern matching: Convert ONNX nodes to computational kernels

This meta-transform provides a convenient one-liner for complete dataflow graph
inference, while maintaining compatibility with manual orchestration for users
who need fine-grained control.

Example usage:
    from brainsmith.primitives.transforms import InferDataflowGraph

    # One-liner for complete inference
    model = model.transform(InferDataflowGraph())

    # Equivalent to manual orchestration:
    model = model.transform(InsertDuplicateStreams())
    model = model.transform(InferKernelList())

    # Advanced: Selective inference
    model = model.transform(InferDataflowGraph(
        kernel_classes=[AddStreams, Softmax],  # Specific kernels only
        skip_topology=False,                    # Run topology analysis
    ))
"""

import logging
from typing import Optional, List, Type

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper

from .insert_duplicate_streams import InsertDuplicateStreams
from .infer_kernel_list import InferKernelList

logger = logging.getLogger(__name__)


class InferDataflowGraph(Transformation):
    """Meta-transform for complete dataflow graph inference.

    Orchestrates topology transforms and pattern matching in the correct order:
    1. Topology analysis (infrastructure insertion):
       - InsertDuplicateStreams: Detect tensor fanout >= 2
       - InsertFIFO: Buffer depth optimization (future)
       - InsertDWC: Stream width mismatch correction (future)

    2. Pattern matching (computational kernels):
       - InferKernelList: Convert ONNX nodes to hardware kernels

    This two-phase approach ensures:
    - Clean topology analysis on ONNX graph
    - Infrastructure nodes present during kernel inference
    - Correct dataflow routing from the start

    Args:
        kernel_classes: Optional list of kernel classes to infer.
                       If None, infers all registered computational kernels.
                       Infrastructure kernels are automatically filtered out.
        skip_topology: Skip topology analysis (infrastructure insertion).
                      Default: False
        skip_pattern_matching: Skip pattern matching (computational kernels).
                              Default: False

    Usage:
        # Simple one-liner (most common)
        model = model.transform(InferDataflowGraph())

        # Explicit control (equivalent to one-liner)
        model = model.transform(InsertDuplicateStreams())
        model = model.transform(InferKernelList())

        # Only topology (no kernel conversion)
        model = model.transform(InferDataflowGraph(skip_pattern_matching=True))

        # Only pattern matching (assume topology already inserted)
        model = model.transform(InferDataflowGraph(skip_topology=True))

        # Custom kernel list with topology
        from brainsmith.kernels.addstreams import AddStreams
        from brainsmith.kernels.softmax import Softmax

        model = model.transform(InferDataflowGraph(
            kernel_classes=[AddStreams, Softmax]
        ))

    Notes:
        - Infrastructure kernels (DuplicateStreams, FIFO, etc.) are registered
          in the component registry but marked with is_infrastructure=True
        - InferKernelList automatically filters infrastructure kernels when
          kernel_classes=None (backward compatible mode)
        - Topology transforms run before pattern matching to analyze clean
          ONNX graph structure
        - Both phases aggregate graph_modified flags correctly
    """

    def __init__(
        self,
        kernel_classes: Optional[List[Type]] = None,
        skip_topology: bool = False,
        skip_pattern_matching: bool = False,
    ):
        """Initialize meta-transform.

        Args:
            kernel_classes: Optional list of kernel classes to infer.
                          If None, infers all registered computational kernels
                          (infrastructure kernels automatically filtered).
            skip_topology: Skip topology analysis (infrastructure insertion).
            skip_pattern_matching: Skip pattern matching (computational kernels).
        """
        super().__init__()
        self.kernel_classes = kernel_classes
        self.skip_topology = skip_topology
        self.skip_pattern_matching = skip_pattern_matching

    def apply(self, model: ModelWrapper):
        """Apply two-phase inference workflow.

        Phase 1: Topology analysis
        - InsertDuplicateStreams: Detect tensor fanout >= 2 and insert routing
        - (Future: InsertFIFO, InsertDWC)

        Phase 2: Pattern matching
        - InferKernelList: Convert ONNX nodes to computational kernels
        - Infrastructure kernels automatically filtered

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)

            - transformed_model: ModelWrapper with dataflow graph
            - graph_modified_flag: True if any transform modified the graph
        """
        graph_modified = False

        # Phase 1: Topology analysis (infrastructure)
        if not self.skip_topology:
            logger.info("Phase 1: Topology analysis (infrastructure insertion)")
            model, modified = self._apply_topology_transforms(model)
            graph_modified = graph_modified or modified

            if modified:
                logger.debug("Topology phase modified graph")

        # Phase 2: Pattern matching (computational)
        if not self.skip_pattern_matching:
            logger.info("Phase 2: Pattern matching (computational kernels)")
            transform = InferKernelList(self.kernel_classes)
            model, modified = transform.apply(model)
            graph_modified = graph_modified or modified

            if modified:
                logger.debug("Pattern matching phase modified graph")

        if graph_modified:
            logger.info("InferDataflowGraph completed: graph was modified")
        else:
            logger.info("InferDataflowGraph completed: no modifications needed")

        return (model, graph_modified)

    def _apply_topology_transforms(self, model: ModelWrapper):
        """Run all infrastructure insertion transforms.

        Currently implemented:
        - InsertDuplicateStreams: Detect tensor fanout >= 2

        Future topology transforms:
        - InsertFIFO: Buffer depth optimization for timing closure
        - InsertDWC: Stream width mismatch detection and correction

        These transforms analyze graph structure and insert infrastructure
        kernels (routing, buffering, conversion) as needed for dataflow
        execution. They run before pattern matching to analyze clean ONNX
        topology.

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        graph_modified = False

        # DuplicateStreams: Detect tensor fanout
        logger.debug("Running InsertDuplicateStreams")
        transform = InsertDuplicateStreams()
        model, modified = transform.apply(model)
        graph_modified = graph_modified or modified

        if modified:
            logger.debug("InsertDuplicateStreams inserted infrastructure nodes")

        # Future topology transforms:
        #
        # # FIFO: Buffer depth optimization
        # logger.debug("Running InsertFIFO")
        # transform = InsertFIFO()
        # model, modified = transform.apply(model)
        # graph_modified = graph_modified or modified
        #
        # # DWC: Stream width mismatch correction
        # logger.debug("Running InsertDWC")
        # transform = InsertDWC()
        # model, modified = transform.apply(model)
        # graph_modified = graph_modified or modified

        return (model, graph_modified)
