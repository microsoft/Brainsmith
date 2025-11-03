############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel operator base class."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union, TYPE_CHECKING

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from onnx import NodeProto

from .schemas import KernelSchema
from .dse_models import KernelDesignSpace, KernelDesignPoint
from .builder import BuildContext, DesignSpaceBuilder
from .transformation import TransformationResult
from .ordered_dimension import OrderedDimension

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Kernel operator exception with node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class.

    Shapes extracted from ModelWrapper context, never stored in nodeattrs.
    Subclasses implement build_schema() to construct their KernelSchema.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Build and freeze schema from node structure
        self.kernel_schema = self.build_schema(onnx_node, model=None)

        # Two-phase caching system for DSE performance
        self._design_space: Optional['KernelDesignSpace'] = None  # Built once, never invalidated
        self._design_point: Optional['KernelDesignPoint'] = None  # Rebuilt on param change

    @classmethod
    @abstractmethod
    def build_schema(cls, node: NodeProto, model: Optional[ModelWrapper]) -> KernelSchema:
        """Build kernel schema from ONNX node.

        Polymorphic method that handles both static and dynamic schemas:
        - Static schemas: return constant, ignore parameters
        - Dynamic schemas: inspect node structure to build schema

        Called in two contexts:
        1. During __init__: model=None (schema built for instance)
        2. During can_infer_from(): model provided (schema built for validation)

        Args:
            node: ONNX node (provides inputs, outputs, attributes)
            model: Optional ModelWrapper (provides shapes, datatypes for validation context)

        Returns:
            KernelSchema defining kernel structure

        Example (static schema):
            @classmethod
            def build_schema(cls, node, model):
                return LAYERNORM_SCHEMA

        Example (dynamic schema):
            @classmethod
            def build_schema(cls, node, model):
                num_inputs = len(node.input)
                inputs = [InputSchema(name=f"input{i}", ...) for i in range(num_inputs)]
                return KernelSchema(name="Concat", inputs=inputs, outputs=[...])
        """
        raise NotImplementedError(f"{cls.__name__}.build_schema()")

    # ====================================================================
    # Inference Support (Unified System)
    # ====================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this kernel can transform the given ONNX node (default: no)."""
        return False

    @classmethod
    def infer_from(cls, node: NodeProto, model: ModelWrapper, insert_index: int) -> TransformationResult:
        """Transform ONNX node to hardware kernel node(s)."""
        raise NotImplementedError(f"{cls.__name__}.infer_from()")

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def get_nodeattr_types(self):
        """Return nodeattr registry (datatypes + user params + kernel params).

        Auto-delegates to kernel_schema.build_nodeattr_registry() which includes:
        - Interface datatypes (input0Datatype, output0Datatype, etc.)
        - Internal datatypes (accumulatorDatatype, etc.)
        - Template parameters (SIMD, PE, etc.)
        - Kernel-specific parameters (epsilon, algorithm, etc.)

        Only override if build_schema() needs to read nodeattrs (circular dependency).
        In that case, define nodeattrs explicitly before calling build_schema().
        """
        base_attrs = super().get_nodeattr_types()

        # Add implementation attribute for backend selection
        # Replaces domain mutation used by FINN's SpecializeLayers
        base_attrs.update({
            "implementation": ("s", False, "", {
                "",              # Not yet specialized
                "vitis_hls",     # Vitis HLS (2020.1+)
                "verilog",       # Verilog RTL
                "systemverilog", # SystemVerilog RTL
                "static_ip",     # Pre-generated IP core
            }),
        })

        try:
            base_attrs.update(self.kernel_schema.build_nodeattr_registry())
        except RecursionError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.kernel_schema property calls get_nodeattr(), "
                f"creating circular dependency. You must override get_nodeattr_types() "
                f"explicitly to define nodeattrs before schema construction. "
                f"See KernelOp docstring for mode-dependent schema pattern."
            ) from e

        return base_attrs

    # ====================================================================
    # Public API: Cached State Access (Properties)
    # ====================================================================

    @property
    def design_space(self) -> KernelDesignSpace:
        """Cached design space (call method with model_w first to initialize)."""
        if self._design_space is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: Not initialized. "
                f"Call a method with model_w parameter first."
            )
        return self._design_space

    @property
    def design_point(self) -> KernelDesignPoint:
        """Cached kernel instance (call method with model_w first to initialize)."""
        if self._design_point is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: Not initialized. "
                f"Call a method with model_w parameter first."
            )
        return self._design_point

    # ====================================================================
    # Public API: Cache Management
    # ====================================================================

    def invalidate(self) -> None:
        """Invalidate cached state after external graph changes.

        Call this after transforms that change:
        - Tensor shapes (padding, reshape)
        - Datatypes in graph metadata
        - Node rewiring (FIFO insertion)

        Next method call with model_w will rebuild automatically.

        Example:
            >>> # After transform changes graph
            >>> model = ApplyPadding().apply(model)
            >>> for node in model.graph.node:
            ...     op = getCustomOp(node)
            ...     if isinstance(op, KernelOp):
            ...         op.invalidate()
        """
        self._design_space = None
        self._design_point = None

    # ====================================================================
    # Private API: Lazy Initialization
    # ====================================================================

    def _ensure_ready(self, model_w: ModelWrapper) -> None:
        """Ensure kernel is ready: initialize design space, sync instance with params.

        Idempotent two-phase operation:
        1. Build design space if not cached (expensive, once per structure)
        2. Reconfigure instance if params changed (fast, as needed)

        Args:
            model_w: ModelWrapper for ONNX graph access

        Raises:
            KernelOpError: If construction or validation fails
        """
        if model_w is None:
            raise self._error("ModelWrapper required")

        # Phase 1: Build design space if not cached
        if self._design_space is None:
            build_ctx = BuildContext(
                schema=self.kernel_schema,
                model_w=model_w,
                node_inputs=list(self.onnx_node.input),
                node_outputs=list(self.onnx_node.output),
                param_getter=self.get_nodeattr,
                param_setter=self.set_nodeattr,
                node_name=self.onnx_node.name
            )

            try:
                self._design_space = DesignSpaceBuilder().build(build_ctx)
                logger.debug(f"{self.onnx_node.name}: Built design space")
            except ValueError as e:
                raise self._error(str(e))

            # Auto-populate missing dimension parameters from auto-computed defaults
            from qonnx.util.basic import get_by_name
            for dim_name, dim in self._design_space.dimensions.items():
                # Check if attribute is actually set in ONNX node (not just has a default)
                if get_by_name(self.onnx_node.attribute, dim_name) is None:
                    # Dimension not set - use default/minimum value
                    if isinstance(dim, OrderedDimension):
                        # OrderedDimension: use get_default() (explicit default or minimum)
                        initial_value = dim.get_default()
                    else:  # frozenset
                        # Discrete: use sorted first value
                        initial_value = sorted(dim)[0]

                    self.set_nodeattr(dim_name, initial_value)
                    logger.debug(
                        f"{self.onnx_node.name}: Auto-populated {dim_name}={initial_value}"
                    )

        # Phase 2: Configure instance if not cached
        if self._design_point is None:
            current_config = {
                dim_name: self.get_nodeattr(dim_name)
                for dim_name in self._design_space.dimensions.keys()
            }
            try:
                self._design_point = self._design_space.configure(current_config)
                logger.debug(
                    f"{self.onnx_node.name}: Configured with {current_config}"
                )
            except ValueError as e:
                raise self._error(str(e))

    def build_design_space(self, model_w: ModelWrapper) -> None:
        """FINN API compatibility: Build design space.

        This method provides compatibility with FINN's getHWCustomOp() utility,
        which detects KernelOp via kernel_schema attribute and calls this method.

        Args:
            model_w: ModelWrapper for graph context
        """
        # Delegate to existing lazy initialization
        self._ensure_ready(model_w)

    def get_valid_ranges(self, model_w: ModelWrapper) -> Dict[str, Union[OrderedDimension, FrozenSet]]:
        """Valid dimension values for DSE (tiling + resource).

        Returns:
            Dict mapping dimension names to OrderedDimension (ordered sequences)
            or frozenset (discrete categories).
        """
        self._ensure_ready(model_w)
        return self.design_space.dimensions

    def infer_node_datatype(self, model_w):
        """Sync datatypes: model → nodeattrs (inputs), nodeattrs → model (outputs).

        Initializes design space which syncs input datatypes from model to nodeattrs.
        Then propagates output datatypes from nodeattrs back to model.
        """
        # Initialize (syncs inputs: model → nodeattrs)
        self._ensure_ready(model_w)

        # Propagate output datatypes: nodeattrs → model
        for i, out_name in enumerate(self.onnx_node.output):
            if out_name:
                odt = self.get_output_datatype(i)
                model_w.set_tensor_datatype(out_name, odt)

    # ====================================================================
    # Public API: Shape/Datatype Queries
    # ====================================================================

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype."""
        return DataType[self.get_nodeattr(f"input{ind}Datatype")]

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype."""
        return DataType[self.get_nodeattr(f"output{ind}Datatype")]

    # FINN API compatibility - delegate to design_point
    def get_normal_input_shape(self, ind=0) -> Tuple[int, ...]:
        """Return normal (unfolded) input shape as immutable tuple (FINN convention)."""
        return self.design_point.input_list[ind].tensor_shape

    def get_normal_output_shape(self, ind=0) -> Tuple[int, ...]:
        """Return normal (unfolded) output shape as immutable tuple (FINN convention)."""
        return self.design_point.output_list[ind].tensor_shape

    def get_folded_input_shape(self, ind=0) -> Tuple[int, ...]:
        tensor_shape = self.design_point.input_list[ind].tensor_shape
        stream_shape = self.design_point.input_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_folded_output_shape(self, ind=0) -> Tuple[int, ...]:
        tensor_shape = self.design_point.output_list[ind].tensor_shape
        stream_shape = self.design_point.output_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_instream_width(self, ind=0) -> int:
        return self.design_point.input_list[ind].stream_width_bits

    def get_outstream_width(self, ind=0) -> int:
        return self.design_point.output_list[ind].stream_width_bits

    def get_number_output_values(self):
        """Get iteration count(s) for output values.

        Matches FINN API pattern:
        - Single-output kernels: Returns int (iteration count)
        - Multi-output kernels: Returns dict mapping output names → iteration counts

        Returns:
            int: For single-output kernels (e.g., MVAU, Thresholding, AddStreams)
            dict: For multi-output kernels (e.g., DuplicateStreams, Split)

        Examples:
            Single-output: 512
            Multi-output: {'out0': 512, 'out1': 512}
        """
        num_outputs = len(self.onnx_node.output)

        if num_outputs == 1:
            # Single-output: Return int (FINN pattern for MVAU, Thresholding, etc.)
            folded_shape = self.get_folded_output_shape(ind=0)
            return math.prod(folded_shape[:-1])
        else:
            # Multi-output: Return dict (FINN pattern for DuplicateStreams, Split)
            out_val = {}
            for i in range(num_outputs):
                folded_shape = self.get_folded_output_shape(ind=i)
                iteration_count = math.prod(folded_shape[:-1])
                out_val[f"out{i}"] = iteration_count
            return out_val

    def get_exp_cycles(self) -> int:
        return self.design_point.initiation_interval

    def make_shape_compatible_op(self, model_w):
        """Create standard ONNX op for shape inference (auto-detects pattern)."""
        from onnx import helper

        num_out = len(self.onnx_node.output)
        num_in = len(self.onnx_node.input)

        if num_in == 1 and num_out > 1:
            return helper.make_node(
                "Split",
                inputs=[self.onnx_node.input[0]],
                outputs=list(self.onnx_node.output),
                axis=-1
            )

        if num_out == 1:
            input_shapes = [tuple(model_w.get_tensor_shape(inp))
                           for inp in self.onnx_node.input]

            if len(set(input_shapes)) == 1:
                return super().make_const_shape_op(input_shapes[0])
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: {num_in} inputs with different shapes "
                    f"{input_shapes}. Override make_shape_compatible_op()."
                )

        raise NotImplementedError(
            f"{self.__class__.__name__}: {num_in} inputs → {num_out} outputs. "
            f"Override make_shape_compatible_op()."
        )

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set nodeattr and auto-invalidate affected caches."""
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            old_value = None

        if old_value != value:
            super().set_nodeattr(name, value)

            # Only invalidate if attribute affects design space or design point
            if name in self.kernel_schema.get_structural_nodeattrs():
                # Structural changes (datatypes) invalidate everything
                self._design_space = self._design_point = None
            elif self._design_space is not None and name in self._design_space.dimensions:
                # Dimension changes (PE, SIMD, etc.) only invalidate design_point
                self._design_point = None
            # else: Runtime attributes (exec_mode, code_gen_dir_*) don't invalidate
