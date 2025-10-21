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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from onnx import NodeProto

from .schemas import KernelSchema
from .models import KernelDesignSpace, KernelInstance
from .builder import BuildContext, build_kernel_design_space
from .validation import KernelValidationContext
from .transformation import TransformationResult, transform_onnx_to_kernel

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Kernel operator exception with node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class (shapes extracted from ModelWrapper, never stored).

    Subclasses must implement build_schema() to construct their KernelSchema.

    Two common usage patterns:

    1. Static schema (most common - LayerNorm, Softmax, AddStreams):
        ```python
        SCHEMA = df.KernelSchema(name="LayerNorm", ...)

        @classmethod
        def build_schema(cls, node, model):
            return SCHEMA  # Constant, ignores parameters
        ```

    2. Dynamic schema (variable I/O - Concat, MVAU):
        ```python
        @classmethod
        def build_schema(cls, node, model):
            # Inspect node structure to determine schema
            num_inputs = len(node.input)
            inputs = [InputSchema(name=f"input{i}", ...) for i in range(num_inputs)]
            return KernelSchema(name="Concat", inputs=inputs, outputs=[...])
        ```

    The schema is built once during __init__ and frozen as an attribute.
    During inference validation, build_schema() is called with full ModelWrapper context.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Build and freeze schema from node structure
        self.kernel_schema = self.build_schema(onnx_node, model=None)

        # Two-phase caching system for DSE performance
        self._design_space: Optional['KernelDesignSpace'] = None  # Built once, never invalidated
        self._configuration: Optional['KernelInstance'] = None  # Rebuilt on param change
        self._current_params: Optional[dict] = None  # Tracks current parallelization params

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
        raise NotImplementedError(
            f"{cls.__name__} must implement build_schema() as a classmethod "
            f"returning a KernelSchema. See KernelOp docstring for examples."
        )

    # ====================================================================
    # Inference Support (Unified System)
    # ====================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this ONNX node can be converted to this hardware kernel.

        Uses schema.can_transform() for unified validation:
        1. Op type matching (from schema.source_ops)
        2. Schema constraints validation (declarative)

        Pure boolean check - no side effects.

        Args:
            node: ONNX node to validate
            model: ModelWrapper for graph context

        Returns:
            True if this kernel can convert the node
        """
        schema = cls.build_schema(node, model)
        return schema.can_transform(node, model)

    @classmethod
    def infer_from(
        cls,
        node: NodeProto,
        model: ModelWrapper,
        insert_index: int
    ) -> TransformationResult:
        """Create HW node(s) from ONNX node using unified schema.

        Default implementation uses pure transform_onnx_to_kernel() function.
        Override for complex transformations (fusion, multi-node, etc.).

        Args:
            node: ONNX node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes

        Returns:
            TransformationResult with created nodes

        Raises:
            ValueError: If transformation fails

        Example (default - most kernels):
            # Don't override - default handles it!

        Example (complex transformation):
            @classmethod
            def infer_from(cls, node, model, insert_index):
                from brainsmith.dataflow.inference import TransformationHelper

                schema = cls.build_schema(node, model)
                helper = TransformationHelper(model, domain=schema.domain)

                # Custom transformation logic
                hw_node = helper.make_node(...)

                return TransformationResult(
                    nodes_to_insert=[hw_node],
                    nodes_to_remove=[node],
                    actual_layouts={...},
                )
        """
        schema = cls.build_schema(node, model)
        return transform_onnx_to_kernel(
            schema=schema,
            node=node,
            model=model,
            insert_index=insert_index,
            kernel_class_name=cls.__name__
        )

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def get_nodeattr_types(self):
        """Return nodeattr registry (datatypes + user params + kernel params).

        Auto-delegates to kernel_schema.get_nodeattr_types() which includes:
        - Interface datatypes (input0Datatype, output0Datatype, etc.)
        - Internal datatypes (accumulatorDatatype, etc.)
        - Template parameters (SIMD, PE, etc.)
        - Kernel-specific parameters (epsilon, algorithm, etc.)

        Only override if build_schema() needs to read nodeattrs (circular dependency).
        In that case, define nodeattrs explicitly before calling build_schema().
        """
        my_attrs = super().get_nodeattr_types()

        try:
            my_attrs.update(self.kernel_schema.get_nodeattr_types())
        except RecursionError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.kernel_schema property calls get_nodeattr(), "
                f"creating circular dependency. You must override get_nodeattr_types() "
                f"explicitly to define nodeattrs before schema construction. "
                f"See KernelOp docstring for mode-dependent schema pattern."
            ) from e

        return my_attrs

    # ====================================================================
    # Public API: Cached State Access (Properties)
    # ====================================================================

    @property
    def design_space(self) -> KernelDesignSpace:
        """Access cached design space.

        Returns the cached KernelDesignSpace instance. This is a read-only
        property that accesses the cached state built by build_design_space().

        Returns:
            Cached KernelDesignSpace instance

        Raises:
            RuntimeError: If design space not built yet.
                         Call build_design_space(model_w) first.

        Example:
            >>> kernel_op.build_design_space(model_w)
            >>> valid_ranges = kernel_op.design_space.parallelization_params
            >>> print(f"SIMD range: {valid_ranges['SIMD']}")
        """
        if self._design_space is None:
            raise RuntimeError(
                f"Design space not built for {self.onnx_node.name}. "
                f"Call build_design_space(model_w) first to build the design space."
            )
        return self._design_space

    @property
    def kernel_instance(self) -> KernelInstance:
        """Access cached kernel instance.

        Returns the cached KernelInstance with current parallelization
        parameters. This is a read-only property that accesses the cached
        state built by build_design_space().

        Returns:
            Cached KernelInstance

        Raises:
            RuntimeError: If instance not built yet.
                         Call build_design_space(model_w) first.

        Example:
            >>> kernel_op.build_design_space(model_w)
            >>> cycles = kernel_op.kernel_instance.initiation_interval
            >>> stream_width = kernel_op.kernel_instance.output_stream_width_bits()
        """
        if self._configuration is None:
            raise RuntimeError(
                f"Kernel instance not built for {self.onnx_node.name}. "
                f"Call build_design_space(model_w) first to build the kernel instance."
            )
        return self._configuration

    # ====================================================================
    # Public API: Design Space Construction
    # ====================================================================

    def get_design_space(self, model_w: ModelWrapper) -> KernelDesignSpace:
        """Get or build kernel design space for DSE.

        Built once per kernel instance, cached for lifetime. Contains all properties
        constant across parallelization configs: tensor shapes, block shapes, datatypes,
        and valid parallelization ranges.

        This method enables efficient DSE by separating expensive design space
        construction (done once) from cheap configuration selection (done many times).

        Args:
            model_w: ModelWrapper for ONNX graph access

        Returns:
            KernelDesignSpace ready for configuration exploration

        Raises:
            KernelOpError: If design space cannot be built or validation fails
        """
        # Return cached design space if available
        if self._design_space is not None:
            return self._design_space

        if model_w is None:
            raise self._error(
                "ModelWrapper (model_w) required to build KernelDesignSpace. "
                "KernelOp needs ModelWrapper to extract tensor shapes from the ONNX graph."
            )

        # Build context for builder
        build_ctx = BuildContext(
            schema=self.kernel_schema,
            model_w=model_w,
            node_inputs=list(self.onnx_node.input),
            node_outputs=list(self.onnx_node.output),
            param_getter=self.get_nodeattr,
            param_setter=self.set_nodeattr,
            node_name=self.onnx_node.name
        )

        # Build design space using module-level function
        # Creates temporary builder, uses it, discards it
        try:
            self._design_space = build_kernel_design_space(build_ctx)
        except ValueError as e:
            raise self._error(str(e))

        return self._design_space

    def get_kernel_instance(self, model_w: ModelWrapper) -> KernelInstance:
        """Get current kernel instance (Phase 2 of two-phase construction).

        Reconfigures if parallelization parameters changed. Fast path (<0.1ms) if
        params unchanged, slow path (<1ms) if reconfiguration needed.

        This method returns KernelInstance (the configured instance from two-phase
        construction).

        Args:
            model_w: ModelWrapper for ONNX graph access

        Returns:
            KernelInstance with current stream shapes

        Raises:
            KernelOpError: If instance cannot be built or validation fails
        """
        # Get design space (cached after first call)
        design_space = self.get_design_space(model_w)

        # Extract current parallelization params from nodeattrs
        current_params = {}
        for param_name in design_space.parallelization_params.keys():
            current_params[param_name] = self.get_nodeattr(param_name)

        # Check if reconfiguration needed
        if self._configuration is None or self._current_params != current_params:
            # Configure with current params (fast: <1ms)
            try:
                self._configuration = design_space.configure(current_params)
                self._current_params = current_params.copy()
            except ValueError as e:
                raise self._error(str(e))

        return self._configuration

    def build_design_space(self, model_w: ModelWrapper) -> None:
        """Build or rebuild design space and configuration from graph.

        This method is idempotent and efficient:
        - If design space invalidated: Rebuilds it (~10ms)
        - If design space valid: Skips rebuild
        - If params changed: Reconfigures (~1ms)
        - If params unchanged: Skips reconfigure

        Call this method:
        - Before first access to design_space/kernel_instance properties
        - After graph structure changes (shapes, datatypes, rewiring)
        - After changing parallelization parameters
        - In transforms that need up-to-date models

        Args:
            model_w: ModelWrapper for ONNX graph access

        Raises:
            KernelOpError: If design space cannot be built or validation fails

        Example:
            >>> # Initial build
            >>> kernel_op.build_design_space(model_w)
            >>> print(kernel_op.design_space.name)

            >>> # After param change
            >>> kernel_op.set_nodeattr("SIMD", 64)
            >>> kernel_op.build_design_space(model_w)  # Reconfigures
            >>> print(kernel_op.kernel_instance.params["SIMD"])  # 64

            >>> # After graph change
            >>> model_w.set_tensor_shape(node.input[0], (1, 1, 1024))
            >>> kernel_op.invalidate_design_space()
            >>> kernel_op.build_design_space(model_w)  # Rebuilds
        """
        if model_w is None:
            raise self._error(
                "ModelWrapper (model_w) required to build design space. "
                "KernelOp needs ModelWrapper to extract tensor shapes from the ONNX graph."
            )

        # Phase 1: Build design space if needed
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
                self._design_space = build_kernel_design_space(build_ctx)
                logger.debug(f"{self.onnx_node.name}: Built design space")
            except ValueError as e:
                raise self._error(str(e))

        # Phase 2: Configure if needed
        current_params = {
            param_name: self.get_nodeattr(param_name)
            for param_name in self._design_space.parallelization_params.keys()
        }

        if self._configuration is None or self._current_params != current_params:
            try:
                self._configuration = self._design_space.configure(current_params)
                self._current_params = current_params.copy()
                logger.debug(
                    f"{self.onnx_node.name}: Configured with {current_params}"
                )
            except ValueError as e:
                raise self._error(str(e))

    def invalidate_design_space(self) -> None:
        """Explicitly invalidate design space cache.

        Call this after external graph changes that aren't captured by
        set_nodeattr():
        - Tensor shapes changed (padding, reshape, etc.)
        - Datatypes changed in graph metadata
        - Node rewiring (FIFO insertion, etc.)

        After invalidation, call build_design_space(model_w) to rebuild.

        Example:
            >>> # After transform changes graph
            >>> model = ApplyPadding().apply(model)
            >>> for node in model.graph.node:
            ...     op = getCustomOp(node)
            ...     if isinstance(op, KernelOp):
            ...         op.invalidate_design_space()
            ...         op.build_design_space(model)
        """
        self._invalidate_design_space()

    def _invalidate_design_space(self) -> None:
        """Internal: Invalidate design space and all dependent caches."""
        self._design_space = None
        self._configuration = None
        self._current_params = None

    def _invalidate_configuration(self) -> None:
        """Internal: Invalidate configuration cache only."""
        self._configuration = None
        self._current_params = None

    def get_valid_ranges(self, model_w: ModelWrapper) -> Dict[str, set]:
        """Get valid parallelization parameter ranges for DSE.

        New API for DSE system to query valid configurations. Returns pre-computed
        divisor sets for each parallelization parameter (e.g., SIMD, PE, MW, MH).

        This enables discrete, exhaustive design space exploration without trial-and-error
        attempts at invalid configurations.

        Args:
            model_w: ModelWrapper for ONNX graph access

        Returns:
            Dict mapping parameter names to valid divisor sets
            Example: {"SIMD": {1, 2, 3, 4, 6, 8, ..., 768}, "PE": {1, 2, 4, 8}}

        Example Usage:
            >>> valid_simd = kernel_op.get_valid_ranges(model_w)["SIMD"]
            >>> for simd in valid_simd:
            ...     kernel_op.set_nodeattr("SIMD", simd)
            ...     kernel_op.build_design_space(model_w)  # Fast reconfiguration
        """
        self.build_design_space(model_w)
        return self.design_space.parallelization_params

    def infer_node_datatype(self, model_w):
        """FINN compatibility wrapper.

        Builds design space if needed to ensure datatypes are set correctly.
        """
        self.build_design_space(model_w)

    # ====================================================================
    # Public API: Shape/Datatype Queries
    # ====================================================================

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype."""
        return DataType[self.get_nodeattr(f"input{ind}Datatype")]

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype."""
        return DataType[self.get_nodeattr(f"output{ind}Datatype")]

    def get_normal_input_shape(self, ind=0, model_w: Optional[ModelWrapper] = None) -> List[int]:
        """Get input tensor shape.

        Args:
            ind: Input index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Tensor shape as list
        """
        if model_w:
            self.build_design_space(model_w)
        return list(self.kernel_instance.input_list[ind].tensor_shape)

    def get_normal_output_shape(self, ind=0, model_w: Optional[ModelWrapper] = None) -> List[int]:
        """Get output tensor shape.

        Args:
            ind: Output index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Tensor shape as list
        """
        if model_w:
            self.build_design_space(model_w)
        return list(self.kernel_instance.output_list[ind].tensor_shape)

    def get_folded_input_shape(self, ind=0, model_w: Optional[ModelWrapper] = None) -> Tuple[int, ...]:
        """Get FINN folded input shape (fold_factors + flattened_stream).

        Args:
            ind: Input index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Folded shape tuple
        """
        if model_w:
            self.build_design_space(model_w)
        tensor_shape = self.kernel_instance.input_list[ind].tensor_shape
        stream_shape = self.kernel_instance.input_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_folded_output_shape(self, ind=0, model_w: Optional[ModelWrapper] = None) -> Tuple[int, ...]:
        """Get FINN folded output shape (fold_factors + flattened_stream).

        Args:
            ind: Output index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Folded shape tuple
        """
        if model_w:
            self.build_design_space(model_w)
        tensor_shape = self.kernel_instance.output_list[ind].tensor_shape
        stream_shape = self.kernel_instance.output_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_instream_width(self, ind=0, model_w: Optional[ModelWrapper] = None) -> int:
        """Get input stream width in bits.

        Args:
            ind: Input index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Stream width in bits
        """
        if model_w:
            self.build_design_space(model_w)
        return self.kernel_instance.input_list[ind].stream_width_bits

    def get_outstream_width(self, ind=0, model_w: Optional[ModelWrapper] = None) -> int:
        """Get output stream width in bits.

        Args:
            ind: Output index
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Stream width in bits
        """
        if model_w:
            self.build_design_space(model_w)
        return self.kernel_instance.output_list[ind].stream_width_bits

    def get_number_output_values(self, model_w: Optional[ModelWrapper] = None):
        """Get number of time-multiplexed output values.

        Args:
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Number of output values
        """
        if model_w:
            self.build_design_space(model_w)
        folded_shape = self.get_folded_output_shape(ind=0)
        return math.prod(folded_shape[:-1])

    def get_exp_cycles(self, model_w: Optional[ModelWrapper] = None):
        """Get expected cycles (initiation interval).

        Args:
            model_w: Optional ModelWrapper. If provided, builds design space first.

        Returns:
            Expected cycles
        """
        if model_w:
            self.build_design_space(model_w)
        return self.kernel_instance.initiation_interval

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
        """Set nodeattr with schema-based eager cache invalidation.

        Invalidation strategy based on nodeattr classification:
        - Structural attrs (datatypes): Invalidate design space + config
        - Parametric attrs (SIMD, PE): Invalidate config only
        - Execution attrs (epsilon): Invalidate config (conservative)

        Args:
            name: Attribute name
            value: New value
        """
        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            pass

        if old_value != value:
            super().set_nodeattr(name, value)

            # Schema-based classification for smart invalidation
            if hasattr(self, 'kernel_schema') and self.kernel_schema:
                if name in self.kernel_schema.get_structural_nodeattrs():
                    # Structural change: invalidate everything
                    logger.debug(
                        f"{self.onnx_node.name}: Structural nodeattr change ({name}), "
                        f"invalidating design space"
                    )
                    self._invalidate_design_space()
                elif name in self.kernel_schema.get_parametric_nodeattrs():
                    # Parametric change: invalidate config only
                    logger.debug(
                        f"{self.onnx_node.name}: Parametric nodeattr change ({name}), "
                        f"invalidating configuration"
                    )
                    self._invalidate_configuration()
                else:
                    # Execution parameter: conservative, invalidate config
                    logger.debug(
                        f"{self.onnx_node.name}: Execution nodeattr change ({name}), "
                        f"invalidating configuration"
                    )
                    self._invalidate_configuration()
            else:
                # Schema not ready: conservative, invalidate config only
                self._invalidate_configuration()
