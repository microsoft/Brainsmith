############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for hardware custom operators automated with Dataflow Modeling.

State Management:
- Nodeattrs: Single source of truth (shapes, datatypes, params)
- Metadata: Optional KernelModel cache for performance
- Cache invalidated on refresh_df_model() or set_nodeattr()
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.dataflow.datatype_sources import DatatypeSource
from brainsmith.dataflow import (
    KernelSchema,
    InputModel,
    OutputModel,
    KernelModel,
    resolve_template
)

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Exception raised by kernel operators with automatic node context."""
    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.

    Subclasses must:
    - Define kernel_schema class attribute with KernelSchema instance
    - Implement abstract methods from HWCustomOp base class
    """

    kernel_schema: KernelSchema  # type: ignore[assignment]  # Subclasses must define

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )

        # Single cache: KernelModel (lazy-built from nodeattrs)
        self._kernel_model: Optional[KernelModel] = None

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def refresh_df_model(self, model: ModelWrapper) -> None:
        """Update all node state from ModelWrapper.

        This is the ONLY method that updates protected nodeattrs.
        Called by FINN transforms after graph modifications.

        Updates:
        1. Tensor shapes from graph → _input*TensorShape, _output*TensorShape
        2. Datatypes from graph → _input*Datatype, _output*Datatype
        3. Builds KernelModel to compute derived shapes
        4. Stores derived shapes → _input*BlockShape, _input*StreamShape
        5. Resolves internal datatypes → _*Datatype
        6. Invalidates cached KernelModel

        Args:
            model: ModelWrapper to extract tensor information from
        """
        logger.debug(f"Refreshing state for {self.onnx_node.name}")

        # ============================================================
        # Step 1: Extract and store tensor shapes
        # ============================================================

        for i, inp_name in enumerate(self.onnx_node.input):
            if inp_name:  # Skip optional inputs
                shape = model.get_tensor_shape(inp_name)
                self._set_protected(f"_input{i}TensorShape", list(shape))

        for i, out_name in enumerate(self.onnx_node.output):
            shape = model.get_tensor_shape(out_name)
            self._set_protected(f"_output{i}TensorShape", list(shape))

        # ============================================================
        # Step 2: Extract and store datatypes
        # ============================================================

        for i, inp_name in enumerate(self.onnx_node.input):
            if inp_name:
                dtype = model.get_tensor_datatype(inp_name)
                self._set_protected(f"_input{i}Datatype", dtype.name)

        for i, out_name in enumerate(self.onnx_node.output):
            dtype = model.get_tensor_datatype(out_name)
            self._set_protected(f"_output{i}Datatype", dtype.name)

        # ============================================================
        # Step 3: Build KernelModel to compute derived values
        # ============================================================

        # Build from graph shapes
        kernel_model = self._build_from_graph_shapes(model)

        # ============================================================
        # Step 4: Store derived hardware shapes
        # ============================================================

        for i, inp_model in enumerate(kernel_model.inputs):
            self._set_protected(f"_input{i}BlockShape", list(inp_model.block_shape))
            self._set_protected(f"_input{i}StreamShape", list(inp_model.stream_shape))

        for i, out_model in enumerate(kernel_model.outputs):
            self._set_protected(f"_output{i}BlockShape", list(out_model.block_shape))
            self._set_protected(f"_output{i}StreamShape", list(out_model.stream_shape))

        # ============================================================
        # Step 5: Store resolved internal datatypes
        # ============================================================

        # Internal datatypes were already resolved during _build_from_graph_shapes()
        # and stored in nodeattrs by _resolve_internal_datatypes()

        # ============================================================
        # Step 6: Cache complete KernelModel (optional optimization)
        # ============================================================

        self._kernel_model = kernel_model
        self._cache_kernel_model_to_metadata(kernel_model)

        logger.debug(f"State refresh complete for {self.onnx_node.name}")

    @property
    def kernel_model(self) -> KernelModel:
        """Lazy-built KernelModel with two-tier reconstruction.

        Reconstruction strategy:
        1. Return cached model if available
        2. Try deserializing from metadata cache
        3. Rebuild from nodeattrs (source of truth)
        4. Fail with helpful error

        Returns:
            Immutable KernelModel instance

        Raises:
            RuntimeError: If node not initialized (no nodeattrs set)
        """
        if self._kernel_model is not None:
            return self._kernel_model

        # ============================================================
        # Tier 1: Try deserializing from metadata cache (fast)
        # ============================================================

        try:
            model_json = self._read_kernel_model_from_metadata()
            if model_json:
                logger.debug(
                    f"Reconstructing {self.onnx_node.name} from cached KernelModel"
                )
                self._kernel_model = KernelModel.from_json(model_json)
                return self._kernel_model
        except Exception as e:
            logger.debug(f"Could not deserialize KernelModel: {e}")

        # ============================================================
        # Tier 2: Rebuild from nodeattrs (source of truth)
        # ============================================================

        try:
            logger.debug(f"Rebuilding {self.onnx_node.name} from nodeattrs")
            self._kernel_model = self._build_kernel_model_from_nodeattrs()

            # Cache for next time
            self._cache_kernel_model_to_metadata(self._kernel_model)

            return self._kernel_model
        except Exception as e:
            logger.debug(f"Could not rebuild from nodeattrs: {e}")
            raise RuntimeError(
                f"Cannot build KernelModel for {self.onnx_node.name}. "
                f"Node not initialized. Call refresh_df_model(model) first.\n"
                f"Error: {e}"
            )

    def get_nodeattr_types(self):
        """Get node attribute types registry.

        Combines attributes from:
        1. FINN base class (HWCustomOp infrastructure attrs)
        2. KernelSchema-derived attrs (datatypes + template params)

        Subclasses can override this to add kernel-specific attributes:
            def get_nodeattr_types(self):
                my_attrs = super().get_nodeattr_types()
                my_attrs.update({
                    "epsilon": ("f", True, 1e-5),  # Example kernel-specific attribute
                })
                return my_attrs

        Returns:
            dict: Mapping of attribute name to (type, required, default_value)
        """
        # Merge FINN infrastructure and schema-derived attributes (schema overrides base)
        return {**super().get_nodeattr_types(), **self.kernel_schema.get_nodeattr_types()}

    def infer_node_datatype(self, model):
        """FINN compatibility wrapper. Modern code should use refresh_df_model()."""
        self.refresh_df_model(model)

    # ====================================================================
    # Public API: Shape/Datatype Queries (HWCustomOp Interface)
    # ====================================================================

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype from nodeattr or KernelModel."""
        try:
            # Fast path: read from nodeattr
            return DataType[self.get_nodeattr(f"_input{ind}Datatype")]
        except (AttributeError, KeyError):
            # Fallback: from cached model
            return self.kernel_model.input_datatype(ind)

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype from nodeattr or KernelModel."""
        try:
            return DataType[self.get_nodeattr(f"_output{ind}Datatype")]
        except (AttributeError, KeyError):
            return self.kernel_model.output_datatype(ind)

    def get_normal_input_shape(self, ind=0) -> List[int]:
        """Get input tensor shape from nodeattr or KernelModel."""
        try:
            # Fast path: read from nodeattr
            return self.get_nodeattr(f"_input{ind}TensorShape")
        except (AttributeError, KeyError):
            # Fallback: from cached model
            return list(self.kernel_model.input_tensor_shape(ind))

    def get_normal_output_shape(self, ind=0) -> List[int]:
        """Get output tensor shape from nodeattr or KernelModel."""
        try:
            return self.get_nodeattr(f"_output{ind}TensorShape")
        except (AttributeError, KeyError):
            return list(self.kernel_model.output_tensor_shape(ind))

    def get_folded_input_shape(self, ind=0) -> Tuple[int, ...]:
        """Get FINN-compatible folded input shape.

        Computed from tensor_shape and stream_shape.
        Format: [fold_factors..., flattened_stream]

        Example: tensor=[1, 128, 768], stream=[1, 1, 64]
            → fold_factors=[1, 128, 12], flattened=64
            → result=(1, 128, 12, 64)
        """
        try:
            # Read from nodeattrs
            tensor_shape = self.get_nodeattr(f"_input{ind}TensorShape")
            stream_shape = self.get_nodeattr(f"_input{ind}StreamShape")

            # Compute fold factors
            fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
            flattened_stream = math.prod(stream_shape)

            return tuple(fold_factors + [flattened_stream])
        except (AttributeError, KeyError):
            # Fallback: from cached model
            return self.kernel_model.get_folded_input_shape(ind)

    def get_folded_output_shape(self, ind=0) -> Tuple[int, ...]:
        """Get FINN-compatible folded output shape."""
        try:
            tensor_shape = self.get_nodeattr(f"_output{ind}TensorShape")
            stream_shape = self.get_nodeattr(f"_output{ind}StreamShape")

            fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
            flattened_stream = math.prod(stream_shape)

            return tuple(fold_factors + [flattened_stream])
        except (AttributeError, KeyError):
            return self.kernel_model.get_folded_output_shape(ind)

    def get_instream_width(self, ind=0) -> int:
        """Get input stream width in bits."""
        try:
            datatype = self.get_input_datatype(ind)
            stream_shape = self.get_nodeattr(f"_input{ind}StreamShape")
            streaming_bandwidth = math.prod(stream_shape)
            return streaming_bandwidth * datatype.bitwidth()
        except (AttributeError, KeyError):
            return self.kernel_model.input_stream_width_bits(ind)

    def get_outstream_width(self, ind=0) -> int:
        """Get output stream width in bits."""
        try:
            datatype = self.get_output_datatype(ind)
            stream_shape = self.get_nodeattr(f"_output{ind}StreamShape")
            streaming_bandwidth = math.prod(stream_shape)
            return streaming_bandwidth * datatype.bitwidth()
        except (AttributeError, KeyError):
            return self.kernel_model.output_stream_width_bits(ind)

    def get_number_output_values(self):
        """Get number of output values for FINN.

        FINN expects this to return the number of time-multiplexed output values,
        which is the product of the folded output shape excluding the parallelism
        dimension (i.e., folded_shape[:-1]).

        This represents the number of cycles/transfers needed to stream the output.

        Returns:
            Number of time-multiplexed output values
        """
        folded_shape = self.get_folded_output_shape()
        return math.prod(folded_shape[:-1])

    def get_exp_cycles(self):
        return self.kernel_model.initiation_interval

    def make_shape_compatible_op(self, model):
        """Create standard ONNX op for shape inference.

        Auto-detects appropriate substitute based on I/O pattern:
        - Single input, single output: RandomNormal with input shape (shape-preserving)
        - Single input, multiple outputs: Split on last axis
        - Multiple inputs (same shape), single output: RandomNormal with input shape
        - Different input shapes or multi-I/O: Requires override

        Override for:
        - Shape-transforming ops (output shape ≠ input shape)
        - Multi-input ops with different input shapes (Concat, MatMul, etc.)
        - Complex multi-input/multi-output patterns
        - Ops without inputs (constant generators)

        Args:
            model: ModelWrapper instance

        Returns:
            ONNX node for shape inference

        Raises:
            NotImplementedError: If pattern requires custom shape logic
        """
        from onnx import helper

        num_out = len(self.onnx_node.output)
        num_in = len(self.onnx_node.input)

        # Multiple outputs from single input: use Split
        if num_in == 1 and num_out > 1:
            return helper.make_node(
                "Split",
                inputs=[self.onnx_node.input[0]],
                outputs=list(self.onnx_node.output),
                axis=-1
            )

        # Single output: verify input shapes before using input[0]
        if num_out == 1:
            # Get all input shapes
            input_shapes = [tuple(model.get_tensor_shape(inp))
                           for inp in self.onnx_node.input]

            # Strict check: all inputs must have identical shape
            if len(set(input_shapes)) == 1:
                # All inputs same shape -> output likely same shape
                return super().make_const_shape_op(input_shapes[0])
            else:
                # Different input shapes -> ambiguous output shape
                raise NotImplementedError(
                    f"{self.__class__.__name__}: {num_in} inputs with different shapes "
                    f"{input_shapes}. Override make_shape_compatible_op() to compute "
                    f"output shape from inputs (e.g., Concat, MatMul, broadcasting ops)."
                )

        # Complex case: multi-input AND multi-output
        raise NotImplementedError(
            f"{self.__class__.__name__}: {num_in} inputs → {num_out} outputs. "
            f"Override make_shape_compatible_op() to specify a standard ONNX op "
            f"with matching shape inference behavior."
        )

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set attribute and invalidate cache.

        Overrides base class to:
        1. Prevent modification of protected attributes
        2. Invalidate KernelModel cache on user param changes

        Args:
            name: Attribute name
            value: New value

        Raises:
            KernelOpError: If attempting to modify protected attribute
        """
        # Enforce protection
        if name in self.kernel_schema.protected_attr_names:
            raise self._error(
                f"Cannot modify protected attribute '{name}'. "
                f"Protected attributes are managed by refresh_df_model()."
            )

        # Check if value actually changed
        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            pass

        if old_value != value:
            # Update value
            super().set_nodeattr(name, value)

            # Invalidate cache (user param changed)
            self._invalidate_cache()

    # ====================================================================
    # Internal Implementation
    # ====================================================================

    def _set_protected(self, name: str, value: Any) -> None:
        """Set protected attribute, bypassing protection check.

        Internal use only by refresh_df_model().
        """
        super().set_nodeattr(name, value)

    def _invalidate_cache(self) -> None:
        """Invalidate cached KernelModel.

        Called when user params change or refresh_df_model() runs.
        """
        self._kernel_model = None
        # Note: Metadata cache is stale but not deleted
        # Will be overwritten next time cache is built

    def _build_from_graph_shapes(self, model: ModelWrapper) -> KernelModel:
        """Build KernelModel from ModelWrapper (called by refresh_df_model).

        This method:
        1. Reads tensor shapes/datatypes from ModelWrapper
        2. Resolves dimensions using template_resolution
        3. Creates InputModel/OutputModel instances
        4. Resolves internal datatypes
        5. Returns complete KernelModel

        The caller (refresh_df_model) then extracts the computed shapes
        and stores them in nodeattrs.

        Args:
            model: ModelWrapper to read graph information from

        Returns:
            Complete KernelModel instance

        Raises:
            KernelOpError: If validation fails
        """
        logger.debug(f"Building KernelModel from graph for {self.onnx_node.name}")

        interfaces = {}

        # ================================================================
        # Build InputModels
        # ================================================================

        input_models = []
        for i, schema in enumerate(self.kernel_schema.inputs):
            inp_name = self.onnx_node.input[i]
            if not inp_name:
                continue  # Skip optional inputs

            # Get tensor shape from graph
            tensor_shape = tuple(model.get_tensor_shape(inp_name))

            # Resolve block shape from schema
            block_shape = self._resolve_dimensions(
                schema.block_tiling,
                tensor_shape,
                f"Input '{schema.name}' block",
                interfaces
            )

            # Resolve stream shape from schema
            stream_shape = self._resolve_dimensions(
                schema.stream_tiling,
                block_shape,
                f"Input '{schema.name}' stream",
                interfaces
            )

            # Get datatype from graph
            datatype = model.get_tensor_datatype(inp_name)

            input_model = InputModel(
                name=schema.name,
                tensor_shape=tensor_shape,
                block_shape=block_shape,
                stream_shape=stream_shape,
                datatype=datatype,
                is_weight=schema.is_weight
            )

            input_models.append(input_model)
            interfaces[schema.name] = input_model

        # ================================================================
        # Resolve Internal Datatypes
        # ================================================================

        if self.kernel_schema.internal_datatypes:
            internal_datatypes = self._resolve_internal_datatypes(interfaces)

            # Add to interfaces dict for output resolution
            for internal_name, datatype in internal_datatypes.items():
                class InternalDatatype:
                    def __init__(self, dt):
                        self.datatype = dt
                interfaces[internal_name] = InternalDatatype(datatype)

        # ================================================================
        # Build OutputModels
        # ================================================================

        output_models = []
        for i, schema in enumerate(self.kernel_schema.outputs):
            out_name = self.onnx_node.output[i]

            # Get tensor shape from graph
            tensor_shape = tuple(model.get_tensor_shape(out_name))

            # Resolve block shape
            block_shape = self._resolve_dimensions(
                schema.block_tiling,
                tensor_shape,
                f"Output '{schema.name}' block",
                interfaces
            )

            # Resolve stream shape (may be None if not specified)
            if schema.stream_tiling is not None:
                stream_shape = self._resolve_dimensions(
                    schema.stream_tiling,
                    block_shape,
                    f"Output '{schema.name}' stream",
                    interfaces
                )
            else:
                # Unset - will be resolved by KernelModel.__post_init__
                stream_shape = tuple([None] * len(block_shape))

            # Get datatype from graph
            datatype = model.get_tensor_datatype(out_name)

            # Validate against schema derivation if specified
            if isinstance(schema.datatype, DatatypeSource):
                expected_datatype = schema.datatype.resolve(interfaces, self.get_nodeattr)
                if datatype != expected_datatype:
                    raise self._error(
                        f"Output '{schema.name}' datatype mismatch: "
                        f"graph has {datatype.name}, schema expects {expected_datatype.name}"
                    )

            output_model = OutputModel(
                name=schema.name,
                tensor_shape=tensor_shape,
                block_shape=block_shape,
                stream_shape=stream_shape,
                datatype=datatype
            )

            output_models.append(output_model)
            interfaces[schema.name] = output_model

        # ================================================================
        # Create and validate KernelModel
        # ================================================================

        kernel_model = KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models)
        )

        self._validate_kernel_model(kernel_model)

        return kernel_model

    def _build_kernel_model_from_nodeattrs(self) -> KernelModel:
        """Reconstruct KernelModel from nodeattrs.

        This is the core reconstruction logic. Reads protected nodeattrs
        to rebuild InputModel/OutputModel instances.

        Returns:
            Fully resolved KernelModel

        Raises:
            RuntimeError: If required nodeattrs missing
            KernelOpError: If validation fails
        """
        logger.debug(
            f"Building KernelModel for {self.onnx_node.name} "
            f"({self.kernel_schema.name})"
        )

        # ============================================================
        # Build InputModels from nodeattrs
        # ============================================================

        input_models = []
        for i, schema in enumerate(self.kernel_schema.inputs):
            try:
                tensor_shape = tuple(self.get_nodeattr(f"_input{i}TensorShape"))
                block_shape = tuple(self.get_nodeattr(f"_input{i}BlockShape"))
                stream_shape = tuple(self.get_nodeattr(f"_input{i}StreamShape"))
                datatype = DataType[self.get_nodeattr(f"_input{i}Datatype")]

                input_model = InputModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_shape=stream_shape,
                    datatype=datatype,
                    is_weight=schema.is_weight
                )
                input_models.append(input_model)

                logger.debug(
                    f"  Input '{input_model.name}': "
                    f"tensor={tensor_shape}, block={block_shape}, "
                    f"stream={stream_shape}, dtype={datatype.name}"
                )
            except (AttributeError, KeyError) as e:
                raise RuntimeError(
                    f"Missing nodeattr for input {i} ({schema.name}): {e}"
                )

        # ============================================================
        # Build OutputModels from nodeattrs
        # ============================================================

        output_models = []
        for i, schema in enumerate(self.kernel_schema.outputs):
            try:
                tensor_shape = tuple(self.get_nodeattr(f"_output{i}TensorShape"))
                block_shape = tuple(self.get_nodeattr(f"_output{i}BlockShape"))
                stream_shape = tuple(self.get_nodeattr(f"_output{i}StreamShape"))
                datatype = DataType[self.get_nodeattr(f"_output{i}Datatype")]

                output_model = OutputModel(
                    name=schema.name,
                    tensor_shape=tensor_shape,
                    block_shape=block_shape,
                    stream_shape=stream_shape,
                    datatype=datatype
                )
                output_models.append(output_model)

                logger.debug(
                    f"  Output '{output_model.name}': "
                    f"tensor={tensor_shape}, block={block_shape}, "
                    f"stream={stream_shape}, dtype={datatype.name}"
                )
            except (AttributeError, KeyError) as e:
                raise RuntimeError(
                    f"Missing nodeattr for output {i} ({schema.name}): {e}"
                )

        # ============================================================
        # Create KernelModel
        # ============================================================

        model = KernelModel(
            name=self.kernel_schema.name,
            inputs=tuple(input_models),
            outputs=tuple(output_models)
        )

        # ============================================================
        # Validate constraints and relationships
        # ============================================================

        self._validate_kernel_model(model)

        logger.debug(f"KernelModel built successfully for {self.onnx_node.name}")
        return model

    def _validate_kernel_model(self, model: KernelModel) -> None:
        """Validate constraints and relationships on KernelModel.

        Args:
            model: KernelModel to validate

        Raises:
            KernelOpError: If validation fails
        """
        logger.debug(f"Validating KernelModel for {self.onnx_node.name}")

        # Validate input constraints
        for i, input_model in enumerate(model.inputs):
            schema = self.kernel_schema.inputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name, input_model, schema.constraints
                )

        # Validate output constraints
        for i, output_model in enumerate(model.outputs):
            schema = self.kernel_schema.outputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    schema.name, output_model, schema.constraints
                )

        # Validate cross-interface relationships
        if self.kernel_schema.relationships:
            logger.debug(
                f"Validating {len(self.kernel_schema.relationships)} relationships"
            )
            for relationship in self.kernel_schema.relationships:
                error = relationship.check(model, self.get_nodeattr)
                if error:
                    raise self._error(str(error))

    def _resolve_dimensions(
        self,
        template: Any,  # TilingSpec or similar sequence
        reference_shape: Tuple[int, ...],
        context_name: str,
        interfaces: dict
    ) -> Tuple[int, ...]:
        """Resolve template dimensions to concrete values.

        Args:
            template: Template specification (e.g., [":", "PE", DerivedDim("Q", 1)])
            reference_shape: Reference shape to resolve against
            context_name: Context string for error messages
            interfaces: Dict of already-built interface models for DerivedDim resolution
        """
        try:
            return resolve_template(
                template, reference_shape, self.get_nodeattr, context_name, interfaces
            )
        except ValueError as e:
            raise self._error(str(e))

    def _validate_interface_constraints(
        self,
        interface_name: str,
        interface_model: Any,
        constraints: List['InterfaceConstraint']
    ) -> None:
        """Validate interface constraints for a single interface.

        Args:
            interface_name: Name of interface being validated
            interface_model: Interface model (InputModel or OutputModel) to validate
            constraints: List of InterfaceConstraints from schema

        Raises:
            KernelOpError: If any constraint is violated
        """
        for constraint in constraints:
            error = constraint.check(interface_model, self.get_nodeattr)
            if error:
                raise self._error(str(error))

    def _resolve_internal_datatypes(self, interfaces: dict) -> dict:
        """Resolve internal datatypes and store in nodeattrs.

        Internal datatypes are resolved using DatatypeSource patterns
        and stored in nodeattrs for persistence.

        Args:
            interfaces: Dict of input models

        Returns:
            Dict mapping internal name → resolved DataType
        """
        internal_datatypes = {}

        for internal_name, datatype_source in self.kernel_schema.internal_datatypes.items():
            try:
                resolved_datatype = datatype_source.resolve(interfaces, self.get_nodeattr)
                internal_datatypes[internal_name] = resolved_datatype

                # Store in nodeattr for persistence
                attr_name = f"_{internal_name}Datatype"
                self._set_protected(attr_name, resolved_datatype.name)

                logger.debug(f"  Internal '{internal_name}': {resolved_datatype.name}")

            except ValueError as e:
                raise self._error(
                    f"Internal datatype '{internal_name}' resolution failed: {e}"
                )

        return internal_datatypes

    # ====================================================================
    # Metadata Caching (Optional Performance Optimization)
    # ====================================================================

    def _cache_kernel_model_to_metadata(self, model: KernelModel) -> None:
        """Cache serialized KernelModel to node metadata.

        Optional performance optimization. Allows fast reconstruction
        without rebuilding from nodeattrs.

        Args:
            model: KernelModel to serialize
        """
        try:
            from onnx import StringStringEntryProto
            from qonnx.util.basic import get_by_name

            # Serialize model to JSON
            model_json = model.to_json()

            # Remove existing cache if present
            existing = get_by_name(
                self.onnx_node.metadata_props,
                'ai.brainsmith.kernel_model',
                'key'
            )
            if existing:
                self.onnx_node.metadata_props.remove(existing)

            # Add new cache
            metadata = StringStringEntryProto()
            metadata.key = 'ai.brainsmith.kernel_model'
            metadata.value = model_json
            self.onnx_node.metadata_props.append(metadata)

            logger.debug(f"Cached KernelModel to metadata for {self.onnx_node.name}")
        except Exception as e:
            # Cache failure is non-fatal
            logger.debug(f"Failed to cache KernelModel: {e}")

    def _read_kernel_model_from_metadata(self) -> Optional[str]:
        """Read serialized KernelModel from node metadata.

        Returns:
            JSON string if cache exists, None otherwise
        """
        try:
            from qonnx.util.basic import get_by_name

            metadata = get_by_name(
                self.onnx_node.metadata_props,
                'ai.brainsmith.kernel_model',
                'key'
            )
            if metadata:
                return metadata.value
        except Exception as e:
            logger.debug(f"Failed to read KernelModel cache: {e}")

        return None
