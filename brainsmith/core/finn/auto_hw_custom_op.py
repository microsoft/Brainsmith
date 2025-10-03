############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Base class for automatic hardware custom operators.

Two-level caching system:
1. _tensor_context: Cached from ModelWrapper (only changes with graph updates)
2. _kernel_model: Computed from schema + context + nodeattrs (invalidated on nodeattr changes)

Direct Model Creation Flow:
    The AutoHWCustomOp directly creates its KernelModel from:
    - kernel_schema (class attribute)
    - _tensor_context (from graph)
    - nodeattrs (via self.get_nodeattr)

Cache invalidation:
  set_nodeattr() → invalidates _kernel_model only (tensor context unchanged)
  graph changes → refresh_kernel_model() updates both caches
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.core.dataflow import (
    KernelSchema,
    TensorContext,
    TensorInfo,
    InputSchema,
    OutputSchema,
    InputModel,
    OutputModel,
    KernelModel,
    create_kernel_model,
    validate_datatype_against_constraints,
    RelationType
)
from brainsmith.core.dataflow.shape_utils import (
    create_folded_shape,
    calculate_stream_width
)
from brainsmith.core.dataflow.template_utils import resolve_template_params
from brainsmith.core.dataflow.types import prod


class AutoHWCustomOp(HWCustomOp, ABC):
    """Base class for automatic hardware custom operators.

    Key features:
    - Direct model creation from schema + context + nodeattrs
    - Two-phase model creation separating nodeattr and ModelWrapper dependencies
    - Automatic cache invalidation on nodeattr changes
    - Efficient partial updates when only nodeattrs change

    Subclasses must:
    - Define kernel_schema class attribute with KernelSchema instance
    - Implement abstract methods from HWCustomOp base class
    """

    # =============================================================================
    # Class Attributes
    # =============================================================================

    # Subclasses must define this class attribute
    kernel_schema: KernelSchema = None

    # =============================================================================
    # Initialization
    # =============================================================================

    def __init__(self, onnx_node, **kwargs):
        """Initialize with two-level cache system."""
        super().__init__(onnx_node, **kwargs)

        if self.kernel_schema is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define kernel_schema class attribute"
            )

        # Two-level cache: tensor context (from graph) and kernel model (computed)
        self._tensor_context: Optional[TensorContext] = None
        self._kernel_model: Optional[KernelModel] = None

    # =============================================================================
    # Public API - Model Access
    # =============================================================================

    def get_kernel_model(self) -> KernelModel:
        """Get the cached kernel model.

        Returns cached model if available, otherwise raises error.
        Call refresh_kernel_model() first to populate cache.
        """
        if self._kernel_model is None:
            raise RuntimeError(
                "Kernel model not initialized. Call refresh_kernel_model() first."
            )
        return self._kernel_model


    # =============================================================================
    # Public API - Model Refresh
    # =============================================================================

    def refresh_kernel_model(self, model: ModelWrapper) -> None:
        """Refresh complete kernel model with ModelWrapper info.

        This should be called by transforms when shapes or types change.

        Args:
            model: The global ModelWrapper instance (not cached)
        """
        # Extract tensor context
        self._tensor_context = TensorContext.from_model_wrapper(
            self.onnx_node, model
        )

        # Create model directly using private methods
        self._kernel_model = self._create_model_from_context()

    def refresh_from_model(self, model: ModelWrapper) -> None:
        """Update tensor context from model and rebuild kernel model.

        This updates the internal state to match the current graph state,
        then propagates output datatypes back to the graph.

        Args:
            model: The ModelWrapper containing the current graph state
        """
        # Update tensor context from current graph
        self._tensor_context = TensorContext.from_model_wrapper(
            self.onnx_node, model
        )

        # Rebuild kernel model with updated context
        self._kernel_model = self._create_model_from_context()

        # Propagate output datatypes back to graph
        for i, output in enumerate(self._kernel_model.outputs):
            model.set_tensor_datatype(
                self.onnx_node.output[i],
                output.datatype
            )

    # =============================================================================
    # Override - Nodeattr Management with Change Detection
    # =============================================================================

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set attribute and invalidate kernel model if needed."""
        # Get old value for comparison
        old_value = None
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            # Attribute not set yet - this is expected for initial setup
            old_value = None

        # Set the new value
        super().set_nodeattr(name, value)

        # If value changed and affects model, invalidate kernel model
        # (but NOT tensor context - that comes from the graph!)
        if old_value != value and self._is_model_parameter(name):
            self._kernel_model = None

    def _is_model_parameter(self, name: str) -> bool:
        """Check if attribute affects model creation."""
        # Cache the parameter set on first use for O(1) lookups
        if not hasattr(self, '_model_parameters'):
            self._model_parameters = self._extract_model_parameters()

        return name in self._model_parameters

    def _extract_model_parameters(self) -> Set[str]:
        """Extract all parameter names that affect model creation."""
        params = {'clock_freq_mhz'}  # Always affects model

        # Add datatype attributes
        for i, inp in enumerate(self.kernel_schema.inputs):
            params.add(inp.get_datatype_attr(i))
        for i, out in enumerate(self.kernel_schema.outputs):
            params.add(out.get_datatype_attr(i))

        # Add tiling parameters from block/stream templates
        for inp in self.kernel_schema.inputs:
            if inp.block_tiling:
                params.update(
                    item for item in inp.block_tiling
                    if isinstance(item, str) and item != ":"
                )
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                params.update(
                    item for item in inp.stream_tiling
                    if isinstance(item, str) and item != ":"
                )

        for out in self.kernel_schema.outputs:
            if out.block_tiling:
                params.update(
                    item for item in out.block_tiling
                    if isinstance(item, str) and item != ":"
                )

        return params

    # =============================================================================
    # Override - HWCustomOp Interface Methods (Delegating to Cached Model)
    # =============================================================================

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape using cached model."""
        model = self.get_kernel_model()
        output = model.outputs[ind]
        return create_folded_shape(output.tensor_dims, output.block_dims)

    def get_number_output_values(self):
        """Get total output values using cached model."""
        model = self.get_kernel_model()
        total = 0
        for output in model.outputs:
            total += prod(output.tensor_dims)
        return total

    def get_exp_cycles(self):
        """Get expected cycles using cached model."""
        model = self.get_kernel_model()
        return model.initiation_interval

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype from cached model."""
        return self.get_kernel_model().inputs[ind].datatype

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype from cached model."""
        return self.get_kernel_model().outputs[ind].datatype

    def get_normal_input_shape(self, ind=0) -> List[int]:
        """Get normal input shape from cached model."""
        return list(self.get_kernel_model().inputs[ind].tensor_dims)

    def get_normal_output_shape(self, ind=0) -> List[int]:
        """Get normal output shape from cached model."""
        return list(self.get_kernel_model().outputs[ind].tensor_dims)

    def get_folded_input_shape(self, ind=0) -> List[int]:
        """Get folded input shape using cached model."""
        model = self.get_kernel_model()
        inp = model.inputs[ind]
        return create_folded_shape(inp.tensor_dims, inp.block_dims)

    def get_instream_width(self, ind=0) -> int:
        """Get input stream width in bits."""
        inp = self.get_kernel_model().inputs[ind]
        return calculate_stream_width(inp.streaming_bandwidth, inp.datatype.bitwidth())

    def get_outstream_width(self, ind=0) -> int:
        """Get output stream width in bits."""
        out = self.get_kernel_model().outputs[ind]
        return calculate_stream_width(out.streaming_rate, out.datatype.bitwidth())

    # =============================================================================
    # Override - Template and Performance Methods
    # =============================================================================

    def get_template_values(self):
        """Get template values from cached model."""
        model = self.get_kernel_model()

        templates = {}

        # Input parameters
        for inp in model.inputs:
            templates[f"{inp.name}_block_dims"] = inp.block_dims
            templates[f"{inp.name}_stream_dims"] = inp.stream_dims

        # Output parameters
        for out in model.outputs:
            templates[f"{out.name}_block_dims"] = out.block_dims

        # Kernel parameters
        templates.update(model.parameters)

        return templates

    # =============================================================================
    # Public API - Tensor Propagation
    # =============================================================================

    def propagate_tensors(self, model: ModelWrapper) -> None:
        """
        Propagate tensor information to node attributes.

        This enforces the core principle that datatypes are model-level
        decisions, not implementation choices. TensorContext datatypes are
        authoritative and will be set as node attributes.

        Args:
            model: The ModelWrapper containing the full ONNX graph
        """
        # First, ensure we have tensor context
        if self._tensor_context is None:
            self._tensor_context = TensorContext.from_model_wrapper(
                self.onnx_node, model
            )

        # Build name lookup for context tensors
        context_inputs = {t.name: t for t in self._tensor_context.inputs}
        context_outputs = {t.name: t for t in self._tensor_context.outputs}

        # Process inputs - enforce TensorContext datatypes
        for i, inp in enumerate(self.kernel_schema.inputs):
            if inp.name in context_inputs:
                tensor_info = context_inputs[inp.name]

                if tensor_info.datatype:
                    # Validate against constraints
                    if inp.datatype_constraints:
                        if not validate_datatype_against_constraints(
                            tensor_info.datatype, inp.datatype_constraints
                        ):
                            raise ValueError(
                                f"Input '{inp.name}' datatype {tensor_info.datatype.name} "
                                f"violates schema constraints"
                            )

                    # Model datatypes are authoritative - set nodeattr
                    datatype_attr = inp.get_datatype_attr(i)
                    final_datatype = tensor_info.datatype

                    # Special case: weights can be minimized for hardware storage
                    if inp.is_weight:
                        minimize_method = f"minimize_bitwidth_weight_{i}"
                        if hasattr(self, minimize_method):
                            method = getattr(self, minimize_method)
                            optimized = method(tensor_info.datatype)
                            if optimized:
                                final_datatype = optimized

                    self.set_nodeattr(datatype_attr, final_datatype.name)

        # Process outputs - enforce TensorContext datatypes with optimization
        for i, out in enumerate(self.kernel_schema.outputs):
            if out.name in context_outputs:
                tensor_info = context_outputs[out.name]

                if tensor_info.datatype:
                    # Validate against constraints
                    if out.datatype_constraints:
                        if not validate_datatype_against_constraints(
                            tensor_info.datatype, out.datatype_constraints
                        ):
                            raise ValueError(
                                f"Output '{out.name}' datatype {tensor_info.datatype.name} "
                                f"violates schema constraints"
                            )

                    # Model datatypes are authoritative - set nodeattr
                    datatype_attr = out.get_datatype_attr(i)
                    final_datatype = tensor_info.datatype

                    # Outputs can be minimized for hardware storage
                    minimize_method = f"minimize_bitwidth_output_{i}"
                    if hasattr(self, minimize_method):
                        method = getattr(self, minimize_method)
                        optimized = method(tensor_info.datatype)
                        if optimized:
                            final_datatype = optimized

                    self.set_nodeattr(datatype_attr, final_datatype.name)

        # Invalidate caches since we've updated attributes
        self._kernel_model = None

    def infer_node_datatype(self, model):
        """FINN compatibility: Infer and set node datatypes from graph context.

        This is a compatibility wrapper for legacy FINN code. Modern code should
        use refresh_from_model() directly.

        Args:
            model: The ModelWrapper containing the graph
        """
        self.refresh_from_model(model)

    # =============================================================================
    # Private - Model Creation Methods (formerly DirectKernelFactory)
    # =============================================================================

    def _create_model_from_context(self) -> KernelModel:
        """
        Create a KernelModel directly from schema, context, and nodeattrs.

        This replaces DirectKernelFactory.create_model() but works directly
        with self.get_nodeattr() instead of a nodeattrs dict.

        Returns:
            KernelModel: Fully validated and ready-to-use model

        Raises:
            ValueError: If validation fails at any stage
        """
        if self._tensor_context is None:
            raise RuntimeError("Tensor context not initialized")

        # Step 1: Validate tensor context satisfies schema constraints
        self._validate_tensor_context()

        # Step 2: Build input models with validation
        input_models = self._create_input_models()

        # Step 3: Build output models with validation
        output_models = self._create_output_models()

        # Step 4: Extract performance attributes
        try:
            clock_freq = self.get_nodeattr("clock_freq_mhz")
        except AttributeError:
            clock_freq = 100  # Default 100 MHz

        # Step 5: Collect all nodeattrs for parameters
        parameters = {}
        for attr_name in self.get_nodeattr_types():
            try:
                parameters[attr_name] = self.get_nodeattr(attr_name)
            except AttributeError:
                # Attribute not set yet - this is expected for optional attrs
                pass

        # Step 6: Build and validate final model
        model = create_kernel_model(
            name=self.kernel_schema.name,
            inputs=input_models,
            outputs=output_models,
            parameters=parameters,
            clock_freq_mhz=clock_freq
        )

        # Step 7: Validate relationships and streaming
        self._validate_model_relationships(model)
        self._validate_streaming_config(model)

        return model

    def _validate_tensor_context(self) -> None:
        """Validate that tensor context satisfies schema constraints."""
        if self._tensor_context is None:
            raise RuntimeError("Tensor context not initialized")

        # Build name lookup for context tensors
        context_inputs = {t.name: t for t in self._tensor_context.inputs}
        context_outputs = {t.name: t for t in self._tensor_context.outputs}

        # Check required inputs exist
        for inp in self.kernel_schema.inputs:
            if not inp.optional and inp.name not in context_inputs:
                raise ValueError(f"Required input '{inp.name}' not found in tensor context")

        # Check all outputs exist
        for out in self.kernel_schema.outputs:
            if out.name not in context_outputs:
                raise ValueError(f"Output '{out.name}' not found in tensor context")

        # Validate datatype constraints
        for inp in self.kernel_schema.inputs:
            if inp.name in context_inputs:
                tensor_info = context_inputs[inp.name]
                if inp.datatype_constraints and tensor_info.datatype:
                    if not validate_datatype_against_constraints(
                        tensor_info.datatype, inp.datatype_constraints
                    ):
                        raise ValueError(
                            f"Input '{inp.name}' datatype {tensor_info.datatype.name} "
                            f"violates schema constraints"
                        )

    def _create_input_models(self) -> List[InputModel]:
        """Create input models with full validation."""
        models = []

        # Build name lookup for context tensors
        context_inputs = {t.name: t for t in self._tensor_context.inputs}

        for inp_schema in self.kernel_schema.inputs:
            # Skip optional inputs not present
            if inp_schema.optional and inp_schema.name not in context_inputs:
                continue

            tensor_info = context_inputs[inp_schema.name]

            # Resolve block dimensions
            block_dims = self._resolve_dimensions(
                inp_schema.block_tiling, tensor_info.shape,
                f"input '{inp_schema.name}' block"
            )

            # Resolve stream dimensions (if specified)
            stream_dims = None
            if inp_schema.stream_tiling:
                stream_dims = self._resolve_dimensions(
                    inp_schema.stream_tiling, block_dims,
                    f"input '{inp_schema.name}' stream"
                )

            # Validate streaming divides block
            self._validate_streaming_divides_block(
                stream_dims, block_dims, f"input '{inp_schema.name}'"
            )

            # Determine datatype
            datatype = self._resolve_datatype(inp_schema, tensor_info)

            # Create shape objects (convert lists to tuples)
            tensor_shape = tuple(tensor_info.shape)
            block_shape = tuple(block_dims)
            stream_shape = tuple(stream_dims) if stream_dims else None

            models.append(InputModel(
                name=inp_schema.name,
                tensor_dims=tensor_shape,
                block_dims=block_shape,
                stream_dims=stream_shape,
                datatype=datatype,
                is_weight=inp_schema.is_weight
            ))

        return models

    def _create_output_models(self) -> List[OutputModel]:
        """Create output models with full validation."""
        models = []

        # Build name lookup for context tensors
        context_outputs = {t.name: t for t in self._tensor_context.outputs}

        for out_schema in self.kernel_schema.outputs:
            tensor_info = context_outputs[out_schema.name]

            # Resolve block dimensions
            block_dims = self._resolve_dimensions(
                out_schema.block_tiling, tensor_info.shape,
                f"output '{out_schema.name}' block"
            )

            # Outputs don't have stream tiling - they output at block rate
            stream_dims = None

            # Validate streaming divides block
            self._validate_streaming_divides_block(
                stream_dims, block_dims, f"output '{out_schema.name}'"
            )

            # Determine datatype
            datatype = self._resolve_datatype(out_schema, tensor_info)

            # Create shape objects (convert lists to tuples)
            tensor_shape = tuple(tensor_info.shape)
            block_shape = tuple(block_dims)
            stream_shape = tuple(stream_dims) if stream_dims else None

            # Calculate streaming rate based on block dimensions
            # For outputs, streaming rate is the product of block dimensions
            streaming_rate = 1
            for dim in block_dims:
                streaming_rate *= dim

            models.append(OutputModel(
                name=out_schema.name,
                tensor_dims=tensor_shape,
                block_dims=block_shape,
                datatype=datatype,
                streaming_rate=streaming_rate
            ))

        return models

    def _resolve_dimensions(
        self,
        template: List[Any],
        reference_shape: List[int],
        context_name: str
    ) -> List[int]:
        """Resolve template dimensions to concrete values."""
        # For streaming, template can be shorter than reference shape
        # Pad with 1s at the beginning
        if len(template) < len(reference_shape):
            padding = len(reference_shape) - len(template)
            template = [1] * padding + template
        elif len(template) > len(reference_shape):
            raise ValueError(
                f"{context_name}: template length {len(template)} exceeds "
                f"tensor rank {len(reference_shape)}"
            )

        # Create param getter function that uses self.get_nodeattr
        def param_getter(name: str):
            try:
                return self.get_nodeattr(name)
            except AttributeError:
                raise ValueError(f"Missing required parameter '{name}' for {context_name}")

        # First resolve parameters
        resolved_template = resolve_template_params(template, param_getter)

        # Then process each dimension
        resolved = []
        for i, (tmpl_resolved, tmpl_orig, ref) in enumerate(zip(resolved_template, template, reference_shape)):
            if tmpl_orig == ":":  # Full dimension
                value = ref
            elif isinstance(tmpl_resolved, int):
                value = tmpl_resolved
            else:
                # This should have been resolved
                raise ValueError(f"{context_name}[{i}]: unresolved template value {tmpl_resolved}")

            # Validate against reference
            if tmpl_orig == ":" and value != ref:
                raise ValueError(
                    f"{context_name}[{i}]: expected full dimension {ref}, got {value}"
                )
            elif value > ref:
                raise ValueError(
                    f"{context_name}[{i}]: resolved value {value} exceeds tensor dimension {ref}"
                )

            resolved.append(value)

        return resolved

    def _validate_streaming_divides_block(
        self,
        stream_dims: Optional[List[int]],
        block_dims: List[int],
        context_name: str
    ) -> None:
        """Validate streaming dimensions divide block dimensions."""
        if not stream_dims:
            return

        for i, (stream, block) in enumerate(zip(stream_dims, block_dims)):
            if block % stream != 0:
                raise ValueError(
                    f"{context_name}: stream[{i}]={stream} doesn't divide "
                    f"block[{i}]={block} evenly"
                )

    def _resolve_datatype(
        self,
        interface_schema: Any,
        tensor_info: TensorInfo
    ) -> DataType:
        """
        Resolve datatype with validation.

        Priority:
        1. NodeAttr override (if allowed by constraints)
        2. Tensor context datatype
        3. Error if neither available
        """
        # Check for nodeattr override
        attr_name = f"{interface_schema.name}Datatype"
        try:
            override_str = self.get_nodeattr(attr_name)
            try:
                override_dt = DataType[override_str]
                # Validate against constraints
                if interface_schema.datatype_constraints:
                    if not validate_datatype_against_constraints(
                        override_dt, interface_schema.datatype_constraints
                    ):
                        raise ValueError(
                            f"NodeAttr datatype {override_str} violates constraints "
                            f"for {interface_schema.name}"
                        )
                return override_dt
            except KeyError:
                raise ValueError(f"Invalid datatype string: {override_str}")
        except AttributeError:
            # No nodeattr override
            pass

        # Use tensor context datatype
        if tensor_info.datatype:
            return tensor_info.datatype

        # No datatype available
        raise ValueError(
            f"No datatype available for {interface_schema.name}. "
            f"Either tensor must have datatype or provide {attr_name} nodeattr"
        )

    def _validate_model_relationships(self, model: KernelModel) -> None:
        """Validate dimension relationships in final model."""
        # Build interface lookup
        interfaces = {}
        for inp in model.inputs:
            interfaces[inp.name] = inp.tensor_dims
        for out in model.outputs:
            interfaces[out.name] = out.tensor_dims

        # Check each relationship
        for rel in self.kernel_schema.relationships:
            source_shape = interfaces.get(rel.source_interface)
            target_shape = interfaces.get(rel.target_interface)

            if not source_shape or not target_shape:
                continue  # Already validated in earlier steps

            # Handle None dimensions (means total size)
            if rel.source_dim is None:
                source_val = 1
                for dim in source_shape:
                    source_val *= dim
            else:
                source_val = source_shape[rel.source_dim]

            if rel.target_dim is None:
                target_val = 1
                for dim in target_shape:
                    target_val *= dim
            else:
                target_val = target_shape[rel.target_dim]

            # Evaluate relationship
            valid = False
            if rel.relation == RelationType.EQUAL:
                valid = source_val == target_val
            elif rel.relation == RelationType.MULTIPLE:
                valid = source_val == rel.factor * target_val
            elif rel.relation == RelationType.DEPENDENT:
                # For DEPENDENT, we just validate it exists
                valid = True

            if not valid:
                raise ValueError(
                    f"Relationship {rel.relation.name} violated: "
                    f"{rel.source_interface}[{rel.source_dim}]={source_val} "
                    f"vs {rel.target_interface}[{rel.target_dim}]={target_val}"
                )

    def _validate_streaming_config(self, model: KernelModel) -> None:
        """Validate overall streaming configuration."""
        # Check bandwidth constraints
        for inp in model.inputs:
            if hasattr(inp, 'bandwidth_bits') and inp.bandwidth_bits > 1024:
                # Warning only - don't fail
                print(f"Warning: Input {inp.name} bandwidth {inp.bandwidth_bits} bits "
                      f"exceeds typical 1024 bit limit")