"""
Step resolution system for Phase 3 build configuration.

This module provides utilities for resolving step specifications (names, indices, 
semantic types) into concrete step references for build execution.
"""

from typing import List, Union, Optional, Dict, Any
from enum import Enum


class InputType(Enum):
    """Semantic input type shortcuts for step configuration."""
    ONNX = "onnx"                    # Start from raw ONNX model
    QONNX = "qonnx"                  # Start from QONNX model
    FINN = "finn"                    # Start from FINN dataflow graph
    HWGRAPH = "hwgraph"              # Start from hardware dataflow graph


class OutputType(Enum):
    """Semantic output type shortcuts for step configuration."""
    QONNX = "qonnx"                  # Stop at QONNX conversion
    FINN = "finn"                    # Stop at FINN dataflow graph
    HWGRAPH = "hwgraph"              # Stop at hardware inference
    RTL = "rtl"                      # Stop at RTL generation
    IP = "ip"                        # Stop at IP generation
    BITSTREAM = "bitstream"          # Stop at bitstream generation


class StepResolver:
    """
    Resolves step specifications into concrete step references.
    
    Supports:
    - Step names (strings): "step_create_dataflow_partition"
    - Step indices (integers): 5
    - Semantic types: input_type="hwgraph", output_type="rtl"
    """
    
    def __init__(self):
        """Initialize step resolver with standard step mappings."""
        # Standard FINN step sequence (based on FINN documentation and BERT demo)
        self.standard_finn_steps = [
            # Cleanup and preparation steps
            "custom_step_cleanup",
            "custom_step_remove_head",
            "custom_step_remove_tail",
            "custom_step_qonnx2finn",
            
            # Analysis and reference generation
            "custom_step_generate_reference_io",
            "custom_streamlining_step", 
            "custom_step_infer_hardware",
            
            # Standard FINN dataflow steps
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_measure_rtlsim_performance",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            
            # Final steps
            "custom_step_shell_metadata_handover",
        ]
        
        # Semantic type to step name mappings
        self.input_type_mappings = {
            InputType.ONNX: "custom_step_cleanup",
            InputType.QONNX: "custom_step_qonnx2finn", 
            InputType.FINN: "custom_streamlining_step",
            InputType.HWGRAPH: "step_create_dataflow_partition",
        }
        
        self.output_type_mappings = {
            OutputType.QONNX: "custom_step_qonnx2finn",
            OutputType.FINN: "custom_streamlining_step",
            OutputType.HWGRAPH: "custom_step_infer_hardware",
            OutputType.RTL: "step_hw_codegen",
            OutputType.IP: "step_hw_ipgen", 
            OutputType.BITSTREAM: "step_create_stitched_ip",
        }
    
    def resolve_step_specification(self, 
                                 step_spec: Optional[Union[str, int]], 
                                 step_list: Optional[List[str]] = None,
                                 semantic_type: Optional[Union[InputType, OutputType, str]] = None) -> Optional[Union[str, int]]:
        """
        Resolve a step specification into a concrete step reference.
        
        Args:
            step_spec: Step name (str) or index (int)
            step_list: Custom step list to resolve against (defaults to standard)
            semantic_type: Semantic type for automatic resolution
            
        Returns:
            Resolved step name or index, or None if not resolvable
            
        Raises:
            ValueError: If step specification is invalid
        """
        # Use standard steps if no custom list provided
        if step_list is None:
            step_list = self.standard_finn_steps
            
        # Handle semantic type resolution first
        if semantic_type is not None:
            resolved_step = self._resolve_semantic_type(semantic_type)
            if resolved_step:
                step_spec = resolved_step
        
        # No step specification provided
        if step_spec is None:
            return None
            
        # Handle string step names
        if isinstance(step_spec, str):
            if step_spec in step_list:
                return step_spec
            else:
                raise ValueError(f"Step '{step_spec}' not found in step list. Available steps: {step_list}")
        
        # Handle integer indices
        if isinstance(step_spec, int):
            if 0 <= step_spec < len(step_list):
                return step_spec
            else:
                raise ValueError(f"Step index {step_spec} out of range [0, {len(step_list)-1}]")
        
        raise ValueError(f"Invalid step specification: {step_spec}. Must be string name or integer index.")
    
    def resolve_step_range(self,
                          start_step: Optional[Union[str, int]] = None,
                          stop_step: Optional[Union[str, int]] = None,
                          input_type: Optional[Union[InputType, str]] = None,
                          output_type: Optional[Union[OutputType, str]] = None,
                          step_list: Optional[List[str]] = None) -> tuple[Optional[Union[str, int]], Optional[Union[str, int]]]:
        """
        Resolve start and stop step specifications.
        
        Args:
            start_step: Starting step name or index
            stop_step: Stopping step name or index
            input_type: Semantic input type for start step
            output_type: Semantic output type for stop step  
            step_list: Custom step list (defaults to standard)
            
        Returns:
            Tuple of (resolved_start_step, resolved_stop_step)
            
        Raises:
            ValueError: If step specifications are invalid or inconsistent
        """
        if step_list is None:
            step_list = self.standard_finn_steps
            
        # Convert string enums to proper enums
        if isinstance(input_type, str):
            try:
                input_type = InputType(input_type)
            except ValueError:
                raise ValueError(f"Invalid input_type: {input_type}")
                
        if isinstance(output_type, str):
            try:
                output_type = OutputType(output_type)
            except ValueError:
                raise ValueError(f"Invalid output_type: {output_type}")
        
        # Resolve start step
        resolved_start = self.resolve_step_specification(
            start_step, step_list, input_type
        )
        
        # Resolve stop step  
        resolved_stop = self.resolve_step_specification(
            stop_step, step_list, output_type
        )
        
        # Validate step ordering if both are specified and are indices
        if (resolved_start is not None and resolved_stop is not None and
            isinstance(resolved_start, int) and isinstance(resolved_stop, int)):
            if resolved_start > resolved_stop:
                raise ValueError(f"Start step index {resolved_start} must be <= stop step index {resolved_stop}")
        
        # Validate step ordering if both are names
        if (resolved_start is not None and resolved_stop is not None and
            isinstance(resolved_start, str) and isinstance(resolved_stop, str)):
            try:
                start_idx = step_list.index(resolved_start)
                stop_idx = step_list.index(resolved_stop)
                if start_idx > stop_idx:
                    raise ValueError(f"Start step '{resolved_start}' (index {start_idx}) must come before stop step '{resolved_stop}' (index {stop_idx})")
            except ValueError as e:
                if "not in list" in str(e):
                    raise ValueError(f"Step not found in step list: {e}")
                raise
        
        return resolved_start, resolved_stop
    
    def _resolve_semantic_type(self, semantic_type: Union[InputType, OutputType, str]) -> Optional[str]:
        """Resolve a semantic type to a step name."""
        # Convert string to enum if needed
        if isinstance(semantic_type, str):
            # Try InputType first
            try:
                semantic_type = InputType(semantic_type)
            except ValueError:
                try:
                    semantic_type = OutputType(semantic_type)
                except ValueError:
                    raise ValueError(f"Invalid semantic type: {semantic_type}")
        
        # Resolve enum to step name
        if isinstance(semantic_type, InputType):
            return self.input_type_mappings.get(semantic_type)
        elif isinstance(semantic_type, OutputType):
            return self.output_type_mappings.get(semantic_type)
        
        return None
    
    def get_step_slice(self, 
                      step_list: List[str],
                      start_step: Optional[Union[str, int]] = None,
                      stop_step: Optional[Union[str, int]] = None) -> List[str]:
        """
        Get a slice of the step list based on start/stop specifications.
        
        Args:
            step_list: List of step names
            start_step: Starting step name or index (inclusive)
            stop_step: Stopping step name or index (inclusive)
            
        Returns:
            Sliced list of steps
        """
        # Convert step names to indices
        start_idx = 0
        if start_step is not None:
            if isinstance(start_step, str):
                start_idx = step_list.index(start_step)
            else:
                start_idx = start_step
                
        stop_idx = len(step_list) - 1
        if stop_step is not None:
            if isinstance(stop_step, str):
                stop_idx = step_list.index(stop_step)
            else:
                stop_idx = stop_step
        
        # Return slice (stop_idx + 1 because Python slicing is exclusive)
        return step_list[start_idx:stop_idx + 1]
    
    def get_standard_steps(self) -> List[str]:
        """Get the standard FINN step sequence."""
        return self.standard_finn_steps.copy()
    
    def get_supported_input_types(self) -> List[str]:
        """Get list of supported input type strings."""
        return [t.value for t in InputType]
    
    def get_supported_output_types(self) -> List[str]:
        """Get list of supported output type strings."""
        return [t.value for t in OutputType]