"""
TensorChunking: ONNX layout to dataflow interface dimension mapping

This module handles mapping between ONNX tensor layouts and dataflow interface
dimensions (qDim/tDim) with support for pragma-based overrides and complex
chunking patterns.
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass

from .dataflow_interface import DataflowInterface
from .validation import ValidationError, ValidationResult, create_validation_result, ValidationSeverity

@dataclass
class TDimPragma:
    """Representation of a TDIM pragma for custom tensor dimension specification"""
    interface_name: str
    dimension_expressions: List[str]
    
    def evaluate_expressions(self, parameters: Dict[str, Any]) -> List[int]:
        """Evaluate dimension expressions using module parameters"""
        evaluated_dims = []
        
        for expr in self.dimension_expressions:
            try:
                # Simple expression evaluator - can be enhanced for complex expressions
                result = self._evaluate_expression(expr, parameters)
                evaluated_dims.append(result)
            except Exception as e:
                raise ValueError(f"Failed to evaluate TDIM expression '{expr}': {e}")
        
        return evaluated_dims
    
    def _evaluate_expression(self, expr: str, parameters: Dict[str, Any]) -> int:
        """Evaluate a single dimension expression"""
        # Handle simple parameter substitution and basic arithmetic
        expr = expr.strip()
        
        # Replace parameter names with values
        for param_name, param_value in parameters.items():
            expr = expr.replace(param_name, str(param_value))
        
        # Handle basic arithmetic operations (*, +, -, /)
        try:
            # Use eval for simple expressions (in production, consider a safer parser)
            result = eval(expr)
            if isinstance(result, (int, float)):
                return int(result)
            else:
                raise ValueError(f"Expression '{expr}' did not evaluate to a number")
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")

class TensorChunking:
    """
    Handles ONNX layout to qDim/tDim mapping with support for
    pragma-based overrides and complex chunking patterns.
    """
    
    # Standard layout mapping table
    LAYOUT_MAPPINGS = {
        "[N, C]": {
            "qDim_func": lambda shape: [1], 
            "tDim_func": lambda shape: [shape[1]]
        },
        "[N, C, H, W]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2] * shape[3]]
        },
        "[N, H, W, C]": {
            "qDim_func": lambda shape: [shape[1] * shape[2]], 
            "tDim_func": lambda shape: [shape[3]]
        },
        "[N, L, C]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2]]
        },
        "[N, C, L]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2]]
        },
        "[N, L, h, d]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2] * shape[3]]
        },
        # Add more common ONNX layouts as needed
        "[N, H, W]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2]]
        },
        "[N, C, H, W, D]": {
            "qDim_func": lambda shape: [shape[1]], 
            "tDim_func": lambda shape: [shape[2] * shape[3] * shape[4]]
        }
    }
    
    @staticmethod
    def infer_dimensions(onnx_layout: str, shape: List[int]) -> Tuple[List[int], List[int]]:
        """
        Map ONNX tensor layout to qDim/tDim using standard patterns
        
        Args:
            onnx_layout: Layout string (e.g., "[N, C, H, W]")
            shape: Tensor shape dimensions
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        if onnx_layout in TensorChunking.LAYOUT_MAPPINGS:
            mapping = TensorChunking.LAYOUT_MAPPINGS[onnx_layout]
            qDim = mapping["qDim_func"](shape)
            tDim = mapping["tDim_func"](shape)
            return qDim, tDim
        else:
            # Default mapping: assume all dimensions except first are chunked together
            if len(shape) >= 2:
                qDim = [1]  # Single query
                tDim = [int(np.prod(shape[1:]))]  # Everything except batch
                return qDim, tDim
            else:
                # Single dimension case
                return [1], [shape[0] if shape else 1]
    
    @staticmethod
    def apply_tdim_pragma(pragma: TDimPragma, parameters: Dict[str, Any]) -> List[int]:
        """
        Apply TDIM pragma to override default chunking
        
        Args:
            pragma: Parsed TDIM pragma with dimension expressions
            parameters: Module parameters for expression evaluation
            
        Returns:
            Computed tDim list
        """
        return pragma.evaluate_expressions(parameters)
    
    @staticmethod
    def validate_chunking(interface: DataflowInterface) -> ValidationResult:
        """
        Validate that qDim, tDim, sDim relationships are mathematically sound
        """
        result = create_validation_result()
        
        # Check dimension count consistency
        if len(interface.qDim) != len(interface.tDim) or len(interface.tDim) != len(interface.sDim):
            error = ValidationError(
                component=f"interface.{interface.name}",
                error_type="dimension_mismatch",
                message=f"qDim, tDim, and sDim must have same length: qDim={len(interface.qDim)}, tDim={len(interface.tDim)}, sDim={len(interface.sDim)}",
                severity=ValidationSeverity.ERROR,
                context={
                    "interface": interface.name,
                    "qDim_length": len(interface.qDim),
                    "tDim_length": len(interface.tDim),
                    "sDim_length": len(interface.sDim)
                }
            )
            result.add_error(error)
            return result
        
        # Check mathematical relationships
        for i, (q, t, s) in enumerate(zip(interface.qDim, interface.tDim, interface.sDim)):
            # Validate positive values
            if q <= 0 or t <= 0 or s <= 0:
                error = ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="invalid_dimension",
                    message=f"All dimensions must be positive: qDim[{i}]={q}, tDim[{i}]={t}, sDim[{i}]={s}",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "interface": interface.name,
                        "dimension_index": i,
                        "qDim": q,
                        "tDim": t,
                        "sDim": s
                    }
                )
                result.add_error(error)
                continue
            
            # Check qDim >= tDim
            if q < t:
                error = ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="dimension_ordering",
                    message=f"qDim[{i}] ({q}) must be >= tDim[{i}] ({t})",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "interface": interface.name,
                        "dimension_index": i,
                        "qDim": q,
                        "tDim": t
                    }
                )
                result.add_error(error)
            
            # Check tDim >= sDim
            if t < s:
                error = ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="dimension_ordering",
                    message=f"tDim[{i}] ({t}) must be >= sDim[{i}] ({s})",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "interface": interface.name,
                        "dimension_index": i,
                        "tDim": t,
                        "sDim": s
                    }
                )
                result.add_error(error)
            
            # Check divisibility: qDim % tDim == 0
            if q % t != 0:
                error = ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="divisibility_violation",
                    message=f"qDim[{i}] ({q}) must be divisible by tDim[{i}] ({t})",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "interface": interface.name,
                        "dimension_index": i,
                        "dividend": q,
                        "divisor": t
                    }
                )
                result.add_error(error)
            
            # Check divisibility: tDim % sDim == 0
            if t % s != 0:
                error = ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="divisibility_violation",
                    message=f"tDim[{i}] ({t}) must be divisible by sDim[{i}] ({s})",
                    severity=ValidationSeverity.ERROR,
                    context={
                        "interface": interface.name,
                        "dimension_index": i,
                        "dividend": t,
                        "divisor": s
                    }
                )
                result.add_error(error)
        
        return result
    
    @staticmethod
    def optimize_chunking(interface: DataflowInterface, target_parallelism: int) -> Tuple[List[int], List[int]]:
        """
        Optimize tDim and sDim for target parallelism while maintaining mathematical constraints
        
        Args:
            interface: Interface to optimize
            target_parallelism: Desired parallelism level
            
        Returns:
            Tuple of optimized (tDim, sDim)
        """
        optimized_tDim = interface.tDim.copy()
        optimized_sDim = interface.sDim.copy()
        
        # For simplicity, optimize only the first dimension
        if len(optimized_tDim) > 0 and len(optimized_sDim) > 0:
            qDim_0 = interface.qDim[0]
            
            # Find best tDim that allows target parallelism
            best_tDim = optimized_tDim[0]
            best_sDim = min(target_parallelism, best_tDim)
            
            # Ensure tDim divides qDim
            for potential_tDim in range(optimized_tDim[0], qDim_0 + 1):
                if qDim_0 % potential_tDim == 0:
                    potential_sDim = min(target_parallelism, potential_tDim)
                    if potential_tDim % potential_sDim == 0:
                        best_tDim = potential_tDim
                        best_sDim = potential_sDim
                        break
            
            optimized_tDim[0] = best_tDim
            optimized_sDim[0] = best_sDim
        
        return optimized_tDim, optimized_sDim
    
    @staticmethod
    def create_chunking_report(interface: DataflowInterface) -> Dict[str, Any]:
        """
        Generate a comprehensive chunking analysis report
        """
        total_elements = 1
        for q in interface.qDim:
            total_elements *= q
        
        elements_per_tensor = 1
        for t in interface.tDim:
            elements_per_tensor *= t
        
        elements_per_cycle = 1
        for s in interface.sDim:
            elements_per_cycle *= s
        
        tensors_per_query = total_elements // elements_per_tensor if elements_per_tensor > 0 else 0
        cycles_per_tensor = elements_per_tensor // elements_per_cycle if elements_per_cycle > 0 else 0
        total_cycles = tensors_per_query * cycles_per_tensor
        
        return {
            "interface_name": interface.name,
            "total_elements": total_elements,
            "elements_per_tensor": elements_per_tensor,
            "elements_per_cycle": elements_per_cycle,
            "tensors_per_query": tensors_per_query,
            "cycles_per_tensor": cycles_per_tensor,
            "total_cycles": total_cycles,
            "parallelism_efficiency": elements_per_cycle / elements_per_tensor if elements_per_tensor > 0 else 0,
            "memory_footprint_bits": total_elements * interface.dtype.bitwidth,
            "bandwidth_per_cycle_bits": elements_per_cycle * interface.dtype.bitwidth
        }

# Import numpy for calculations
import numpy as np

# Factory functions for common chunking patterns
def create_simple_chunking(interface_name: str, total_elements: int, parallelism: int = 1) -> Tuple[List[int], List[int], List[int]]:
    """Create simple 1D chunking pattern"""
    qDim = [total_elements]
    tDim = [total_elements]
    sDim = [min(parallelism, total_elements)]
    return qDim, tDim, sDim

def create_2d_chunking(interface_name: str, height: int, width: int, parallelism: int = 1) -> Tuple[List[int], List[int], List[int]]:
    """Create 2D chunking pattern (e.g., for images)"""
    qDim = [height, width]
    tDim = [height, width]
    sDim = [1, min(parallelism, width)]
    return qDim, tDim, sDim

def create_channel_chunking(interface_name: str, channels: int, spatial_size: int, parallelism: int = 1) -> Tuple[List[int], List[int], List[int]]:
    """Create channel-wise chunking pattern"""
    qDim = [channels]
    tDim = [spatial_size]
    sDim = [min(parallelism, spatial_size)]
    return qDim, tDim, sDim
