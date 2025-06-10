############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Automatic Layout Detection for Interface-Wise Dataflow Modeling
############################################################################

"""Automatic layout detection and chunking strategy inference.

This module provides sophisticated tensor layout detection capabilities
that automatically determine optimal chunking strategies based on tensor
shapes and patterns, supporting various ML model architectures.

Components:
- TensorLayout: Enumeration of supported tensor layouts
- LayoutDetector: Main layout detection engine
- Layout-specific chunking strategy generation
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


class TensorLayout(Enum):
    """Supported tensor layout patterns for automatic detection."""
    # CNN layouts
    NCHW = "NCHW"  # [Batch, Channels, Height, Width]
    NHWC = "NHWC"  # [Batch, Height, Width, Channels] 
    CHW = "CHW"    # [Channels, Height, Width]
    HWC = "HWC"    # [Height, Width, Channels]
    
    # Transformer layouts
    NLC = "NLC"    # [Batch, Sequence, Features]
    NCL = "NCL"    # [Batch, Features, Sequence]
    LC = "LC"      # [Sequence, Features]
    CL = "CL"      # [Features, Sequence]
    
    # Multi-head attention layouts
    NLHD = "NLHD"  # [Batch, Sequence, Heads, HeadDim]
    NHLD = "NHLD"  # [Batch, Heads, Sequence, HeadDim]
    
    # Matrix layouts
    MM = "MM"      # [Rows, Cols] - generic matrix
    NC = "NC"      # [Batch, Features]
    
    # Vector layouts
    N = "N"        # [Batch]
    C = "C"        # [Features]
    
    # Unknown layout
    UNKNOWN = "UNKNOWN"


@dataclass
class LayoutCharacteristics:
    """Characteristics of a detected tensor layout."""
    layout: TensorLayout
    confidence: float  # 0.0 to 1.0 confidence in detection
    chunking_strategy: Dict[str, Any]
    parallelism_opportunities: Dict[str, int]
    memory_pattern: str  # "sequential", "spatial", "feature-wise"
    optimal_chunk_dims: List[int]  # Recommended dimensions to chunk along


@dataclass
class ChunkingRecommendation:
    """Recommendation for tensor chunking based on layout analysis."""
    tensor_dims: List[int]  # Original tensor dimensions
    block_dims: List[int]   # Recommended block dimensions
    chunk_dimensions: List[int]  # Dimensions along which to chunk
    parallelism_factor: int  # Expected parallelism from this chunking
    memory_efficiency: float  # Memory access efficiency (0.0-1.0)
    reasoning: str  # Human-readable explanation


class LayoutDetector:
    """Advanced tensor layout detection and chunking strategy inference."""
    
    def __init__(self):
        """Initialize layout detector with pattern recognition rules."""
        self.layout_patterns = self._initialize_layout_patterns()
        self.heuristics_weights = {
            "shape_ratio": 0.3,
            "dimension_count": 0.2, 
            "size_pattern": 0.3,
            "context_hints": 0.2
        }
    
    def detect_layout(self, tensor_shape: List[int], 
                     context_hints: Optional[Dict[str, Any]] = None) -> LayoutCharacteristics:
        """Detect tensor layout from shape and optional context.
        
        Args:
            tensor_shape: Tensor dimensions 
            context_hints: Optional hints like operation type, interface name, etc.
            
        Returns:
            LayoutCharacteristics with detected layout and recommendations
        """
        if not tensor_shape:
            return self._create_fallback_layout([1])
            
        logger.debug(f"Detecting layout for shape {tensor_shape}")
        
        # Apply pattern matching rules
        candidates = []
        
        for layout_type, pattern in self.layout_patterns.items():
            confidence = self._calculate_layout_confidence(
                tensor_shape, pattern, context_hints or {}
            )
            if confidence > 0.1:  # Minimum confidence threshold
                candidates.append((layout_type, confidence, pattern))
        
        # Select best candidate
        if candidates:
            best_layout, best_confidence, best_pattern = max(
                candidates, key=lambda x: x[1]
            )
            
            # Generate chunking strategy for detected layout
            chunking_strategy = self._generate_chunking_strategy(
                tensor_shape, best_layout, best_pattern
            )
            
            return LayoutCharacteristics(
                layout=best_layout,
                confidence=best_confidence,
                chunking_strategy=chunking_strategy,
                parallelism_opportunities=self._analyze_parallelism(
                    tensor_shape, best_layout
                ),
                memory_pattern=best_pattern.get("memory_pattern", "sequential"),
                optimal_chunk_dims=chunking_strategy.get("chunk_dimensions", [0])
            )
        else:
            # Fallback to heuristic detection
            return self._fallback_layout_detection(tensor_shape, context_hints or {})
    
    def recommend_chunking(self, tensor_shape: List[int],
                          interface_type: str = "input",
                          context_hints: Optional[Dict[str, Any]] = None) -> ChunkingRecommendation:
        """Recommend optimal chunking strategy based on layout detection.
        
        Args:
            tensor_shape: Input tensor shape
            interface_type: Type of interface ("input", "weight", "output")
            context_hints: Optional context information
            
        Returns:
            ChunkingRecommendation with specific chunking advice
        """
        # Detect layout first
        layout_chars = self.detect_layout(tensor_shape, context_hints)
        
        # Generate chunking recommendation
        if layout_chars.layout in [TensorLayout.NCHW, TensorLayout.CHW]:
            return self._recommend_cnn_chunking(tensor_shape, layout_chars, interface_type)
        elif layout_chars.layout in [TensorLayout.NHWC, TensorLayout.HWC]:
            return self._recommend_spatial_chunking(tensor_shape, layout_chars, interface_type)
        elif layout_chars.layout in [TensorLayout.NLC, TensorLayout.LC]:
            return self._recommend_sequence_chunking(tensor_shape, layout_chars, interface_type)
        elif layout_chars.layout in [TensorLayout.NLHD, TensorLayout.NHLD]:
            return self._recommend_attention_chunking(tensor_shape, layout_chars, interface_type)
        else:
            return self._recommend_generic_chunking(tensor_shape, layout_chars, interface_type)
    
    def _initialize_layout_patterns(self) -> Dict[TensorLayout, Dict[str, Any]]:
        """Initialize pattern matching rules for each layout type."""
        return {
            TensorLayout.NCHW: {
                "dimension_count": [4],
                "typical_ratios": {"dim1_larger_than_dim0": True, "spatial_dims": [2, 3]},
                "size_patterns": {"channels_reasonable": lambda s: 1 <= s[1] <= 2048},
                "memory_pattern": "channel-wise",
                "chunk_preference": [1]  # Chunk along channels
            },
            TensorLayout.NHWC: {
                "dimension_count": [4], 
                "typical_ratios": {"dim3_reasonable_channels": True, "spatial_dims": [1, 2]},
                "size_patterns": {"channels_last": lambda s: 1 <= s[-1] <= 2048},
                "memory_pattern": "spatial",
                "chunk_preference": [1, 2]  # Chunk along spatial dimensions
            },
            TensorLayout.CHW: {
                "dimension_count": [3],
                "typical_ratios": {"spatial_dims": [1, 2]},
                "size_patterns": {"channels_first": lambda s: 1 <= s[0] <= 2048},
                "memory_pattern": "channel-wise", 
                "chunk_preference": [0]  # Chunk along channels
            },
            TensorLayout.HWC: {
                "dimension_count": [3],
                "typical_ratios": {"spatial_dims": [0, 1]},
                "size_patterns": {"channels_last": lambda s: 1 <= s[-1] <= 2048},
                "memory_pattern": "spatial",
                "chunk_preference": [0, 1]  # Chunk along spatial dimensions
            },
            TensorLayout.NLC: {
                "dimension_count": [3],
                "typical_ratios": {"sequence_reasonable": True},
                "size_patterns": {"seq_len": lambda s: s[1] > s[2] or s[1] in [128, 256, 512, 1024]},
                "memory_pattern": "sequential",
                "chunk_preference": [1]  # Chunk along sequence
            },
            TensorLayout.NCL: {
                "dimension_count": [3],
                "typical_ratios": {"features_first": True},
                "size_patterns": {"features": lambda s: s[1] in [256, 512, 768, 1024]},
                "memory_pattern": "feature-wise",
                "chunk_preference": [1]  # Chunk along features
            },
            TensorLayout.LC: {
                "dimension_count": [2],
                "typical_ratios": {"sequence_features": True},
                "size_patterns": {"reasonable_2d": lambda s: max(s) > min(s)},
                "memory_pattern": "sequential",
                "chunk_preference": [0]  # Chunk along sequence
            },
            TensorLayout.NLHD: {
                "dimension_count": [4],
                "typical_ratios": {"attention_pattern": True},
                "size_patterns": {"head_dim": lambda s: s[3] in [32, 64, 128]},
                "memory_pattern": "attention",
                "chunk_preference": [1]  # Chunk along sequence
            },
            TensorLayout.MM: {
                "dimension_count": [2],
                "typical_ratios": {"matrix_like": True},
                "size_patterns": {"reasonable_matrix": lambda s: min(s) > 1},
                "memory_pattern": "sequential",
                "chunk_preference": [0]  # Chunk along rows
            },
            TensorLayout.NC: {
                "dimension_count": [2],
                "typical_ratios": {"batch_features": True},
                "size_patterns": {"small_batch": lambda s: s[0] < s[1]},
                "memory_pattern": "feature-wise",
                "chunk_preference": [0]  # Chunk along batch
            }
        }
    
    def _calculate_layout_confidence(self, tensor_shape: List[int], 
                                   pattern: Dict[str, Any],
                                   context_hints: Dict[str, Any]) -> float:
        """Calculate confidence score for a layout pattern match."""
        confidence = 0.0
        
        # Check dimension count
        if len(tensor_shape) in pattern.get("dimension_count", []):
            confidence += 0.4
        
        # Check size patterns
        size_patterns = pattern.get("size_patterns", {})
        for pattern_name, pattern_func in size_patterns.items():
            try:
                if pattern_func(tensor_shape):
                    confidence += 0.3
            except Exception:
                pass  # Pattern check failed
        
        # Check context hints
        context_boost = 0.0
        if "operation_type" in context_hints:
            op_type = context_hints["operation_type"].lower()
            if "conv" in op_type and pattern.get("memory_pattern") == "channel-wise":
                context_boost += 0.2
            elif "attention" in op_type and pattern.get("memory_pattern") == "attention":
                context_boost += 0.3
            elif "linear" in op_type and pattern.get("memory_pattern") == "feature-wise":
                context_boost += 0.2
        
        if "interface_name" in context_hints:
            name = context_hints["interface_name"].lower()
            if "weight" in name and len(tensor_shape) == 2:
                context_boost += 0.1
        
        confidence += context_boost
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _generate_chunking_strategy(self, tensor_shape: List[int],
                                  layout: TensorLayout,
                                  pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chunking strategy for detected layout."""
        chunk_dimensions = pattern.get("chunk_preference", [0])
        
        # Calculate recommended block dimensions
        block_dims = list(tensor_shape)
        
        for dim_idx in chunk_dimensions:
            if dim_idx < len(block_dims):
                # Use chunking heuristic based on dimension size
                original_size = block_dims[dim_idx]
                if original_size > 64:
                    block_dims[dim_idx] = min(16, original_size // 4)
                elif original_size > 16:
                    block_dims[dim_idx] = min(8, original_size // 2)
                else:
                    block_dims[dim_idx] = 1
        
        return {
            "layout": layout.value,
            "block_dims": block_dims,
            "chunk_dimensions": chunk_dimensions,
            "memory_pattern": pattern.get("memory_pattern", "sequential"),
            "parallelism_estimate": np.prod([tensor_shape[i] // block_dims[i] 
                                           for i in chunk_dimensions if i < len(tensor_shape)])
        }
    
    def _analyze_parallelism(self, tensor_shape: List[int], 
                           layout: TensorLayout) -> Dict[str, int]:
        """Analyze parallelism opportunities for detected layout."""
        opportunities = {}
        
        if layout in [TensorLayout.NCHW, TensorLayout.CHW]:
            opportunities["channel_parallelism"] = tensor_shape[0 if layout == TensorLayout.CHW else 1]
            if len(tensor_shape) >= 3:
                opportunities["spatial_parallelism"] = np.prod(tensor_shape[-2:])
        
        elif layout in [TensorLayout.NHWC, TensorLayout.HWC]:
            spatial_dims = tensor_shape[:-1] if layout == TensorLayout.HWC else tensor_shape[1:-1]
            opportunities["spatial_parallelism"] = np.prod(spatial_dims)
            opportunities["channel_parallelism"] = tensor_shape[-1]
        
        elif layout in [TensorLayout.NLC, TensorLayout.LC]:
            seq_idx = 0 if layout == TensorLayout.LC else 1
            opportunities["sequence_parallelism"] = tensor_shape[seq_idx]
            opportunities["feature_parallelism"] = tensor_shape[-1]
        
        else:
            opportunities["generic_parallelism"] = max(tensor_shape)
        
        return opportunities
    
    def _recommend_cnn_chunking(self, tensor_shape: List[int], 
                               layout_chars: LayoutCharacteristics,
                               interface_type: str) -> ChunkingRecommendation:
        """Recommend chunking for CNN-style tensors (NCHW/CHW)."""
        # Remove batch dimension if present
        if layout_chars.layout == TensorLayout.NCHW:
            tensor_dims = tensor_shape[1:]  # Remove batch
            channel_idx = 0
        else:  # CHW
            tensor_dims = tensor_shape
            channel_idx = 0
        
        block_dims = list(tensor_dims)
        
        # Chunk along channel dimension for CNN layers
        if tensor_dims[channel_idx] > 1:
            block_dims[channel_idx] = 1
            chunk_dims = [channel_idx]
            parallelism = tensor_dims[channel_idx]
            reasoning = f"CNN layout detected: chunking along channels for {parallelism}x parallelism"
        else:
            # Fallback to spatial chunking if single channel
            if len(tensor_dims) >= 3:
                block_dims[-2] = min(block_dims[-2], 16)  # Chunk height
                chunk_dims = [-2]
                parallelism = tensor_dims[-2] // block_dims[-2]
                reasoning = "Single channel CNN: chunking along spatial height"
            else:
                chunk_dims = [0]
                parallelism = 1
                reasoning = "Minimal CNN chunking strategy"
        
        return ChunkingRecommendation(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            chunk_dimensions=chunk_dims,
            parallelism_factor=parallelism,
            memory_efficiency=0.8,  # Good for channel-wise access
            reasoning=reasoning
        )
    
    def _recommend_spatial_chunking(self, tensor_shape: List[int],
                                   layout_chars: LayoutCharacteristics,
                                   interface_type: str) -> ChunkingRecommendation:
        """Recommend chunking for spatial-first tensors (NHWC/HWC)."""
        # Remove batch dimension if present
        if layout_chars.layout == TensorLayout.NHWC:
            tensor_dims = tensor_shape[1:]  # Remove batch
            spatial_dims = [0, 1]  # H, W
        else:  # HWC
            tensor_dims = tensor_shape
            spatial_dims = [0, 1]  # H, W
        
        block_dims = list(tensor_dims)
        
        # Chunk along spatial dimensions
        total_spatial = np.prod([tensor_dims[i] for i in spatial_dims])
        if total_spatial > 64:
            # Chunk spatial dimensions
            block_dims[0] = min(8, tensor_dims[0])  # Height chunks
            block_dims[1] = min(8, tensor_dims[1])  # Width chunks
            chunk_dims = spatial_dims
            parallelism = (tensor_dims[0] // block_dims[0]) * (tensor_dims[1] // block_dims[1])
            reasoning = f"Spatial layout: chunking {tensor_dims[0]}×{tensor_dims[1]} spatial for {parallelism}x parallelism"
        else:
            # Small spatial size - no chunking
            chunk_dims = []
            parallelism = 1
            reasoning = "Small spatial dimensions: no chunking needed"
        
        return ChunkingRecommendation(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            chunk_dimensions=chunk_dims,
            parallelism_factor=parallelism,
            memory_efficiency=0.9,  # Excellent for spatial locality
            reasoning=reasoning
        )
    
    def _recommend_sequence_chunking(self, tensor_shape: List[int],
                                    layout_chars: LayoutCharacteristics,
                                    interface_type: str) -> ChunkingRecommendation:
        """Recommend chunking for sequence tensors (NLC/LC)."""
        # Remove batch dimension if present  
        if layout_chars.layout == TensorLayout.NLC:
            tensor_dims = tensor_shape[1:]  # Remove batch
            seq_idx = 0
            feature_idx = 1
        else:  # LC
            tensor_dims = tensor_shape
            seq_idx = 0
            feature_idx = 1
        
        block_dims = list(tensor_dims)
        
        # Chunk along sequence dimension for transformer-style processing
        seq_len = tensor_dims[seq_idx]
        if seq_len > 1:
            block_dims[seq_idx] = 1  # Process one token at a time
            chunk_dims = [seq_idx]
            parallelism = seq_len
            reasoning = f"Sequence layout: chunking {seq_len} tokens for token-parallel processing"
        else:
            chunk_dims = []
            parallelism = 1
            reasoning = "Single token: no sequence chunking needed"
        
        return ChunkingRecommendation(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            chunk_dimensions=chunk_dims,
            parallelism_factor=parallelism,
            memory_efficiency=0.85,  # Good for sequential access
            reasoning=reasoning
        )
    
    def _recommend_attention_chunking(self, tensor_shape: List[int],
                                     layout_chars: LayoutCharacteristics,
                                     interface_type: str) -> ChunkingRecommendation:
        """Recommend chunking for multi-head attention tensors."""
        # Remove batch dimension
        tensor_dims = tensor_shape[1:]  # Remove batch
        
        if layout_chars.layout == TensorLayout.NLHD:
            seq_idx, head_idx, dim_idx = 0, 1, 2
        else:  # NHLD  
            head_idx, seq_idx, dim_idx = 0, 1, 2
        
        block_dims = list(tensor_dims)
        
        # Chunk along sequence dimension, preserve head structure
        seq_len = tensor_dims[seq_idx]
        if seq_len > 1:
            block_dims[seq_idx] = 1  # Process one token at a time across all heads
            chunk_dims = [seq_idx]
            parallelism = seq_len
            reasoning = f"Multi-head attention: chunking {seq_len} tokens across {tensor_dims[head_idx]} heads"
        else:
            chunk_dims = []
            parallelism = 1
            reasoning = "Single token attention: no chunking needed"
        
        return ChunkingRecommendation(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            chunk_dimensions=chunk_dims,
            parallelism_factor=parallelism,
            memory_efficiency=0.75,  # Moderate due to attention patterns
            reasoning=reasoning
        )
    
    def _recommend_generic_chunking(self, tensor_shape: List[int],
                                   layout_chars: LayoutCharacteristics,
                                   interface_type: str) -> ChunkingRecommendation:
        """Recommend chunking for generic/unknown tensor layouts."""
        tensor_dims = tensor_shape
        block_dims = list(tensor_dims)
        
        # Simple heuristic: chunk along largest dimension
        largest_dim_idx = np.argmax(tensor_dims)
        largest_dim_size = tensor_dims[largest_dim_idx]
        
        if largest_dim_size > 16:
            block_dims[largest_dim_idx] = min(8, largest_dim_size // 4)
            chunk_dims = [largest_dim_idx]
            parallelism = largest_dim_size // block_dims[largest_dim_idx]
            reasoning = f"Generic layout: chunking along largest dimension ({largest_dim_size}) for {parallelism}x parallelism"
        else:
            chunk_dims = []
            parallelism = 1
            reasoning = "Small tensor: no chunking recommended"
        
        return ChunkingRecommendation(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            chunk_dimensions=chunk_dims,
            parallelism_factor=parallelism,
            memory_efficiency=0.6,  # Conservative estimate
            reasoning=reasoning
        )
    
    def _fallback_layout_detection(self, tensor_shape: List[int],
                                  context_hints: Dict[str, Any]) -> LayoutCharacteristics:
        """Fallback layout detection using simple heuristics."""
        if len(tensor_shape) == 4:
            layout = TensorLayout.NCHW  # Default CNN layout
        elif len(tensor_shape) == 3:
            layout = TensorLayout.NLC   # Default sequence layout
        elif len(tensor_shape) == 2:
            layout = TensorLayout.MM    # Default matrix layout
        else:
            layout = TensorLayout.UNKNOWN
        
        return LayoutCharacteristics(
            layout=layout,
            confidence=0.3,  # Low confidence fallback
            chunking_strategy={"layout": layout.value, "block_dims": tensor_shape, "chunk_dimensions": [0]},
            parallelism_opportunities={"generic": max(tensor_shape) if tensor_shape else 1},
            memory_pattern="sequential",
            optimal_chunk_dims=[0]
        )
    
    def _create_fallback_layout(self, tensor_shape: List[int]) -> LayoutCharacteristics:
        """Create minimal fallback layout for edge cases."""
        return LayoutCharacteristics(
            layout=TensorLayout.UNKNOWN,
            confidence=0.1,
            chunking_strategy={"layout": "UNKNOWN", "block_dims": tensor_shape, "chunk_dimensions": []},
            parallelism_opportunities={},
            memory_pattern="sequential", 
            optimal_chunk_dims=[]
        )


def detect_tensor_layout(tensor_shape: List[int], 
                        context_hints: Optional[Dict[str, Any]] = None) -> LayoutCharacteristics:
    """Convenience function for tensor layout detection.
    
    Args:
        tensor_shape: Tensor dimensions
        context_hints: Optional context like operation type, interface name
        
    Returns:
        LayoutCharacteristics with detected layout and recommendations
    """
    detector = LayoutDetector()
    return detector.detect_layout(tensor_shape, context_hints)


def recommend_chunking_strategy(tensor_shape: List[int],
                               interface_type: str = "input",
                               context_hints: Optional[Dict[str, Any]] = None) -> ChunkingRecommendation:
    """Convenience function for chunking strategy recommendation.
    
    Args:
        tensor_shape: Input tensor shape
        interface_type: Type of interface ("input", "weight", "output")
        context_hints: Optional context information
        
    Returns:
        ChunkingRecommendation with specific chunking advice
    """
    detector = LayoutDetector()
    return detector.recommend_chunking(tensor_shape, interface_type, context_hints)