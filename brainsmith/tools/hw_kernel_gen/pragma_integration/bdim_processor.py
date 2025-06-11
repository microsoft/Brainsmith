"""
BDIM pragma processor for advanced mode.

Extracts and processes enhanced BDIM pragma information following
Interface-Wise Dataflow Modeling axioms for tensor_dims, block_dims, stream_dims.
"""

from typing import Dict, List, Any, Optional
from ..errors import BDimProcessingError


class BDimProcessor:
    """
    Enhanced BDIM pragma processor for advanced mode.
    
    Following Interface-Wise Dataflow Axiom 4: Pragma-to-Chunking Conversion
    and RTL Parser Axiom 4: Pragma-Driven Metadata.
    """
    
    def __init__(self):
        """Initialize BDIM processor."""
        self.debug = False
    
    def extract_bdim_metadata(self, hw_kernel) -> Dict[str, Any]:
        """
        Extract BDIM metadata from sophisticated RTL parser output.
        
        Args:
            hw_kernel: HWKernel object from sophisticated RTL parser
            
        Returns:
            Dict containing BDIM metadata with chunking strategies
            
        Following Interface-Wise Dataflow Axiom 2: Core Relationship
        tensor_dims → chunked into → num_blocks pieces of shape block_dims
        """
        try:
            bdim_metadata = {
                'chunking_strategies': {},
                'tensor_dims': {},
                'block_dims': {},
                'stream_dims': {}
            }
            
            # Process each interface for BDIM pragmas
            for interface_name, interface_obj in hw_kernel.interfaces.items():
                chunking_strategy = self._extract_interface_chunking(interface_obj)
                if chunking_strategy:
                    bdim_metadata['chunking_strategies'][interface_name] = chunking_strategy
            
            # Extract global tensor/block/stream dimension information
            self._extract_dimension_metadata(hw_kernel, bdim_metadata)
            
            return bdim_metadata if bdim_metadata['chunking_strategies'] else None
            
        except Exception as e:
            raise BDimProcessingError(f"Failed to extract BDIM metadata: {e}") from e
    
    def _extract_interface_chunking(self, interface_obj) -> Optional[Dict[str, Any]]:
        """
        Extract chunking strategy from interface BDIM pragmas.
        
        Following RTL Parser Axiom 4: BDIM pragma format:
        // @brainsmith BDIM <interface> <chunk_index> [<chunk_sizes>]
        """
        try:
            # Look for enhanced BDIM pragma in interface metadata
            enhanced_bdim = getattr(interface_obj, 'metadata', {}).get('enhanced_bdim')
            if enhanced_bdim:
                return {
                    'tensor_dims': self._infer_tensor_dims(interface_obj),
                    'block_dims': self._extract_block_dims(enhanced_bdim),
                    'stream_dims': self._extract_stream_dims(enhanced_bdim),
                    'strategy_type': enhanced_bdim.get('chunking_strategy_type', 'block_based'),
                    'chunk_index': enhanced_bdim.get('chunk_index', 0),
                    'chunk_sizes': enhanced_bdim.get('chunk_sizes', [16])
                }
            
            # Fallback: check pragmas directly
            if hasattr(interface_obj, 'pragmas'):
                for pragma in interface_obj.pragmas:
                    if hasattr(pragma, 'type') and 'BDIM' in str(pragma.type):
                        return self._parse_bdim_pragma(pragma)
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to extract chunking for interface {interface_obj.name}: {e}")
            return None
    
    def _parse_bdim_pragma(self, pragma) -> Dict[str, Any]:
        """
        Parse BDIM pragma content into chunking strategy.
        
        Following Interface-Wise Dataflow Axiom terminology.
        """
        content = getattr(pragma, 'content', '')
        if not content:
            return {}
        
        # Parse pragma content: BDIM <interface> <chunk_index> [<chunk_sizes>]
        parts = content.strip().split()
        if len(parts) < 3:
            return {}
        
        try:
            chunk_index = int(parts[2])
            chunk_sizes = [16]  # Default
            
            if len(parts) > 3:
                # Parse chunk sizes: "[16,32]" or "16"
                chunk_str = parts[3].strip('[]')
                if ',' in chunk_str:
                    chunk_sizes = [int(x.strip()) for x in chunk_str.split(',')]
                else:
                    chunk_sizes = [int(chunk_str)]
            
            return {
                'tensor_dims': [],  # To be inferred
                'block_dims': chunk_sizes,
                'stream_dims': [1],  # Default stream dimension
                'strategy_type': 'block_based',
                'chunk_index': chunk_index,
                'chunk_sizes': chunk_sizes
            }
            
        except (ValueError, IndexError) as e:
            if self.debug:
                print(f"Warning: Failed to parse BDIM pragma '{content}': {e}")
            return {}
    
    def _infer_tensor_dims(self, interface_obj) -> List[int]:
        """
        Infer tensor dimensions from interface characteristics.
        
        Following Interface-Wise Dataflow Axiom 1: Data Hierarchy
        Tensor → Block → Stream → Element
        """
        # Default tensor dimensions based on interface type and name patterns
        name = getattr(interface_obj, 'name', '').lower()
        
        if 'weight' in name or 'param' in name:
            # Weight interfaces typically have [OutChannels, InChannels] or similar
            return [256, 128]  # Default weight tensor shape
        elif any(pattern in name for pattern in ['input', 'activation', 's_axis']):
            # Input activation interfaces: [Channels, Height, Width] or [Length, Channels]
            return [128, 32, 32]  # Default activation tensor shape
        elif any(pattern in name for pattern in ['output', 'm_axis']):
            # Output activation interfaces
            return [256, 32, 32]  # Default output tensor shape
        else:
            # Generic interface
            return [128]  # Default 1D tensor
    
    def _extract_block_dims(self, enhanced_bdim: Dict[str, Any]) -> List[int]:
        """
        Extract block dimensions from enhanced BDIM metadata.
        
        Following Interface-Wise Dataflow Axiom 2: block_dims shape.
        """
        chunk_sizes = enhanced_bdim.get('chunk_sizes', [16])
        return chunk_sizes if isinstance(chunk_sizes, list) else [chunk_sizes]
    
    def _extract_stream_dims(self, enhanced_bdim: Dict[str, Any]) -> List[int]:
        """
        Extract stream dimensions from enhanced BDIM metadata.
        
        Following Interface-Wise Dataflow Axiom 6: Stream Relationships
        stream_dims relates to parallelism parameters (iPar, wPar).
        """
        # Stream dimensions typically based on parallelism
        # For now, use simple default based on block dimensions
        block_dims = self._extract_block_dims(enhanced_bdim)
        
        # Stream dimension is usually a factor of block dimension
        if block_dims:
            return [min(block_dims[0], 4)]  # Conservative stream parallelism
        return [1]
    
    def _extract_dimension_metadata(self, hw_kernel, bdim_metadata: Dict[str, Any]):
        """
        Extract global dimension metadata from kernel.
        
        Following Interface-Wise Dataflow Axiom 5: Parallelism Parameters
        iPar (input parallelism) and wPar (weight parallelism).
        """
        # Extract parallelism parameters from RTL parameters if available
        for param in hw_kernel.parameters:
            param_name = getattr(param, 'name', '').lower()
            param_value = getattr(param, 'default_value', None)
            
            if param_name in ['ipar', 'simd', 'input_parallelism']:
                bdim_metadata['parallelism'] = bdim_metadata.get('parallelism', {})
                bdim_metadata['parallelism']['iPar'] = param_value
            elif param_name in ['wpar', 'pe', 'weight_parallelism']:
                bdim_metadata['parallelism'] = bdim_metadata.get('parallelism', {})
                bdim_metadata['parallelism']['wPar'] = param_value
    
    def convert_to_chunking_strategy(self, bdim_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert BDIM metadata to chunking strategy for dataflow interface.
        
        Following Interface-Wise Dataflow Axiom 4: Pragma-to-Chunking Conversion.
        """
        chunking_strategies = {}
        
        for interface_name, chunking_info in bdim_metadata.get('chunking_strategies', {}).items():
            strategy = {
                'chunking_type': chunking_info.get('strategy_type', 'block_based'),
                'tensor_dims': chunking_info.get('tensor_dims', []),
                'block_dims': chunking_info.get('block_dims', []),
                'stream_dims': chunking_info.get('stream_dims', []),
                'chunk_index': chunking_info.get('chunk_index', 0)
            }
            chunking_strategies[interface_name] = strategy
        
        return chunking_strategies