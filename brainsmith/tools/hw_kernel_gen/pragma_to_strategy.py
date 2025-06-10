"""
HWKG Pragma to Chunking Strategy Converter

This module converts parsed RTL pragmas to chunking strategies for the AutoHWCustomOp
per-interface strategy pattern. This enables clean separation between HWKG (pragma-aware)
and dataflow (computational) layers.

Supports both BDIM (block dimension) pragmas and legacy TDIM pragmas for backward compatibility.
"""

from typing import Dict, Any, Optional
from brainsmith.dataflow.core.tensor_chunking import (
    ChunkingStrategy, default_chunking, index_chunking, last_dim_chunking, 
    spatial_chunking, FullTensorChunkingStrategy
)


class PragmaToStrategyConverter:
    """Convert parsed RTL pragmas to chunking strategies."""
    
    def convert_bdim_pragma(self, pragma_data: Dict[str, Any]) -> ChunkingStrategy:
        """
        Convert BDIM pragma to appropriate chunking strategy.
        
        Args:
            pragma_data: Parsed pragma data containing type and parameters
            
        Returns:
            ChunkingStrategy object for the interface
        """
        pragma_type = pragma_data.get('type', 'default')
        
        if pragma_type == 'index':
            return index_chunking(
                pragma_data['start_index'], 
                pragma_data['shape']
            )
        elif pragma_type == 'spatial':
            return spatial_chunking(
                pragma_data['height'],
                pragma_data['width'] 
            )
        elif pragma_type == 'last_dim':
            return last_dim_chunking(pragma_data['chunk_size'])
        elif pragma_type == 'none' or pragma_type == 'full':
            return FullTensorChunkingStrategy()
        else:
            return default_chunking()
    
    def parse_enhanced_bdim_pragma(self, pragma_string: str) -> Dict[str, Any]:
        """
        Parse enhanced BDIM pragma string.
        
        Supported formats:
        - "@brainsmith BDIM in0_V_data_V -1 [16]" -> index-based chunking
        - "@brainsmith BDIM weights spatial 8x8" -> spatial chunking
        - "@brainsmith BDIM bias none" -> no chunking
        - "@brainsmith BDIM input last_dim 32" -> last dimension chunking
        
        Args:
            pragma_string: Raw pragma string from RTL
            
        Returns:
            Dict with parsed pragma data
        """
        parts = pragma_string.strip().split()
        
        if len(parts) < 4 or parts[0] != "@brainsmith" or (parts[1] != "BDIM" and parts[1] != "TDIM"):
            raise ValueError(f"Invalid BDIM/TDIM pragma format: {pragma_string}")
        
        interface_name = parts[2]
        pragma_type = parts[3]
        
        if pragma_type == "none" or pragma_type == "full":
            return {
                'interface_name': interface_name,
                'type': 'none'
            }
        elif pragma_type == "default":
            return {
                'interface_name': interface_name,
                'type': 'default'
            }
        elif pragma_type.isdigit() or pragma_type.startswith('-'):
            # Format: "@brainsmith TDIM in0_V_data_V -1 [16]"
            start_index = int(pragma_type)
            
            if len(parts) < 5:
                raise ValueError(f"Index-based TDIM pragma missing shape: {pragma_string}")
            
            # Parse shape specification
            shape_spec = ' '.join(parts[4:])
            shape = self._parse_shape_specification(shape_spec)
            
            return {
                'interface_name': interface_name,
                'type': 'index',
                'start_index': start_index,
                'shape': shape
            }
        elif pragma_type == "spatial":
            # Format: "@brainsmith TDIM weights spatial 8x8"
            if len(parts) < 5:
                raise ValueError(f"Spatial TDIM pragma missing dimensions: {pragma_string}")
            
            spatial_spec = parts[4]
            if 'x' in spatial_spec:
                height, width = map(int, spatial_spec.split('x'))
            else:
                # Square spatial chunking
                height = width = int(spatial_spec)
            
            return {
                'interface_name': interface_name,
                'type': 'spatial',
                'height': height,
                'width': width
            }
        elif pragma_type == "last_dim":
            # Format: "@brainsmith TDIM input last_dim 32"
            if len(parts) < 5:
                raise ValueError(f"Last dimension TDIM pragma missing chunk size: {pragma_string}")
            
            chunk_size = int(parts[4])
            
            return {
                'interface_name': interface_name,
                'type': 'last_dim',
                'chunk_size': chunk_size
            }
        else:
            raise ValueError(f"Unknown TDIM pragma type '{pragma_type}': {pragma_string}")
    
    def _parse_shape_specification(self, shape_spec: str) -> list:
        """
        Parse shape specification from pragma.
        
        Supported formats:
        - "[16]" -> [16]
        - "[16, 32]" -> [16, 32]
        - "[:]" -> [":"]
        - "16" -> [16]
        
        Args:
            shape_spec: Shape specification string
            
        Returns:
            List of shape values
        """
        shape_spec = shape_spec.strip()
        
        if shape_spec.startswith('[') and shape_spec.endswith(']'):
            # List format: [16, 32] or [:]
            inner = shape_spec[1:-1].strip()
            if inner == ':':
                return [':']
            elif ',' in inner:
                return [int(x.strip()) if x.strip() != ':' else ':' for x in inner.split(',')]
            else:
                return [int(inner) if inner != ':' else ':']
        else:
            # Simple format: 16
            return [int(shape_spec)]
    
    def create_strategy_from_pragma_dict(self, interface_name: str, pragma_dict: Dict[str, Any]) -> ChunkingStrategy:
        """
        Create chunking strategy from a pragma dictionary.
        
        This is a convenience method for direct pragma dictionary conversion.
        
        Args:
            interface_name: Name of the interface
            pragma_dict: Dictionary containing pragma configuration
            
        Returns:
            ChunkingStrategy object
        """
        if 'enhanced_bdim' in pragma_dict:
            return self.convert_bdim_pragma(pragma_dict['enhanced_bdim'])
        elif 'enhanced_tdim' in pragma_dict:
            # Backward compatibility for legacy enhanced_tdim
            import warnings
            warnings.warn(
                "enhanced_tdim pragma dictionary key is deprecated. Use enhanced_bdim instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self.convert_bdim_pragma(pragma_dict['enhanced_tdim'])
        else:
            # No specific pragma configuration - use default
            return default_chunking()
    
    def create_index_chunking_strategy(self, chunk_index: int, chunk_sizes: list) -> ChunkingStrategy:
        """
        Create an index-based chunking strategy directly.
        
        Args:
            chunk_index: Index to chunk (-1 for last dimension)
            chunk_sizes: List of chunk sizes
            
        Returns:
            IndexBasedChunkingStrategy object
        """
        return index_chunking(chunk_index, chunk_sizes)
    
    def create_spatial_chunking_strategy(self, layout: str, streaming_dim: str) -> ChunkingStrategy:
        """
        Create a spatial chunking strategy directly.
        
        Args:
            layout: Tensor layout (e.g., "NCHW", "CHW")
            streaming_dim: Dimension to stream on (e.g., "width", "height")
            
        Returns:
            SpatialChunkingStrategy object
        """
        return spatial_chunking(layout, streaming_dim)
    
    # Backward compatibility methods for legacy TDIM pragma support
    def convert_tdim_pragma(self, pragma_data: Dict[str, Any]) -> ChunkingStrategy:
        """
        Backward compatibility method for TDIM pragma conversion.
        
        DEPRECATED: Use convert_bdim_pragma() instead.
        """
        import warnings
        warnings.warn(
            "convert_tdim_pragma() is deprecated. Use convert_bdim_pragma() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.convert_bdim_pragma(pragma_data)
    
    def parse_enhanced_tdim_pragma(self, pragma_string: str) -> Dict[str, Any]:
        """
        Backward compatibility method for TDIM pragma parsing.
        
        DEPRECATED: Use parse_enhanced_bdim_pragma() instead.
        """
        import warnings
        warnings.warn(
            "parse_enhanced_tdim_pragma() is deprecated. Use parse_enhanced_bdim_pragma() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.parse_enhanced_bdim_pragma(pragma_string.replace("TDIM", "BDIM"))
    
    def create_last_dim_chunking_strategy(self, chunk_size: int) -> ChunkingStrategy:
        """
        Create a last dimension chunking strategy directly.
        
        Args:
            chunk_size: Size of chunks for last dimension
            
        Returns:
            IndexBasedChunkingStrategy configured for last dimension
        """
        return last_dim_chunking(chunk_size)
    
    def convert_pragma_to_strategy(self, pragma_string: str) -> ChunkingStrategy:
        """
        Convert a pragma string directly to a chunking strategy.
        
        Args:
            pragma_string: Raw pragma string from RTL
            
        Returns:
            ChunkingStrategy object
        """
        parsed = self.parse_enhanced_bdim_pragma(pragma_string)
        return self.convert_bdim_pragma(parsed)


def convert_pragmas_to_strategies(pragmas: Dict[str, Dict[str, Any]]) -> Dict[str, ChunkingStrategy]:
    """
    Convert a collection of interface pragmas to chunking strategies.
    
    Args:
        pragmas: Dict mapping interface names to pragma configurations
        
    Returns:
        Dict mapping interface names to ChunkingStrategy objects
    """
    converter = PragmaToStrategyConverter()
    strategies = {}
    
    for interface_name, pragma_config in pragmas.items():
        strategies[interface_name] = converter.create_strategy_from_pragma_dict(
            interface_name, pragma_config
        )
    
    return strategies


def generate_strategy_code(strategy: ChunkingStrategy) -> str:
    """
    Generate Python code string for a chunking strategy object.
    
    This is used in template generation to create the strategy constructor calls.
    
    Args:
        strategy: ChunkingStrategy object
        
    Returns:
        Python code string for creating the strategy
    """
    from brainsmith.dataflow.core.tensor_chunking import (
        DefaultChunkingStrategy, IndexBasedChunkingStrategy, FullTensorChunkingStrategy
    )
    
    if isinstance(strategy, DefaultChunkingStrategy):
        return "default_chunking()"
    elif isinstance(strategy, IndexBasedChunkingStrategy):
        return f"index_chunking({strategy.start_index}, {strategy.shape})"
    elif isinstance(strategy, FullTensorChunkingStrategy):
        return "FullTensorChunkingStrategy()"
    else:
        # Fallback to default
        return "default_chunking()"


# Example usage for HWKG integration:
def example_hwkg_integration():
    """Example of how HWKG would use this converter."""
    
    # 1. HWKG parses RTL pragmas
    rtl_pragmas = [
        "@brainsmith TDIM in0_V_data_V -1 [16]",
        "@brainsmith TDIM weights spatial 8x8", 
        "@brainsmith TDIM bias none"
    ]
    
    # 2. Convert pragmas to strategy configurations
    converter = PragmaToStrategyConverter()
    pragma_configs = {}
    
    for pragma_str in rtl_pragmas:
        parsed = converter.parse_enhanced_bdim_pragma(pragma_str)
        interface_name = parsed['interface_name']
        pragma_configs[interface_name] = {'enhanced_bdim': parsed}
    
    # 3. Generate chunking strategies
    strategies = convert_pragmas_to_strategies(pragma_configs)
    
    # 4. Generate template code
    template_strategies = {}
    for interface_name, strategy in strategies.items():
        template_strategies[interface_name] = generate_strategy_code(strategy)
    
    return template_strategies


if __name__ == "__main__":
    # Demo the converter
    print("=== HWKG Pragma to Strategy Converter Demo ===")
    
    strategies = example_hwkg_integration()
    for interface_name, strategy_code in strategies.items():
        print(f"{interface_name}: {strategy_code}")