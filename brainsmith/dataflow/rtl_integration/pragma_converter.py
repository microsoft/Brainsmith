"""
Pragma to Strategy Converter.

This module converts @brainsmith pragmas (BDIM, DATATYPE, WEIGHT) into 
ChunkingStrategy and DataTypeConstraint instances for use in DataflowInterface
creation. This bridges the gap between RTL pragma parsing and dataflow modeling.

Implements HWKG Axiom 4: Pragma-to-Chunking Conversion by providing systematic
conversion from BDIM pragmas to ChunkingStrategy instances.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from ..core.interface_metadata import DataTypeConstraint
from ..core.block_chunking import ChunkingStrategy, DefaultChunkingStrategy, IndexBasedChunkingStrategy, FullTensorChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class PragmaConversionResult:
    """Result of pragma conversion operation."""
    strategy: Optional[ChunkingStrategy] = None
    constraint: Optional[DataTypeConstraint] = None
    success: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PragmaToStrategyConverter:
    """
    Convert @brainsmith pragmas to ChunkingStrategy instances.
    
    This class provides systematic conversion from RTL pragmas to dataflow
    modeling constructs, enabling automatic translation of user intentions
    expressed in RTL comments into computational strategies.
    
    Supported Pragmas:
    - BDIM: Block dimension chunking specifications
    - DATATYPE: Interface datatype constraints  
    - WEIGHT: Weight interface specifications
    """
    
    def __init__(self):
        """Initialize pragma to strategy converter."""
        pass
    
    def convert_bdim_pragma(self, pragma) -> Optional[ChunkingStrategy]:
        """
        Convert BDIM pragma to appropriate chunking strategy.
        
        Supports both legacy and enhanced BDIM pragma formats:
        - Legacy: @brainsmith BDIM <interface_name> <dim1_expr> <dim2_expr> ... 
        - Enhanced: @brainsmith BDIM <interface_name> <chunk_index> [<chunk_sizes>]
        
        Args:
            pragma: BDimPragma instance from RTL parser
            
        Returns:
            ChunkingStrategy instance or None if conversion fails
        """
        try:
            if not hasattr(pragma, 'parsed_data') or not pragma.parsed_data:
                logger.warning(f"BDIM pragma at line {pragma.line_number} has no parsed_data")
                return None
                
            # Determine pragma format from parsed data
            format_type = pragma.parsed_data.get('format', 'legacy')
            
            if format_type == 'enhanced':
                return self._convert_enhanced_bdim(pragma)
            else:
                return self._convert_legacy_bdim(pragma)
                
        except Exception as e:
            logger.error(f"Failed to convert BDIM pragma: {str(e)}")
            return None
    
    def _convert_enhanced_bdim(self, pragma) -> Optional[ChunkingStrategy]:
        """
        Convert enhanced BDIM pragma format.
        
        Enhanced format: @brainsmith BDIM <interface_name> <chunk_index> [<chunk_sizes>]
        Example: @brainsmith BDIM in0_V_data_V -1 [16]
        
        Args:
            pragma: BDimPragma with enhanced format
            
        Returns:
            IndexBasedChunkingStrategy or FullTensorChunkingStrategy
        """
        try:
            chunk_index = pragma.parsed_data.get('chunk_index')
            chunk_sizes = pragma.parsed_data.get('chunk_sizes', [])
            
            if chunk_index is None:
                logger.warning(f"Enhanced BDIM pragma missing chunk_index at line {pragma.line_number}")
                return None
            
            # Convert to IndexBasedChunkingStrategy
            if chunk_index == -1:
                # Last dimension chunking
                if chunk_sizes:
                    return IndexBasedChunkingStrategy(
                        start_index=-1,
                        shape=chunk_sizes
                    )
                else:
                    # Full tensor processing
                    return FullTensorChunkingStrategy()
            else:
                # Specific dimension chunking
                return IndexBasedChunkingStrategy(
                    start_index=chunk_index,
                    shape=chunk_sizes if chunk_sizes else [1]
                )
                
        except Exception as e:
            logger.error(f"Failed to convert enhanced BDIM pragma: {str(e)}")
            return None
    
    def _convert_legacy_bdim(self, pragma) -> Optional[ChunkingStrategy]:
        """
        Convert legacy BDIM pragma format.
        
        Legacy format: @brainsmith BDIM <interface_name> <dim1_expr> <dim2_expr> ...
        Example: @brainsmith BDIM in0 PE*CHANNELS 1
        
        Args:
            pragma: BDimPragma with legacy format
            
        Returns:
            DefaultChunkingStrategy with parsed dimensions
        """
        try:
            dimension_expressions = pragma.parsed_data.get('dimension_expressions', [])
            
            if not dimension_expressions:
                logger.warning(f"Legacy BDIM pragma missing dimensions at line {pragma.line_number}")
                return DefaultChunkingStrategy()
            
            # For legacy format, create a default strategy with the expressions
            # The actual evaluation would happen at runtime with proper context
            return DefaultChunkingStrategy(
                block_dimensions=dimension_expressions
            )
            
        except Exception as e:
            logger.error(f"Failed to convert legacy BDIM pragma: {str(e)}")
            return DefaultChunkingStrategy()
    
    def convert_datatype_pragma(self, pragma) -> Optional[DataTypeConstraint]:
        """
        Convert DATATYPE pragma to constraint.
        
        DATATYPE format: @brainsmith DATATYPE <interface_name> <base_types> <min_bits> <max_bits>
        Example: @brainsmith DATATYPE in0_V_data_V INT,UINT 1 16
        
        Args:
            pragma: DatatypePragma instance from RTL parser
            
        Returns:
            DataTypeConstraint instance or None if conversion fails
        """
        try:
            if not hasattr(pragma, 'parsed_data') or not pragma.parsed_data:
                logger.warning(f"DATATYPE pragma at line {pragma.line_number} has no parsed_data")
                return None
                
            base_types = pragma.parsed_data.get('base_types', [])
            min_bitwidth = pragma.parsed_data.get('min_bitwidth', 1)
            max_bitwidth = pragma.parsed_data.get('max_bitwidth', 32)
            
            if not base_types:
                logger.warning(f"DATATYPE pragma missing base_types at line {pragma.line_number}")
                return None
            
            # Convert to DataTypeConstraint
            # Use first base type as default, full list as allowed types
            default_type = base_types[0]
            signed = 'INT' in base_types or 'FIXED' in base_types
            
            # Create FINN-compatible type string
            finn_type = f"{default_type}{max_bitwidth}"
            
            constraint = DataTypeConstraint(
                allowed_types=base_types,
                min_bits=min_bitwidth,
                max_bits=max_bitwidth,
                signed=signed,
                finn_type=finn_type
            )
            
            logger.debug(f"Created DataTypeConstraint from DATATYPE pragma: {constraint}")
            return constraint
            
        except Exception as e:
            logger.error(f"Failed to convert DATATYPE pragma: {str(e)}")
            return None
    
    def convert_weight_pragma(self, pragma) -> Optional[Dict[str, Any]]:
        """
        Convert WEIGHT pragma to interface metadata.
        
        WEIGHT format: @brainsmith WEIGHT <interface_name>
        Example: @brainsmith WEIGHT weights_V_data_V
        
        Args:
            pragma: WeightPragma instance from RTL parser
            
        Returns:
            Dict with weight interface metadata or None if conversion fails
        """
        try:
            if not hasattr(pragma, 'parsed_data') or not pragma.parsed_data:
                logger.warning(f"WEIGHT pragma at line {pragma.line_number} has no parsed_data")
                return None
                
            interface_name = pragma.parsed_data.get('interface_name')
            
            if not interface_name:
                logger.warning(f"WEIGHT pragma missing interface_name at line {pragma.line_number}")
                return None
            
            # Create weight interface metadata
            weight_metadata = {
                'interface_type': 'WEIGHT',
                'weight_source': 'external',  # Default to external weight loading
                'weight_format': 'default',   # Use default weight format
                'interface_name': interface_name,
                'pragma_line': pragma.line_number
            }
            
            logger.debug(f"Created weight metadata from WEIGHT pragma: {weight_metadata}")
            return weight_metadata
            
        except Exception as e:
            logger.error(f"Failed to convert WEIGHT pragma: {str(e)}")
            return None
    
    def convert_all_pragmas(self, pragmas: List) -> Dict[str, Any]:
        """
        Convert all pragmas in a list to their respective strategies/constraints.
        
        Args:
            pragmas: List of pragma objects
            
        Returns:
            Dict containing converted strategies and constraints by interface name
        """
        results = {
            'chunking_strategies': {},
            'datatype_constraints': {},
            'weight_metadata': {},
            'conversion_errors': [],
            'conversion_warnings': []
        }
        
        for pragma in pragmas:
            try:
                if hasattr(pragma, 'type') and pragma.type:
                    pragma_type = pragma.type.name if hasattr(pragma.type, 'name') else str(pragma.type)
                    interface_name = self._extract_interface_name(pragma)
                    
                    if pragma_type == 'BDIM':
                        strategy = self.convert_bdim_pragma(pragma)
                        if strategy and interface_name:
                            results['chunking_strategies'][interface_name] = strategy
                            
                    elif pragma_type == 'DATATYPE':
                        constraint = self.convert_datatype_pragma(pragma)
                        if constraint and interface_name:
                            results['datatype_constraints'][interface_name] = constraint
                            
                    elif pragma_type == 'WEIGHT':
                        metadata = self.convert_weight_pragma(pragma)
                        if metadata and interface_name:
                            results['weight_metadata'][interface_name] = metadata
                    
                    else:
                        results['conversion_warnings'].append(
                            f"Unknown pragma type: {pragma_type} at line {pragma.line_number}"
                        )
                        
            except Exception as e:
                error_msg = f"Error converting pragma at line {pragma.line_number}: {str(e)}"
                logger.error(error_msg)
                results['conversion_errors'].append(error_msg)
        
        return results
    
    def _extract_interface_name(self, pragma) -> Optional[str]:
        """
        Extract interface name from pragma parsed_data.
        
        Args:
            pragma: Pragma object with parsed_data
            
        Returns:
            Interface name or None if not found
        """
        if hasattr(pragma, 'parsed_data') and pragma.parsed_data:
            return pragma.parsed_data.get('interface_name')
        return None
    
    def handle_both_formats(self, bdim_pragmas: List) -> Dict[str, ChunkingStrategy]:
        """
        Handle both enhanced and legacy BDIM pragma formats.
        
        This method provides backwards compatibility while supporting
        the enhanced pragma format for better chunking specifications.
        
        Args:
            bdim_pragmas: List of BDIM pragma objects
            
        Returns:
            Dict mapping interface names to ChunkingStrategy instances
        """
        strategies = {}
        
        for pragma in bdim_pragmas:
            if hasattr(pragma, 'type') and pragma.type.name == 'BDIM':
                interface_name = self._extract_interface_name(pragma)
                strategy = self.convert_bdim_pragma(pragma)
                
                if interface_name and strategy:
                    strategies[interface_name] = strategy
                    
                    format_type = pragma.parsed_data.get('format', 'legacy')
                    logger.info(f"Converted {format_type} BDIM pragma for interface {interface_name}")
        
        return strategies


def create_pragma_converter() -> PragmaToStrategyConverter:
    """
    Factory function for creating PragmaToStrategyConverter instances.
    
    Returns:
        PragmaToStrategyConverter: Configured converter instance
    """
    return PragmaToStrategyConverter()