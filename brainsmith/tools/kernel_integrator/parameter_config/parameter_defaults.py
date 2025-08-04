"""
Parameter whitelist and default value configuration for RTL to HWCustomOp generation.

This module defines which RTL parameters are allowed to have default values
when generating AutoHWCustomOp subclasses. Only whitelisted parameters can
have defaults; all others must be provided by FINN at instantiation time.
"""

from typing import Dict, Set

# Whitelisted parameters that can have default values
# These are common architectural parameters in hardware accelerators
WHITELISTED_DEFAULTS: Set[str] = {
    # Parallelism parameters
    "PE",           # Processing Elements
    "SIMD",         # Single Instruction Multiple Data width
    
    # Memory/buffer parameters  
    "DEPTH",        # Buffer/FIFO depth
    "MEM_DEPTH",    # Memory depth
    "FIFO_DEPTH",   # FIFO specific depth
    
    # Bit width parameters
    "WIDTH",        # Generic data width
    "DATA_WIDTH",   # Data bus width
    "ADDR_WIDTH",   # Address bus width
    
    # Architecture parameters
    "STAGES",       # Pipeline stages
    "LATENCY",      # Operation latency
    "II",           # Initiation Interval
    
    # Quantization parameters
    "QM",           # Quantization method
    "BIAS",         # Bias enable/value
    
    # Common defaults for testing
    "DEBUG",        # Debug mode enable
    "TEST_MODE",    # Test mode enable
}

# Default values for whitelisted parameters
# Used when no default is specified in RTL
PARAMETER_DEFAULTS: Dict[str, int] = {
    "PE": 1,
    "SIMD": 1,
    "DEPTH": 512,
    "MEM_DEPTH": 512,
    "FIFO_DEPTH": 32,
    "WIDTH": 32,
    "DATA_WIDTH": 32,
    "ADDR_WIDTH": 16,
    "STAGES": 1,
    "LATENCY": 1,
    "II": 1,
    "QM": 0,
    "BIAS": 0,
    "DEBUG": 0,
    "TEST_MODE": 0,
}

def is_parameter_whitelisted(param_name: str) -> bool:
    """Check if a parameter is whitelisted for having default values."""
    return param_name in WHITELISTED_DEFAULTS

def get_default_value(param_name: str) -> int:
    """Get the default value for a whitelisted parameter."""
    return PARAMETER_DEFAULTS.get(param_name, 1)  # Default to 1 if not specified

def validate_parameter_default(param_name: str, has_default: bool) -> bool:
    """
    Validate if a parameter can have a default value.
    
    Args:
        param_name: Name of the parameter
        has_default: Whether the parameter has a default in RTL
        
    Returns:
        True if valid (whitelisted or no default), False otherwise
    """
    if has_default and not is_parameter_whitelisted(param_name):
        return False
    return True