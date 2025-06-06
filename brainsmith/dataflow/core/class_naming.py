"""
Class naming utilities for auto-generated hardware kernel classes.

This module provides utilities for converting kernel names to proper CamelCase
class names, fixing the issue where "thresholding_axi" becomes "AutoThresholdingaxi"
instead of the correct "AutoThresholdingAxi".
"""


def generate_class_name(kernel_name: str, prefix: str = "Auto") -> str:
    """
    Convert kernel_name to proper CamelCase class name.
    
    Examples:
        thresholding_axi -> AutoThresholdingAxi
        conv_layer -> AutoConvLayer
        batch_norm -> AutoBatchNorm
        my_custom_kernel -> AutoMyCustomKernel
    
    Args:
        kernel_name: Underscore-separated kernel name
        prefix: Class name prefix (default: "Auto")
        
    Returns:
        Properly formatted CamelCase class name
    """
    # Split on underscores and capitalize each part
    parts = kernel_name.split('_')
    camel_case = ''.join(word.capitalize() for word in parts)
    return f"{prefix}{camel_case}"


def generate_test_class_name(kernel_name: str) -> str:
    """
    Generate test class name for a kernel.
    
    Examples:
        thresholding_axi -> TestAutoThresholdingAxi
        conv_layer -> TestAutoConvLayer
    
    Args:
        kernel_name: Underscore-separated kernel name
        
    Returns:
        Test class name in proper CamelCase
    """
    base_class_name = generate_class_name(kernel_name)
    return f"Test{base_class_name}"


def generate_backend_class_name(kernel_name: str) -> str:
    """
    Generate RTL backend class name for a kernel.
    
    Examples:
        thresholding_axi -> AutoThresholdingAxiRTLBackend
        conv_layer -> AutoConvLayerRTLBackend
    
    Args:
        kernel_name: Underscore-separated kernel name
        
    Returns:
        RTL backend class name in proper CamelCase
    """
    base_class_name = generate_class_name(kernel_name)
    return f"{base_class_name}RTLBackend"