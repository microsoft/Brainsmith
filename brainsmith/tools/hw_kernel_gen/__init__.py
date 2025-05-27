# Expose the main HardwareKernelGenerator class and potentially errors/data structures
# from .hkg import HardwareKernelGenerator, HardwareKernelGeneratorError
from .rtl_parser import HWKernel, Port, Parameter, Interface, Pragma # Expose data structures

__all__ = [
    "HWKernel",
    "Port",
    "Parameter",
    "Interface",
    "Pragma",
]
