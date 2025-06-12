# Expose the main HardwareKernelGenerator class and functions
from .hkg import HardwareKernelGenerator, HardwareKernelGeneratorError
from .rtl_parser import HWKernel, Port, Parameter, Interface, Pragma # Expose data structures

def generate_hw_kernel(rtl_file_path: str, compiler_data_path: str, output_dir: str, custom_doc_path: str = None):
    """
    Generate hardware kernel integration files.
    
    Args:
        rtl_file_path: Path to SystemVerilog RTL source file
        compiler_data_path: Path to Python file containing compiler data
        output_dir: Directory where generated files will be saved
        custom_doc_path: Optional path to custom documentation
        
    Returns:
        Dictionary containing paths to generated files
    """
    generator = HardwareKernelGenerator(
        rtl_file_path=rtl_file_path,
        compiler_data_path=compiler_data_path,
        output_dir=output_dir,
        custom_doc_path=custom_doc_path
    )
    return generator.run()

__all__ = [
    "HWKernel",
    "Port", 
    "Parameter",
    "Interface",
    "Pragma",
    "generate_hw_kernel",
    "HardwareKernelGenerator",
    "HardwareKernelGeneratorError"
]
