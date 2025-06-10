"""
Simple command-line interface for HWKG.

Provides clean CLI without complex orchestration layers.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .config import Config
from .data import HWKernel, GenerationResult
from .rtl_parser import parse_rtl_file
from .generators import HWCustomOpGenerator, RTLBackendGenerator, TestSuiteGenerator
from .errors import HWKGError, CompilerDataError


def create_hw_kernel(config: Config) -> HWKernel:
    """Create HWKernel from RTL file and compiler data."""
    # Parse RTL file
    if config.debug:
        print(f"Parsing RTL file: {config.rtl_file}")
    
    rtl_data = parse_rtl_file(config.rtl_file)
    
    if config.debug:
        print(f"Found module: {rtl_data.module_name}")
        print(f"Interfaces: {len(rtl_data.interfaces)}")
        print(f"Parameters: {len(rtl_data.parameters)}")
    
    # Load compiler data
    if config.debug:
        print(f"Loading compiler data: {config.compiler_data_file}")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("compiler_data", config.compiler_data_file)
        compiler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiler_module)
        
        # Get compiler data - try common attribute names
        if hasattr(compiler_module, 'compiler_data'):
            compiler_data = compiler_module.compiler_data
        elif hasattr(compiler_module, 'data'):
            compiler_data = compiler_module.data
        elif hasattr(compiler_module, 'config'):
            compiler_data = compiler_module.config
        else:
            # Use module dict as fallback
            compiler_data = {
                key: value for key, value in vars(compiler_module).items()
                if not key.startswith('_')
            }
        
    except Exception as e:
        raise CompilerDataError(f"Failed to load compiler data: {e}") from e
    
    # Create kernel object
    class_name = _generate_class_name(rtl_data.module_name)
    
    return HWKernel(
        name=rtl_data.module_name,
        class_name=class_name,
        interfaces=rtl_data.interfaces,
        rtl_parameters=rtl_data.parameters,
        source_file=config.rtl_file,
        compiler_data=compiler_data
    )


def generate_all(hw_kernel: HWKernel, config: Config) -> GenerationResult:
    """Generate all output files."""
    generators = [
        HWCustomOpGenerator(config.template_dir),
        RTLBackendGenerator(config.template_dir),
        TestSuiteGenerator(config.template_dir)
    ]
    
    result = GenerationResult(generated_files=[], success=True)
    
    for generator in generators:
        try:
            output_file = generator.generate(hw_kernel, config.output_dir)
            result.add_generated_file(output_file)
            
            if config.debug:
                print(f"Generated: {output_file}")
                
        except Exception as e:
            error_msg = f"Failed to generate {generator.__class__.__name__}: {e}"
            result.add_error(error_msg)
            
            if config.debug:
                print(f"Error: {error_msg}")
                import traceback
                traceback.print_exc()
    
    return result


def _generate_class_name(module_name: str) -> str:
    """Generate Python class name from module name."""
    # Convert snake_case or kebab-case to PascalCase
    parts = module_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hardware Kernel Generator - Simplified Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainsmith.tools.hw_kernel_gen_simple thresholding.sv compiler_data.py -o output/
  python -m brainsmith.tools.hw_kernel_gen_simple input.sv data.py -o generated/ --debug

This simplified HWKG eliminates enterprise bloat while preserving all functionality.
Generate HWCustomOp, RTL backend, and test suite files from SystemVerilog RTL.
        """
    )
    
    parser.add_argument(
        'rtl_file', 
        type=str, 
        help='SystemVerilog RTL file to process'
    )
    parser.add_argument(
        'compiler_data', 
        type=str, 
        help='Python file containing compiler data'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--template-dir', 
        type=str, 
        help='Custom template directory (optional)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.from_args(args)
        
        if config.debug:
            print("=== HWKG Simplified Implementation ===")
            print(f"RTL file: {config.rtl_file}")
            print(f"Compiler data: {config.compiler_data_file}")
            print(f"Output directory: {config.output_dir}")
            if config.template_dir:
                print(f"Template directory: {config.template_dir}")
            print()
        
        # Create hardware kernel representation
        hw_kernel = create_hw_kernel(config)
        
        if config.debug:
            print(f"Created kernel: {hw_kernel.name} → {hw_kernel.class_name}")
            print(f"Interfaces: {len(hw_kernel.interfaces)}")
            print(f"Parameters: {len(hw_kernel.rtl_parameters)}")
            print(f"Kernel type: {hw_kernel.kernel_type}")
            print(f"Complexity: {hw_kernel.kernel_complexity}")
            print()
        
        # Generate all outputs
        print("Generating files...")
        result = generate_all(hw_kernel, config)
        
        if result.success:
            print(f"✅ Successfully generated {len(result.generated_files)} files:")
            for file_path in result.generated_files:
                print(f"   📄 {file_path.relative_to(config.output_dir)}")
            
            if result.warnings:
                print(f"\n⚠️  {len(result.warnings)} warnings:")
                for warning in result.warnings:
                    print(f"   - {warning}")
        else:
            print(f"❌ Generation failed with {len(result.errors)} errors:")
            for error in result.errors:
                print(f"   - {error}")
            
            if result.generated_files:
                print(f"\n✅ {len(result.generated_files)} files were generated successfully:")
                for file_path in result.generated_files:
                    print(f"   📄 {file_path.relative_to(config.output_dir)}")
            
            sys.exit(1)
            
    except HWKGError as e:
        print(f"❌ HWKG Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()