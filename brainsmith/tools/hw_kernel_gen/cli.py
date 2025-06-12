"""
Simple CLI for HWKG.

Radically simplified from the over-engineered 453-line version.
Single generation path: parse RTL → generate templates → done.
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .rtl_parser import RTLParser
from .generators import HWCustomOpGenerator, RTLBackendGenerator, TestSuiteGenerator


def load_compiler_data(compiler_data_file: Path) -> dict:
    """Load compiler data from Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("compiler_data", compiler_data_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load compiler data from {compiler_data_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Try common attribute names
    for attr in ['compiler_data', 'data', 'config']:
        if hasattr(module, attr):
            return getattr(module, attr)
    
    # Fallback: use all non-private attributes
    return {k: v for k, v in vars(module).items() if not k.startswith('_')}


def generate_files(config: Config) -> list:
    """Generate all files from RTL and compiler data."""
    # Parse RTL file
    parser = RTLParser()
    parsed_data = parser.parse_file(str(config.rtl_file))
    
    if config.debug:
        print(f"Parsed module: {parsed_data.name}")
        print(f"Interfaces: {len(parsed_data.interfaces)}")
        print(f"Parameters: {len(parsed_data.parameters)}")
    
    # Load compiler data (currently unused but kept for compatibility)
    compiler_data = load_compiler_data(config.compiler_data_file)
    
    if config.debug:
        print(f"Loaded compiler data with {len(compiler_data)} entries")
    
    # Generate all files
    generators = [
        HWCustomOpGenerator(),
        RTLBackendGenerator(), 
        TestSuiteGenerator()
    ]
    
    generated_files = []
    for generator in generators:
        try:
            output_file = generator.generate(parsed_data, config.output_dir)
            generated_files.append(output_file)
            
            if config.debug:
                print(f"Generated: {output_file.name}")
                
        except Exception as e:
            print(f"Warning: Failed to generate {generator.__class__.__name__}: {e}")
            if config.debug:
                import traceback
                traceback.print_exc()
    
    return generated_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate FINN components from SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/ --debug
        """
    )
    
    parser.add_argument('rtl_file', help='SystemVerilog RTL file to process')
    parser.add_argument('compiler_data', help='Python file containing compiler data')
    parser.add_argument('-o', '--output', required=True, help='Output directory for generated files')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.from_args(args)
        
        if config.debug:
            print("=== Hardware Kernel Generator ===")
            print(f"RTL file: {config.rtl_file}")
            print(f"Compiler data: {config.compiler_data_file}")
            print(f"Output directory: {config.output_dir}")
            print()
        
        # Generate files
        generated_files = generate_files(config)
        
        # Report results
        if generated_files:
            print(f"✅ Successfully generated {len(generated_files)} files:")
            for file_path in generated_files:
                print(f"   📄 {file_path.relative_to(config.output_dir)}")
        else:
            print("❌ No files were generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()