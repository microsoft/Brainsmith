"""
CLI for Kernel Integrator.

Modular CLI using the current infrastructure with KernelIntegrator and GeneratorManager.
Single generation path: parse RTL → generate all templates → write files → done.
"""

import argparse
import sys
import time
from pathlib import Path

from .rtl_parser.parser import RTLParser
from .kernel_integrator import KernelIntegrator


def create_parser() -> argparse.ArgumentParser:
    """Create simplified argument parser for KI CLI."""
    parser = argparse.ArgumentParser(
        description="Generate FINN-compatible HWCustomOp from SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainsmith.tools.kernel_integrator matrix_mult.sv -o output/
  python -m brainsmith.tools.kernel_integrator conv2d.sv -o output/ --debug
  python -m brainsmith.tools.kernel_integrator thresholding.sv -o output/

Notes:
  - RTL file should contain @brainsmith BDIM and @brainsmith DATATYPE pragmas
  - Generated files will be organized in kernel-specific subdirectories
  - Use --debug for detailed generation information
        """
    )
    
    # Required arguments
    parser.add_argument('rtl_file', type=Path, help='SystemVerilog RTL file to process')
    parser.add_argument('-o', '--output', type=Path, required=False, 
                       help='Output directory for generated files (default: brainsmith/hw_kernels/<kernel_name>)')
    
    # Optional arguments
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging and detailed output')
    parser.add_argument('--no-strict', action='store_true',
                       help='Disable strict validation (allows parsing files that don\'t meet all requirements)')
    
    return parser


def main():
    """Main CLI entry point using KernelIntegrator."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not args.rtl_file.exists():
        print(f"❌ Error: RTL file not found: {args.rtl_file}")
        return 1
    
    try:
        # Step 1: Parse RTL first to get kernel name for default output directory
        parser_instance = RTLParser(strict=not args.no_strict)
        kernel_metadata = parser_instance.parse_file(str(args.rtl_file))
        
        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            # Default: brainsmith/hw_kernels/<kernel_name>
            # Find brainsmith root by looking for parent directories containing "brainsmith"
            current_path = args.rtl_file.resolve().parent
            brainsmith_root = None
            while current_path != current_path.parent:
                if current_path.name == "brainsmith" and (current_path / "kernels").exists():
                    brainsmith_root = current_path
                    break
                current_path = current_path.parent
            
            if brainsmith_root:
                output_dir = brainsmith_root / "kernels" / kernel_metadata.name
            else:
                # Fallback: create in current directory
                output_dir = Path.cwd() / "brainsmith" / "kernels" / kernel_metadata.name
        
        if args.debug:
            print("=== Kernel Integrator ===")
            print(f"RTL file: {args.rtl_file}")
            print(f"Output directory: {output_dir}")
            print()
        
        # Step 1 (cont): Report parsing results
        if args.debug:
            print("🔍 Step 1: Parsing RTL with parameter and BDIM validation...")
            if args.no_strict:
                print("   ⚠️  Running in non-strict mode (validation disabled)")
        
        if args.debug:
            print(f"   ✅ Parsed module: {kernel_metadata.name}")
            print(f"   ✅ Found {len(kernel_metadata.parameters)} parameters: {[p.name for p in kernel_metadata.parameters]}")
            print(f"   ✅ Found {len(kernel_metadata.interfaces)} interfaces: {[i.name for i in kernel_metadata.interfaces]}")
            print()
        
        # Step 2: Integrated generation and file writing
        if args.debug:
            print("🏭 Step 2: Generating templates and writing files...")
        
        # Use KernelIntegrator for modular generation
        integrator = KernelIntegrator(output_dir=output_dir)
        result = integrator.generate_and_write(kernel_metadata)
        
        if args.debug:
            print(f"   ✅ Generated {len(result.generated_files)} files:")
            for filename in result.generated_files.keys():
                print(f"      📄 {filename}")
            if result.files_written:
                print(f"   ✅ Written {len(result.files_written)} files to filesystem")
            print()
        
        # Step 3: Report success
        if result.is_success:
            print(f"✅ Successfully generated HWCustomOp for {kernel_metadata.name}")
            print(f"📁 Output directory: {result.output_directory}")
            print(f"⚡ Generated {len(result.generated_files)} files in {result.generation_time_ms:.1f}ms")
        else:
            print(f"❌ Generation failed for {kernel_metadata.name}")
            for error in result.errors:
                print(f"   Error: {error}")
            return 1
        
        if args.debug and result.is_success:
            print()
            print("Generated files:")
            for file_path in result.files_written:
                if file_path.exists() and not file_path.name.startswith('generation_'):
                    file_size = file_path.stat().st_size
                    print(f"   📄 {file_path.name} ({file_size:,} bytes)")
            
            print()
            print("Metadata files:")
            for file_path in result.metadata_files:
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"   📋 {file_path.name} ({file_size:,} bytes)")
        
        elif args.debug and not result.is_success:
            print()
            print("Errors encountered:")
            for error in result.errors:
                print(f"   ❌ {error}")
            if result.warnings:
                print()
                print("Warnings:")
                for warning in result.warnings:
                    print(f"   ⚠️ {warning}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.debug:
            import traceback
            import sys
            print()
            print("Debug traceback:")
            traceback.print_exc(file=sys.stdout)
        return 1


if __name__ == '__main__':
    sys.exit(main())