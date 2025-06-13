"""
Phase 3 CLI for Hardware Kernel Generator.

Unified CLI using the Phase 3 infrastructure with UnifiedGenerator and ResultHandler.
Single generation path: parse RTL → generate all templates → write files → done.
"""

import argparse
import sys
import time
from pathlib import Path

from .rtl_parser.parser import RTLParser
from .unified_generator import UnifiedGenerator
from .result_handler import ResultHandler, GenerationResult


def create_parser() -> argparse.ArgumentParser:
    """Create simplified argument parser for Phase 3 CLI."""
    parser = argparse.ArgumentParser(
        description="Generate FINN-compatible HWCustomOp from SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m brainsmith.tools.hw_kernel_gen matrix_mult.sv -o output/
  python -m brainsmith.tools.hw_kernel_gen conv2d.sv -o output/ --debug
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv -o output/ --template-version phase2

Notes:
  - RTL file should contain @brainsmith BDIM and @brainsmith DATATYPE pragmas
  - Generated files will be organized in kernel-specific subdirectories
  - Use --debug for detailed generation information
        """
    )
    
    # Required arguments
    parser.add_argument('rtl_file', type=Path, help='SystemVerilog RTL file to process')
    parser.add_argument('-o', '--output', type=Path, required=True, 
                       help='Output directory for generated files')
    
    # Optional arguments
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging and detailed output')
    parser.add_argument('--template-version', choices=['phase2'], default='phase2',
                       help='Template version to use (default: phase2)')
    
    return parser


def main():
    """Phase 3 main CLI entry point using UnifiedGenerator."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not args.rtl_file.exists():
        print(f"❌ Error: RTL file not found: {args.rtl_file}")
        return 1
    
    try:
        if args.debug:
            print("=== Phase 3 Hardware Kernel Generator ===")
            print(f"RTL file: {args.rtl_file}")
            print(f"Output directory: {args.output}")
            print(f"Template version: {args.template_version}")
            print()
        
        start_time = time.time()
        
        # Step 1: Parse RTL with Phase 1 validation
        if args.debug:
            print("🔍 Step 1: Parsing RTL with parameter and BDIM validation...")
        
        parser_instance = RTLParser()
        kernel_metadata = parser_instance.parse_file(str(args.rtl_file))
        
        if args.debug:
            print(f"   ✅ Parsed module: {kernel_metadata.name}")
            print(f"   ✅ Found {len(kernel_metadata.parameters)} parameters: {[p.name for p in kernel_metadata.parameters]}")
            print(f"   ✅ Found {len(kernel_metadata.interfaces)} interfaces: {[i.name for i in kernel_metadata.interfaces]}")
            print()
        
        # Step 2: Generate all templates with Phase 3 unified generator
        if args.debug:
            print("🏭 Step 2: Generating all templates with Phase 3 unified system...")
        
        generator = UnifiedGenerator()
        generated_files = generator.generate_all(kernel_metadata)
        
        if args.debug:
            print(f"   ✅ Generated {len(generated_files)} files:")
            for filename in generated_files.keys():
                print(f"      📄 {filename}")
            print()
        
        # Step 3: Write results with enhanced result handler
        if args.debug:
            print("💾 Step 3: Writing files and metadata...")
        
        generation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = GenerationResult(
            kernel_name=kernel_metadata.name,
            source_file=args.rtl_file,
            generated_files=generated_files,
            generation_time_ms=generation_time
        )
        
        handler = ResultHandler(args.output)
        kernel_dir = handler.write_result(result)
        
        # Step 4: Report success
        print(f"✅ Successfully generated HWCustomOp for {kernel_metadata.name}")
        print(f"📁 Output directory: {kernel_dir}")
        print(f"⚡ Generated {len(generated_files)} files in {generation_time:.1f}ms")
        
        if args.debug:
            print()
            print("Generated files:")
            for filename in generated_files.keys():
                file_path = kernel_dir / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"   📄 {filename} ({file_size:,} bytes)")
            
            print()
            print("Metadata files:")
            metadata_file = kernel_dir / "generation_metadata.json"
            summary_file = kernel_dir / "generation_summary.txt"
            if metadata_file.exists():
                print(f"   📋 generation_metadata.json ({metadata_file.stat().st_size:,} bytes)")
            if summary_file.exists():
                print(f"   📋 generation_summary.txt ({summary_file.stat().st_size:,} bytes)")
        
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