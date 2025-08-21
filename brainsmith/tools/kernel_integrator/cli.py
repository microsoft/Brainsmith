"""Command-line interface for kernel integrator."""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .rtl_parser.parser import RTLParser
from .generator import KernelGenerator


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='kernel_integrator',
        description='Generate FINN-compatible HWCustomOp from SystemVerilog RTL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s design.sv                    # Generate files in ./output/<kernel_name>/
  %(prog)s design.sv -o /path/output    # Generate files in specified directory
  %(prog)s design.sv --no-strict        # Disable strict validation
  %(prog)s design.sv --verbose          # Enable verbose output

Notes:
  RTL files should contain @brainsmith pragmas to define interfaces and parameters.
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'rtl_file',
        type=Path,
        help='SystemVerilog RTL file to process'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=Path,
        metavar='DIR',
        help='output directory (default: ./output/<kernel_name>/)'
    )
    
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='disable strict validation'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 4.0.0'
    )
    
    return parser


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_kernel_files(
    rtl_file: Path,
    output_dir: Path,
    strict: bool = True
) -> Tuple[List[Path], float]:
    """
    Generate kernel files from RTL.
    
    Args:
        rtl_file: Path to SystemVerilog file
        output_dir: Directory for generated files
        strict: Enable strict validation
        
    Returns:
        Tuple of (list of generated file paths, generation time in ms)
        
    Raises:
        FileNotFoundError: If RTL file doesn't exist
        RuntimeError: If generation fails
    """
    start_time = time.time()
    
    # Parse RTL
    parser = RTLParser(strict=strict)
    kernel_metadata = parser.parse_file(str(rtl_file))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate files
    generator = KernelGenerator()
    kernel_name = kernel_metadata.name
    
    generators = {
        f"{kernel_name}_hw_custom_op.py": generator.generate_autohwcustomop,
        f"{kernel_name}_rtl.py": generator.generate_rtl_backend,
        f"{kernel_name}_wrapper.v": generator.generate_rtl_wrapper
    }
    
    generated_files = []
    for filename, generate_func in generators.items():
        content = generate_func(kernel_metadata)
        file_path = output_dir / filename
        file_path.write_text(content, encoding='utf-8')
        generated_files.append(file_path)
    
    elapsed_ms = (time.time() - start_time) * 1000
    return generated_files, elapsed_ms


def format_file_info(files: List[Path]) -> str:
    """Format file information for display."""
    lines = []
    for path in files:
        size = path.stat().st_size
        lines.append(f"  📄 {path.name} ({size:,} bytes)")
    return '\n'.join(lines)


def main(argv=None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input file
    if not args.rtl_file.exists():
        print(f"❌ Error: RTL file not found: {args.rtl_file}", file=sys.stderr)
        return 1
    
    try:
        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            # Need to parse to get kernel name for default output
            parser_inst = RTLParser(strict=not args.no_strict)
            metadata = parser_inst.parse_file(str(args.rtl_file))
            output_dir = Path('output') / metadata.name
        
        # Generate files
        files, elapsed_ms = generate_kernel_files(
            rtl_file=args.rtl_file,
            output_dir=output_dir,
            strict=not args.no_strict
        )
        
        # Report success
        kernel_name = files[0].name.split('_hw_custom_op.py')[0]
        print(f"✅ Successfully generated HWCustomOp for {kernel_name}")
        print(f"📁 Output directory: {output_dir}")
        print(f"⚡ Generated {len(files)} files in {elapsed_ms:.1f}ms")
        
        if args.verbose:
            print("\nGenerated files:")
            print(format_file_info(files))
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            print("\nTraceback:", file=sys.stderr)
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())