"""Command-line interface for kernel integrator."""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .rtl_parser.parser import RTLParser
from .generator import KernelGenerator
from .metadata import KernelMetadata


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='kernel_integrator',
        description='Generate FINN-compatible HWCustomOp from SystemVerilog RTL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s design.sv                           # Generate all files in ./output/<kernel_name>/
  %(prog)s design.sv -o /path/output           # Generate files in specified directory
  %(prog)s design.sv --validate                # Validate RTL only (no file generation)
  %(prog)s design.sv --info                    # Display parsed kernel metadata
  %(prog)s design.sv --artifacts wrapper,autohwcustomop  # Generate specific files only
  %(prog)s design.sv --no-strict --verbose     # Disable strict validation with verbose output

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
    
    # Operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--validate',
        action='store_true',
        help='validate RTL only without generating files'
    )
    mode_group.add_argument(
        '--info',
        action='store_true',
        help='display parsed kernel metadata without generating files'
    )
    
    parser.add_argument(
        '--artifacts',
        type=str,
        metavar='LIST',
        help='comma-separated list of artifacts to generate (autohwcustomop,rtlbackend,wrapper)'
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


def parse_rtl_file(rtl_file: Path) -> KernelMetadata:
    """
    Parse RTL file and return kernel metadata.
    
    This function provides a clean interface for RTL parsing that returns
    KernelMetadata for direct template generation and dataflow integration.
    
    Args:
        rtl_file: Path to SystemVerilog RTL file or Path object
        
    Returns:
        KernelMetadata: Parsed kernel metadata with InterfaceMetadata objects
        
    Raises:
        RuntimeError: If RTL parsing fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure rtl_file is a Path object
        if isinstance(rtl_file, str):
            rtl_file = Path(rtl_file)
        
        # Create RTL parser instance
        parser = RTLParser()
        
        # Parse the RTL file and return KernelMetadata directly
        parsed_data = parser.parse_file(str(rtl_file))
        
        logger.info(f"Successfully parsed RTL file {rtl_file} → KernelMetadata '{parsed_data.name}'")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to parse RTL file {rtl_file}: {e}")
        # Re-raise for consistent error handling
        raise RuntimeError(f"RTL parsing failed for {rtl_file}: {e}") from e


def generate_kernel_files(
    rtl_file: Path,
    output_dir: Path,
    kernel_metadata: Optional['KernelMetadata'] = None,
    artifacts: Optional[List[str]] = None,
    strict: bool = True
) -> Tuple[List[Path], float]:
    """
    Generate kernel files from RTL.
    
    Args:
        rtl_file: Path to SystemVerilog file
        output_dir: Directory for generated files
        kernel_metadata: Pre-parsed metadata (to avoid double parsing)
        artifacts: List of specific artifacts to generate (None = all)
        strict: Enable strict validation
        
    Returns:
        Tuple of (list of generated file paths, generation time in ms)
        
    Raises:
        FileNotFoundError: If RTL file doesn't exist
        RuntimeError: If generation fails
    """
    start_time = time.time()
    
    # Parse RTL if metadata not provided
    if kernel_metadata is None:
        parser = RTLParser(strict=strict)
        kernel_metadata = parser.parse_file(str(rtl_file))
    
    # Generate files
    generator = KernelGenerator()
    
    if artifacts:
        # Generate only specified artifacts
        outputs = {}
        for artifact in artifacts:
            outputs[artifact] = generator.generate(artifact, kernel_metadata, output_dir)
    else:
        # Generate all files
        outputs = generator.generate_all(kernel_metadata, output_dir)
    
    # Convert dict values to list for return
    generated_files = list(outputs.values())
    
    elapsed_ms = (time.time() - start_time) * 1000
    return generated_files, elapsed_ms


def format_file_info(files: List[Path]) -> str:
    """Format file information for display."""
    lines = []
    for path in files:
        size = path.stat().st_size
        lines.append(f"  📄 {path.name} ({size:,} bytes)")
    return '\n'.join(lines)


def display_kernel_info(metadata: 'KernelMetadata') -> None:
    """Display parsed kernel metadata in a readable format."""
    print(f"\n🔍 Kernel Metadata for '{metadata.name}'")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\n📦 Module: {metadata.name}")
    if metadata.description:
        print(f"📝 Description: {metadata.description}")
    
    # Parameters
    if metadata.parameters:
        print(f"\n⚙️  Parameters ({len(metadata.parameters)}):")
        for param in metadata.parameters.values():
            default = f" = {param.default_value}" if param.default_value else ""
            print(f"  - {param.name}: {param.rtl_type or 'unknown'}{default}")
    
    # Interfaces
    if metadata.interfaces:
        print(f"\n🔌 Interfaces ({len(metadata.interfaces)}):")
        for iface in metadata.interfaces.values():
            protocol = iface.protocol_type.value if hasattr(iface, 'protocol_type') else 'unknown'
            direction = iface.direction.value if hasattr(iface, 'direction') else 'unknown'
            print(f"  - {iface.name}: {protocol} ({direction})")
            if hasattr(iface, 'ports') and iface.ports:
                for port in iface.ports.values():
                    print(f"    • {port.name}[{port.width}]")
    
    # Pragmas summary
    pragma_count = len(getattr(metadata, '_applied_pragmas', []))
    if pragma_count > 0:
        print(f"\n🏷️  Pragmas applied: {pragma_count}")
    
    print()


def validate_only(rtl_file: Path, strict: bool = True) -> int:
    """Validate RTL file without generating output.
    
    Returns:
        0 if valid, 1 if invalid
    """
    try:
        parser = RTLParser(strict=strict)
        parser.parse_file(str(rtl_file))
        print(f"✅ RTL file '{rtl_file}' is valid")
        return 0
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def parse_artifacts_list(artifacts_str: str) -> List[str]:
    """Parse comma-separated artifacts list and validate."""
    if not artifacts_str:
        return []
    
    artifacts = [a.strip() for a in artifacts_str.split(',')]
    valid_artifacts = {'autohwcustomop', 'rtlbackend', 'wrapper'}
    
    invalid = [a for a in artifacts if a not in valid_artifacts]
    if invalid:
        raise ValueError(f"Invalid artifacts: {', '.join(invalid)}. Valid options: {', '.join(valid_artifacts)}")
    
    return artifacts


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
        # Handle validate-only mode
        if args.validate:
            return validate_only(args.rtl_file, strict=not args.no_strict)
        
        # Parse RTL once (used by all modes)
        parser_inst = RTLParser(strict=not args.no_strict)
        metadata = parser_inst.parse_file(str(args.rtl_file))
        
        # Handle info mode
        if args.info:
            display_kernel_info(metadata)
            return 0
        
        # Parse artifacts list if provided
        artifacts = None
        if args.artifacts:
            artifacts = parse_artifacts_list(args.artifacts)
        
        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = Path('output') / metadata.name
        
        # Generate files (passing metadata to avoid re-parsing)
        files, elapsed_ms = generate_kernel_files(
            rtl_file=args.rtl_file,
            output_dir=output_dir,
            kernel_metadata=metadata,
            artifacts=artifacts,
            strict=not args.no_strict
        )
        
        # Report success
        print(f"✅ Successfully generated HWCustomOp for {metadata.name}")
        print(f"📁 Output directory: {output_dir}")
        if artifacts:
            print(f"⚡ Generated {len(files)} selected files in {elapsed_ms:.1f}ms")
        else:
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