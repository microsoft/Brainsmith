"""
Command-line interface for HWKG.

Simple by default, powerful when needed. Based on hw_kernel_gen_simple CLI
with enhancements for complexity levels and feature flags.
Follows HWKG Axiom 10: Architecture Principle.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .config import Config
from .data import GenerationResult
from .rtl_parser import parse_rtl_file, ParsedKernelData
from .generators import HWCustomOpGenerator, RTLBackendGenerator, TestSuiteGenerator
from .errors import HWKGError, CompilerDataError


def create_parsed_kernel_data(config: Config) -> ParsedKernelData:
    """
    Create ParsedKernelData from RTL file and compiler data.
    
    Uses RTL parser with direct ParsedKernelData generation,
    maintaining error resilience and simple-by-default philosophy.
    """
    # Parse RTL file with appropriate sophistication level
    if config.debug:
        print(f"Parsing RTL file: {config.rtl_file}")
        if config.advanced_pragmas:
            print("Advanced BDIM pragma processing enabled")
    
    parsed_data = parse_rtl_file(config.rtl_file, advanced_pragmas=config.advanced_pragmas)
    
    if config.debug:
        print(f"Found module: {parsed_data.name}")
        print(f"Interfaces: {len(parsed_data.interfaces)}")
        print(f"Parameters: {len(parsed_data.parameters)}")
        print(f"Complexity level: {config.complexity_level}")
    
    # Load compiler data (same robust logic as simple system)
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
        
        # Store compiler data for generator use (ParsedKernelData doesn't need this attached)
        if config.debug:
            print(f"Loaded compiler data with {len(compiler_data)} entries")
        
    except Exception as e:
        raise CompilerDataError(f"Failed to load compiler data: {e}") from e
    
    return parsed_data


def _load_compiler_data(config: Config) -> dict:
    """Load compiler data from file."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("compiler_data", config.compiler_data_file)
        if spec is None or spec.loader is None:
            raise CompilerDataError(f"Could not load compiler data from {config.compiler_data_file}")
        
        compiler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiler_module)
        
        if not hasattr(compiler_module, 'compiler_data'):
            raise CompilerDataError(f"No 'compiler_data' variable found in {config.compiler_data_file}")
        
        return compiler_module.compiler_data
        
    except Exception as e:
        raise CompilerDataError(f"Failed to load compiler data: {e}") from e


def generate_all_enhanced(config: Config) -> GenerationResult:
    """
    Enhanced generation using direct RTL-to-templates pipeline.
    
    This bypasses HWKernel and DataflowModel conversion for improved performance.
    """
    try:
        # Import here to avoid circular import
        from ..unified_hwkg.generator import create_enhanced_generator
        
        # Create enhanced generator
        generator = create_enhanced_generator()
        
        # Load compiler data
        compiler_data = _load_compiler_data(config)
        
        # Use enhanced generation pipeline
        unified_result = generator.generate_from_rtl(
            rtl_file=config.rtl_file,
            compiler_data=compiler_data,
            output_dir=config.output_dir,
            generate_wrapper=True,
            generate_docs=config.debug  # Generate docs in debug mode
        )
        
        # Convert to legacy GenerationResult format for CLI compatibility
        result = GenerationResult(
            generated_files=unified_result.generated_files,
            success=unified_result.success,
            complexity_level=config.complexity_level
        )
        
        if unified_result.errors:
            for error in unified_result.errors:
                result.add_error(error)
        
        if unified_result.warnings:
            for warning in unified_result.warnings:
                result.add_warning(warning)
        
        if config.debug:
            print(f"Enhanced generation completed: {len(unified_result.generated_files)} files")
            for file_path in unified_result.generated_files:
                print(f"Generated: {file_path}")
        
        return result
        
    except Exception as e:
        result = GenerationResult(
            generated_files=[], 
            success=False,
            complexity_level=config.complexity_level
        )
        result.add_error(f"Enhanced generation failed: {e}")
        
        if config.debug:
            import traceback
            traceback.print_exc()
        
        return result


def generate_all(parsed_data: ParsedKernelData, config: Config) -> GenerationResult:
    """
    Generation using ParsedKernelData.
    
    Simple mode: Direct generation like hw_kernel_gen_simple
    Advanced mode: Optional multi-phase execution with stop points
    """
    result = GenerationResult(
        generated_files=[], 
        success=True,
        complexity_level=config.complexity_level
    )
    # Check for enhanced BDIM processing based on pragma sophistication
    result.set_bdim_processing(len(parsed_data.pragmas) > 0)
    
    # Multi-phase execution for advanced users
    if config.multi_phase_execution:
        return _generate_multi_phase(parsed_data, config, result)
    else:
        return _generate_simple_mode(parsed_data, config, result)


def _generate_simple_mode(parsed_data: ParsedKernelData, config: Config, result: GenerationResult) -> GenerationResult:
    """Simple mode generation using ParsedKernelData."""
    generators = [
        HWCustomOpGenerator(config.template_dir),
        RTLBackendGenerator(config.template_dir),
        TestSuiteGenerator(config.template_dir)
    ]
    
    for generator in generators:
        try:
            output_file = generator.generate(parsed_data, config.output_dir)
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


def _generate_multi_phase(parsed_data: ParsedKernelData, config: Config, result: GenerationResult) -> GenerationResult:
    """
    Multi-phase generation with debugging stops.
    
    Following HWKG Axiom 2: Multi-Phase Pipeline for expert users.
    """
    phases = [
        ('parse_rtl', lambda: print("✅ RTL parsing completed")),
        ('parse_compiler_data', lambda: print("✅ Compiler data loaded")),
        ('build_dataflow_model', lambda: _build_dataflow_model(parsed_data, config)),
        ('generate_hw_custom_op', lambda: _generate_hw_custom_op(parsed_data, config, result)),
        ('generate_rtl_backend', lambda: _generate_rtl_backend(parsed_data, config, result)),
        ('generate_test_suite', lambda: _generate_test_suite(parsed_data, config, result))
    ]
    
    for phase_name, phase_func in phases:
        if config.debug:
            print(f"🔄 Executing phase: {phase_name}")
        
        try:
            phase_func()
            
            if config.debug:
                print(f"✅ Phase {phase_name} completed")
            
            # Stop if requested
            if config.stop_after == phase_name:
                if config.debug:
                    print(f"🛑 Stopping after phase: {phase_name}")
                break
                
        except Exception as e:
            error_msg = f"Phase {phase_name} failed: {e}"
            result.add_error(error_msg)
            if config.debug:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            break
    
    return result


def _build_dataflow_model(parsed_data: ParsedKernelData, config: Config):
    """Build dataflow model from interfaces."""
    if config.debug:
        dataflow_interfaces = parsed_data.get_dataflow_interfaces()
        print(f"Building dataflow model with {len(dataflow_interfaces)} interfaces")
        if len(parsed_data.pragmas) > 0:
            print("Using pragma-based chunking strategies")


def _generate_hw_custom_op(parsed_data: ParsedKernelData, config: Config, result: GenerationResult):
    """Generate HWCustomOp in multi-phase mode."""
    generator = HWCustomOpGenerator(config.template_dir)
    output_file = generator.generate(parsed_data, config.output_dir)
    result.add_generated_file(output_file)


def _generate_rtl_backend(parsed_data: ParsedKernelData, config: Config, result: GenerationResult):
    """Generate RTLBackend in multi-phase mode."""
    generator = RTLBackendGenerator(config.template_dir)
    output_file = generator.generate(parsed_data, config.output_dir)
    result.add_generated_file(output_file)


def _generate_test_suite(parsed_data: ParsedKernelData, config: Config, result: GenerationResult):
    """Generate test suite in multi-phase mode."""
    generator = TestSuiteGenerator(config.template_dir)
    output_file = generator.generate(parsed_data, config.output_dir)
    result.add_generated_file(output_file)


def _generate_class_name(module_name: str) -> str:
    """Generate Python class name from module name."""
    # Convert snake_case or kebab-case to PascalCase
    parts = module_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts)


def main():
    """
    Main CLI entry point for HW Kernel Generator.
    
    Simple by default, powerful when needed.
    Maintains hw_kernel_gen_simple UX while enabling advanced features.
    """
    parser = argparse.ArgumentParser(
        description="Hardware Kernel Generator - Simple by default, powerful when needed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode (identical to hw_kernel_gen_simple)
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/
  
  # Advanced mode (enhanced BDIM pragma processing)
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/ --advanced-pragmas
  
  # Expert mode (multi-phase execution with debugging)
  python -m brainsmith.tools.hw_kernel_gen thresholding.sv compiler_data.py -o output/ --advanced-pragmas --multi-phase --debug

This HWKG eliminates dual-architecture complexity while preserving all functionality.
Based on hw_kernel_gen_simple foundation with optional BDIM sophistication.
Follows Interface-Wise Dataflow Modeling axioms for consistent terminology.
        """
    )
    
    # Core arguments (identical to hw_kernel_gen_simple for backward compatibility)
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
    
    # Feature flags for complexity levels
    parser.add_argument(
        '--advanced-pragmas', 
        action='store_true',
        help='Enable enhanced BDIM pragma processing with Interface-Wise Dataflow integration'
    )
    parser.add_argument(
        '--multi-phase', 
        action='store_true',
        help='Enable multi-phase execution with debugging stops (requires expert knowledge)'
    )
    parser.add_argument(
        '--stop-after', 
        type=str,
        choices=['parse_rtl', 'parse_compiler_data', 'build_dataflow_model', 
                'generate_hw_custom_op', 'generate_rtl_backend', 'generate_test_suite'],
        help='Stop execution after specified phase (requires --multi-phase)'
    )
    
    # Optional configuration
    parser.add_argument(
        '--template-dir', 
        type=str, 
        help='Custom template directory (optional, uses existing templates by default)'
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
            print("=== Hardware Kernel Generator ===")
            print(f"RTL file: {config.rtl_file}")
            print(f"Compiler data: {config.compiler_data_file}")
            print(f"Output directory: {config.output_dir}")
            print(f"Complexity level: {config.complexity_level}")
            if config.template_dir:
                print(f"Template directory: {config.template_dir}")
            if config.advanced_pragmas:
                print("🔬 Advanced BDIM pragma processing enabled")
            if config.multi_phase_execution:
                print("🔧 Multi-phase execution enabled")
            print()
        
        # Generate all outputs using enhanced approach by default
        if not config.multi_phase_execution:
            print("Generating files...")
        
        # Try enhanced generation first (faster, no DataflowModel overhead)
        try:
            result = generate_all_enhanced(config)
            if config.debug:
                print("✨ Used enhanced generation pipeline (no DataflowModel conversion)")
        except Exception as e:
            if config.debug:
                print(f"⚠️  Enhanced generation failed, falling back to legacy: {e}")
            
            # Fallback to ParsedKernelData approach
            parsed_data = create_parsed_kernel_data(config)
            
            if config.debug:
                print(f"Created kernel: {parsed_data.name} → {parsed_data.get_class_name()}")
                print(f"Interfaces: {len(parsed_data.interfaces)}")
                print(f"Parameters: {len(parsed_data.parameters)}")
                print(f"Kernel type: {parsed_data._infer_kernel_type()}")
                print(f"Complexity: {parsed_data._estimate_complexity()}")
                if parsed_data.parsing_warnings:
                    print(f"⚠️  Parsing warnings: {len(parsed_data.parsing_warnings)}")
                print("🔄 Using ParsedKernelData generation pipeline")
                print()
            
            result = generate_all(parsed_data, config)
        
        # Report results with enhanced information
        if result.success:
            print(f"✅ Successfully generated {len(result.generated_files)} files:")
            for file_path in result.generated_files:
                print(f"   📄 {file_path.relative_to(config.output_dir)}")
            
            # Additional information for advanced modes
            if result.bdim_processing_used:
                print(f"🔬 Enhanced BDIM pragma processing was used")
            if config.complexity_level != "simple":
                print(f"🎯 Complexity level: {config.complexity_level}")
            
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