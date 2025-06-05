############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import os
import importlib.util
import ast
import argparse # Added for CLI
import sys # Added for CLI exit
from pathlib import Path
from typing import Optional, Dict, Any

# Assuming RTLParser and HWKernel data structure are in the rtl_parser sibling directory
# Adjust the import path based on your final project structure
# Ensure rtl_parser is correctly importable relative to this script's execution context
try:
    from .rtl_parser import RTLParser, HWKernel, ParserError
    from .generators.rtl_template_generator import generate_rtl_template
    # from .generators.hw_custom_op_generator import generate_hw_custom_op
    # from .generators.rtl_backend_generator import generate_rtl_backend
    # from .generators.doc_generator import generate_documentation
except ImportError:
    # Fallback for running script directly (adjust as needed)
    print("Warning: Running script directly, attempting relative imports from parent.")
    sys.path.append(str(Path(__file__).parent.parent)) # Add tools dir to path
    from hw_kernel_gen.rtl_parser import RTLParser, HWKernel, ParserError
    from hw_kernel_gen.generators.rtl_template_generator import generate_rtl_template
    # from hw_kernel_gen.generators.hw_custom_op_generator import generate_hw_custom_op
    # from hw_kernel_gen.generators.rtl_backend_generator import generate_rtl_backend
    # from hw_kernel_gen.generators.doc_generator import generate_documentation

# Import dataflow framework components for enhanced HKG
try:
    from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter, validate_conversion_result
    from brainsmith.dataflow.core.dataflow_model import DataflowModel
    from brainsmith.dataflow.core.validation import ValidationSeverity
    DATAFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dataflow framework not available: {e}")
    DATAFLOW_AVAILABLE = False


class HardwareKernelGeneratorError(Exception):
    """Custom exception for HKG errors."""
    pass

class HardwareKernelGenerator:
    """
    Orchestrates the generation of FINN integration files for a custom RTL HW Kernel.

    Takes an RTL source file and supplementary compiler data, parses them,
    and generates:
    1. A parameterizable RTL wrapper template.
    2. A HWCustomOp instance for FINN DSE.
    3. An RTLBackend instance for FINN RTL synthesis.
    4. Documentation for the kernel.
    """

    def __init__(
        self,
        rtl_file_path: str,
        compiler_data_path: str,
        output_dir: str,
        custom_doc_path: Optional[str] = None,
    ):
        """
        Initializes the HardwareKernelGenerator.

        Args:
            rtl_file_path: Path to the SystemVerilog RTL source file.
            compiler_data_path: Path to the Python file containing compiler data
                                (ONNX pattern, cost functions).
            output_dir: Directory where generated files will be saved.
            custom_doc_path: Optional path to a Markdown file with custom documentation.

        Raises:
            FileNotFoundError: If input files do not exist.
            HardwareKernelGeneratorError: For configuration errors.
        """
        self.rtl_file_path = Path(rtl_file_path)
        self.compiler_data_path = Path(compiler_data_path)
        self.output_dir = Path(output_dir)
        self.custom_doc_path = Path(custom_doc_path) if custom_doc_path else None

        # Validate inputs
        if not self.rtl_file_path.is_file():
            raise FileNotFoundError(f"RTL file not found: {self.rtl_file_path}")
        if not self.compiler_data_path.is_file():
            raise FileNotFoundError(f"Compiler data file not found: {self.compiler_data_path}")
        if self.custom_doc_path and not self.custom_doc_path.is_file():
            raise FileNotFoundError(f"Custom documentation file not found: {self.custom_doc_path}")
        if not self.output_dir.is_dir():
             # Attempt to create the output directory if it doesn't exist
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created output directory: {self.output_dir}")
            except OSError as e:
                raise HardwareKernelGeneratorError(f"Could not create output directory {self.output_dir}: {e}")


        self.hw_kernel_data: Optional[HWKernel] = None
        self.compiler_data_module: Optional[Any] = None
        self.compiler_data_ast: Optional[ast.Module] = None
        self.custom_doc_content: Optional[str] = None

        # Instantiate the parser with debug enabled
        self.rtl_parser = RTLParser(debug=True) # Pass debug=True

        # Dictionary to store paths of generated files
        self.generated_files: Dict[str, Path] = {}
        
        # Enhanced dataflow modeling support
        self.dataflow_enabled = DATAFLOW_AVAILABLE
        self.dataflow_interfaces: Optional[list] = None
        self.dataflow_model: Optional[DataflowModel] = None
        self.rtl_converter: Optional[RTLInterfaceConverter] = None
        
        if self.dataflow_enabled:
            print("Dataflow framework available - enhanced generation enabled")
        else:
            print("Dataflow framework not available - basic generation only")

    def _parse_rtl(self):
        """Parses the input RTL file using RTLParser."""
        print(f"--- Parsing RTL file: {self.rtl_file_path} ---")
        try:
            self.hw_kernel_data = self.rtl_parser.parse_file(str(self.rtl_file_path))
            print("RTL parsing successful.")
            # TODO: Add more detailed logging of extracted info (params, ports, interfaces)
        except ParserError as e:
            raise HardwareKernelGeneratorError(f"Failed to parse RTL: {e}")
        except Exception as e:
            raise HardwareKernelGeneratorError(f"An unexpected error occurred during RTL parsing: {e}")

    def _parse_compiler_data(self):
        """Imports and parses the compiler data Python file."""
        print(f"--- Parsing Compiler Data file: {self.compiler_data_path} ---")
        try:
            # 1. Import the module to access objects (ONNX model, functions)
            spec = importlib.util.spec_from_file_location("compiler_data", self.compiler_data_path)
            if spec is None or spec.loader is None:
                 raise HardwareKernelGeneratorError(f"Could not create module spec for {self.compiler_data_path}")
            self.compiler_data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.compiler_data_module)
            print("Compiler data module imported successfully.")
            # TODO: Add validation checks for required objects (ONNX pattern, cost functions)

            # 2. Parse the file content into an AST for potential regeneration/analysis
            with open(self.compiler_data_path, 'r') as f:
                source_code = f.read()
            self.compiler_data_ast = ast.parse(source_code)
            print("Compiler data AST parsed successfully.")

        except FileNotFoundError:
             raise HardwareKernelGeneratorError(f"Compiler data file not found at {self.compiler_data_path}")
        except SyntaxError as e:
            raise HardwareKernelGeneratorError(f"Syntax error in compiler data file {self.compiler_data_path}: {e}")
        except ImportError as e:
            raise HardwareKernelGeneratorError(f"Failed to import compiler data module from {self.compiler_data_path}: {e}")
        except Exception as e:
            raise HardwareKernelGeneratorError(f"An unexpected error occurred during compiler data parsing: {e}")

    def _load_custom_documentation(self):
        """Loads content from the optional custom documentation file."""
        if self.custom_doc_path:
            print(f"--- Loading Custom Documentation: {self.custom_doc_path} ---")
            try:
                with open(self.custom_doc_path, 'r') as f:
                    self.custom_doc_content = f.read()
                print("Custom documentation loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load custom documentation file: {e}")
                self.custom_doc_content = None # Ensure it's None if loading fails


    def _generate_rtl_template(self):
        """Generates the RTL wrapper template."""
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate RTL template: RTL data not parsed.")
        print("--- Generating RTL Template ---")
        # Placeholder: Call the actual generator function
        output_path = generate_rtl_template(self.hw_kernel_data, self.output_dir)
        self.generated_files["rtl_template"] = output_path
        print(f"RTL Template generation placeholder complete. Output: {output_path}")


    def _build_dataflow_model(self):
        """
        Build dataflow model from RTL interfaces if dataflow framework is available.
        
        This enhanced method converts RTL Parser interfaces to DataflowInterface objects
        and creates a unified computational model for performance analysis.
        """
        if not self.dataflow_enabled:
            print("Dataflow framework not available - skipping dataflow model generation")
            return
            
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot build dataflow model: RTL data not parsed.")
            
        print("--- Building Dataflow Model ---")
        
        try:
            # Extract ONNX metadata from compiler data if available
            onnx_metadata = {}
            if self.compiler_data_module and hasattr(self.compiler_data_module, 'onnx_metadata'):
                onnx_metadata = self.compiler_data_module.onnx_metadata
                
            # Initialize RTL converter with ONNX metadata
            self.rtl_converter = RTLInterfaceConverter(onnx_metadata)
            
            # Convert RTL interfaces to DataflowInterface objects
            self.dataflow_interfaces = self.rtl_converter.convert_interfaces(
                self.hw_kernel_data.interfaces,
                # Convert parameters to dict for TDIM pragma evaluation
                {param.name: param.default_value for param in self.hw_kernel_data.parameters if param.default_value}
            )
            
            # Validate conversion results
            conversion_errors = validate_conversion_result(self.dataflow_interfaces)
            error_count = len([e for e in conversion_errors if e.severity == ValidationSeverity.ERROR])
            warning_count = len([e for e in conversion_errors if e.severity == ValidationSeverity.WARNING])
            
            if error_count > 0:
                print(f"Dataflow conversion completed with {error_count} errors and {warning_count} warnings")
                for error in conversion_errors:
                    if error.severity == ValidationSeverity.ERROR:
                        print(f"  ERROR: {error.message}")
            else:
                print(f"Dataflow conversion successful: {len(self.dataflow_interfaces)} interfaces converted")
                if warning_count > 0:
                    print(f"  {warning_count} warnings generated")
            
            # Build unified computational model
            if self.dataflow_interfaces:
                self.dataflow_model = DataflowModel(self.dataflow_interfaces, {})
                print(f"Dataflow model created with {len(self.dataflow_interfaces)} interfaces")
            else:
                print("No dataflow interfaces available for model creation")
                
        except Exception as e:
            print(f"Warning: Failed to build dataflow model: {e}")
            # Don't raise error - continue with basic generation
            self.dataflow_interfaces = None
            self.dataflow_model = None

    def _generate_hw_custom_op(self):
        """
        Enhanced HWCustomOp generation with dataflow modeling support.
        
        Generates AutoHWCustomOp classes with unified computational model integration.
        Requires dataflow framework to be available and successfully initialized.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: RTL data not parsed.")
            
        if not self.dataflow_enabled:
            raise HardwareKernelGeneratorError(
                "HWCustomOp generation requires dataflow framework. "
                "Please ensure brainsmith.dataflow is available."
            )
            
        if not self.dataflow_interfaces or not self.dataflow_model:
            raise HardwareKernelGeneratorError(
                "HWCustomOp generation requires successful dataflow model initialization. "
                "Please check RTL pragmas and interface definitions."
            )
            
        print("--- Generating HWCustomOp Instance ---")
        
        # Generate AutoHWCustomOp with dataflow modeling
        output_path = self._generate_auto_hwcustomop_with_dataflow()
            
        self.generated_files["hw_custom_op"] = output_path
        print(f"HWCustomOp generation complete. Output: {output_path}")
        
    def _generate_auto_hwcustomop_with_dataflow(self) -> Path:
        """
        Generate AutoHWCustomOp with full dataflow modeling support.
        
        Returns:
            Path to generated AutoHWCustomOp file
        """
        from jinja2 import Environment, FileSystemLoader
        
        print("Generating enhanced AutoHWCustomOp with dataflow modeling")
        
        # Build template context with dataflow information
        template_context = self._build_enhanced_template_context()
        
        # Generate class using template (placeholder for now)
        class_name = f"Auto{self.hw_kernel_data.name.title()}"
        output_file = self.output_dir / f"{class_name.lower()}.py"
        
        # For now, write a placeholder with dataflow metadata
        with open(output_file, 'w') as f:
            f.write(f'"""\nAuto-generated HWCustomOp for {self.hw_kernel_data.name}\n')
            f.write(f'Generated with dataflow modeling support\n')
            f.write(f'Interfaces: {len(self.dataflow_interfaces)}\n')
            f.write(f'Dataflow model: {self.dataflow_model is not None}\n')
            f.write('"""\n\n')
            f.write(f'# TODO: Implement {class_name} using template system\n')
            f.write(f'# Template context available: {list(template_context.keys())}\n')
        
        return output_file
        
    def _build_enhanced_template_context(self) -> Dict[str, Any]:
        """
        Build comprehensive template context with dataflow modeling information.
        
        Returns:
            Dictionary containing all template variables for enhanced code generation
        """
        from datetime import datetime
        
        context = {
            # Kernel metadata
            "kernel_name": self.hw_kernel_data.name,
            "class_name": f"Auto{self.hw_kernel_data.name.replace('_', '').title()}",
            "source_file": str(self.rtl_file_path),
            "generation_timestamp": datetime.now().isoformat(),
            
            # RTL Parser data
            "rtl_parameters": self.hw_kernel_data.parameters,
            "rtl_interfaces": self.hw_kernel_data.interfaces,
            "rtl_pragmas": self.hw_kernel_data.pragmas,
            
            # Dataflow framework data
            "dataflow_interfaces": self.dataflow_interfaces or [],
            "dataflow_model": self.dataflow_model,
            
            # Interface organization for templates
            "input_interfaces": [iface for iface in (self.dataflow_interfaces or [])
                               if hasattr(iface, 'interface_type') and
                               str(iface.interface_type).endswith('INPUT')],
            "output_interfaces": [iface for iface in (self.dataflow_interfaces or [])
                                if hasattr(iface, 'interface_type') and
                                str(iface.interface_type).endswith('OUTPUT')],
            "weight_interfaces": [iface for iface in (self.dataflow_interfaces or [])
                                if hasattr(iface, 'interface_type') and
                                str(iface.interface_type).endswith('WEIGHT')],
            "config_interfaces": [iface for iface in (self.dataflow_interfaces or [])
                                if hasattr(iface, 'interface_type') and
                                str(iface.interface_type).endswith('CONFIG')],
            
            # Computational model data
            "has_unified_model": self.dataflow_model is not None,
            "parallelism_bounds": self.dataflow_model.get_parallelism_bounds() if self.dataflow_model else {},
            
            # Compiler data
            "compiler_data_available": self.compiler_data_module is not None,
        }
        
        return context
    
    def generate_auto_hwcustomop(self, template_path: str, output_path: str) -> str:
        """
        Public method for generating AutoHWCustomOp with dataflow modeling.
        
        Args:
            template_path: Path to Jinja2 template file
            output_path: Output file path for generated class
            
        Returns:
            Path to generated file
            
        Raises:
            HardwareKernelGeneratorError: If generation fails
        """
        if not self.dataflow_enabled:
            raise HardwareKernelGeneratorError("AutoHWCustomOp generation requires dataflow framework")
            
        if not self.hw_kernel_data:
            self._parse_rtl()
            
        if not self.dataflow_model:
            self._build_dataflow_model()
            
        if not self.dataflow_model:
            raise HardwareKernelGeneratorError("Failed to build dataflow model for AutoHWCustomOp generation")
            
        try:
            from jinja2 import Environment, FileSystemLoader
            import os
            
            # Load template
            template_dir = os.path.dirname(template_path)
            template_name = os.path.basename(template_path)
            
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template(template_name)
            
            # Build context
            context = self._build_enhanced_template_context()
            
            # Render template
            generated_code = template.render(**context)
            
            # Write output
            with open(output_path, 'w') as f:
                f.write(generated_code)
                
            print(f"AutoHWCustomOp generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            raise HardwareKernelGeneratorError(f"AutoHWCustomOp generation failed: {e}")


    def _generate_rtl_backend(self):
        pass # Commented out until implemented
        # """Generates the RTLBackend instance file."""
        # if not self.hw_kernel_data or not self.compiler_data_module:
        #      raise HardwareKernelGeneratorError("Cannot generate RTLBackend: Required data not parsed.")
        # print("--- Generating RTLBackend Instance ---")
        # # Placeholder: Call the actual generator function
        # output_path = generate_rtl_backend(self.hw_kernel_data, self.compiler_data_module, self.output_dir)
        # self.generated_files["rtl_backend"] = output_path
        # print(f"RTLBackend generation placeholder complete. Output: {output_path}")


    def _generate_documentation(self):
        pass # Commented out until implemented
        # """Generates the documentation file."""
        # if not self.hw_kernel_data:
        #     raise HardwareKernelGeneratorError("Cannot generate documentation: RTL data not parsed.")
        # print("--- Generating Documentation ---")
        # # Placeholder: Call the actual generator function
        # output_path = generate_documentation(self.hw_kernel_data, self.custom_doc_content, self.output_dir)
        # self.generated_files["documentation"] = output_path
        # print(f"Documentation generation placeholder complete. Output: {output_path}")


    def get_parsed_rtl_data(self):
        """
        Returns the parsed RTL data for testing purposes.
        This is useful for testing components in isolation without running the full pipeline.
        
        Returns:
            The parsed HWKernel data, or None if it hasn't been parsed yet.
        """
        if not self.hw_kernel_data:
            self._parse_rtl()
        return self.hw_kernel_data


    def run(self, stop_after: Optional[str] = None):
        """
        Executes the HKG pipeline phases.

        Args:
            stop_after: Optional phase name ('parse_rtl', 'parse_compiler_data',
                        'generate_rtl_template', etc.) to stop execution after.
                        If None, runs all phases.

        Returns:
            A dictionary containing the paths to the generated files.

        Raises:
            HardwareKernelGeneratorError: If any phase encounters an error.
        """
        phases = [
            ("parse_rtl", self._parse_rtl),
            ("parse_compiler_data", self._parse_compiler_data),
            ("load_custom_documentation", self._load_custom_documentation),
            ("build_dataflow_model", self._build_dataflow_model),
            ("generate_rtl_template", self._generate_rtl_template),
            ("generate_hw_custom_op", self._generate_hw_custom_op),
            ("generate_rtl_backend", self._generate_rtl_backend),
            ("generate_documentation", self._generate_documentation),
        ]

        try:
            for name, phase_func in phases:
                phase_func()
                if stop_after and name == stop_after:
                    print(f"--- Stopping execution after phase: {name} ---")
                    break
        except HardwareKernelGeneratorError as e:
            print(f"Error during phase '{name}': {e}")
            # Potentially re-raise or handle differently
            raise # Re-raise the specific HKG error
        except Exception as e:
            print(f"An unexpected error occurred during phase '{name}': {e}")
            # Wrap unexpected errors
            raise HardwareKernelGeneratorError(f"Unexpected error in phase '{name}': {e}")


        print("--- Hardware Kernel Generation Complete ---")
        print("Generated files:")
        for key, path in self.generated_files.items():
            print(f"  {key}: {path}")

        return self.generated_files

# --- Command Line Interface ---
def main():
    parser = argparse.ArgumentParser(
        description="Hardware Kernel Generator (HKG) for Brainsmith/FINN."
    )
    parser.add_argument(
        "rtl_file",
        type=str,
        help="Path to the SystemVerilog RTL source file (.sv)."
    )
    parser.add_argument(
        "compiler_data",
        type=str,
        help="Path to the Python file containing compiler data (ONNX pattern, cost functions)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory where generated files will be saved."
    )
    parser.add_argument(
        "-d", "--custom-doc",
        type=str,
        default=None,
        help="Optional path to a Markdown file with custom documentation sections."
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=[
            "parse_rtl",
            "parse_compiler_data",
            "load_custom_documentation",
            "build_dataflow_model",
            "generate_rtl_template",
            "generate_hw_custom_op",
            "generate_rtl_backend",
            "generate_documentation"
        ],
        help="Stop execution after completing the specified phase (for debugging)."
    )

    args = parser.parse_args()

    try:
        print("--- Initializing Hardware Kernel Generator ---")
        hkg = HardwareKernelGenerator(
            rtl_file_path=args.rtl_file,
            compiler_data_path=args.compiler_data,
            output_dir=args.output_dir,
            custom_doc_path=args.custom_doc
        )
        generated_files = hkg.run(stop_after=args.stop_after)
        print("--- HKG Execution Successful ---")
        print("Generated files:")
        for name, path in generated_files.items():
            print(f"- {name}: {path}")
        sys.exit(0) # Success

    except (HardwareKernelGeneratorError, FileNotFoundError, ParserError) as e:
        print(f"\n--- HKG Error ---")
        print(f"Error: {e}")
        sys.exit(1) # Failure
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        sys.exit(2) # Unexpected failure


if __name__ == "__main__":
    main()
