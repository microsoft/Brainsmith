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

# Import new error handling framework
from .errors import BrainsmithError, RTLParsingError, CodeGenerationError, handle_error_with_recovery

# Assuming RTLParser and HWKernel data structure are in the rtl_parser sibling directory
# Adjust the import path based on your final project structure
# Ensure rtl_parser is correctly importable relative to this script's execution context
try:
    from .rtl_parser import RTLParser, HWKernel, ParserError
    from .generators.rtl_template_generator import generate_rtl_template
    # NOTE: HWCustomOp generation now uses HWCustomOpGenerator class
    # from .generators.rtl_backend_generator import RTLBackendGenerator  # Future
    # from .generators.doc_generator import DocumentationGenerator  # Future
except ImportError:
    # Fallback for running script directly (adjust as needed)
    print("Warning: Running script directly, attempting relative imports from parent.")
    sys.path.append(str(Path(__file__).parent.parent)) # Add tools dir to path
    from hw_kernel_gen.rtl_parser import RTLParser, HWKernel, ParserError
    from hw_kernel_gen.generators.rtl_template_generator import generate_rtl_template
    # NOTE: HWCustomOp generation now uses HWCustomOpGenerator class
    # from hw_kernel_gen.generators.rtl_backend_generator import RTLBackendGenerator  # Future
    # from hw_kernel_gen.generators.doc_generator import DocumentationGenerator  # Future

# Import dataflow framework components for enhanced HKG
try:
    from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter, validate_conversion_result
    from brainsmith.dataflow.core.dataflow_model import DataflowModel
    from brainsmith.dataflow.core.validation import ValidationSeverity
    from brainsmith.dataflow.core.class_naming import generate_class_name, generate_backend_class_name, generate_test_class_name
    DATAFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dataflow framework not available: {e}")
    DATAFLOW_AVAILABLE = False
    # Fallback class naming implementation
    def generate_class_name(kernel_name: str, prefix: str = "Auto") -> str:
        parts = kernel_name.split('_')
        camel_case = ''.join(word.capitalize() for word in parts)
        return f"{prefix}{camel_case}"


# Legacy compatibility - use new BrainsmithError framework
HardwareKernelGeneratorError = BrainsmithError

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
                raise BrainsmithError(
                    message=f"Could not create output directory {self.output_dir}",
                    context={"output_dir": str(self.output_dir), "original_error": str(e)},
                    suggestions=[
                        "Check directory permissions",
                        "Ensure parent directory exists",
                        "Verify disk space availability"
                    ]
                )


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
        
        # Generator instances (lazy initialization)
        self._hw_custom_op_generator = None
        
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
            raise RTLParsingError(
                message="Failed to parse RTL file",
                file_path=str(self.rtl_file_path),
                context={"original_error": str(e)},
                suggestions=[
                    "Check RTL syntax",
                    "Ensure ANSI-style port declarations",
                    "Verify required interfaces (ap_clk, ap_rst_n)"
                ]
            )
        except Exception as e:
            raise RTLParsingError(
                message="Unexpected error during RTL parsing",
                file_path=str(self.rtl_file_path),
                context={"original_error": str(e)},
                suggestions=[
                    "Check file format and encoding",
                    "Verify file is valid SystemVerilog",
                    "Review RTL parser debug output"
                ]
            )

    def _parse_compiler_data(self):
        """Imports and parses the compiler data Python file."""
        print(f"--- Parsing Compiler Data file: {self.compiler_data_path} ---")
        try:
            # 1. Import the module to access objects (ONNX model, functions)
            spec = importlib.util.spec_from_file_location("compiler_data", self.compiler_data_path)
            if spec is None or spec.loader is None:
                 raise BrainsmithError(
                     message="Could not create module spec for compiler data",
                     context={"compiler_data_path": str(self.compiler_data_path)},
                     suggestions=[
                         "Check file path and permissions",
                         "Ensure file is valid Python",
                         "Verify file extension is .py"
                     ]
                 )
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
             raise BrainsmithError(
                 message="Compiler data file not found",
                 context={"compiler_data_path": str(self.compiler_data_path)},
                 suggestions=[
                     "Check file path",
                     "Ensure file exists",
                     "Verify file permissions"
                 ]
             )
        except SyntaxError as e:
            raise BrainsmithError(
                message="Syntax error in compiler data file",
                context={
                    "compiler_data_path": str(self.compiler_data_path),
                    "syntax_error": str(e)
                },
                suggestions=[
                    "Check Python syntax in compiler data file",
                    "Verify proper indentation",
                    "Check for missing imports"
                ]
            )
        except ImportError as e:
            raise BrainsmithError(
                message="Failed to import compiler data module",
                context={
                    "compiler_data_path": str(self.compiler_data_path),
                    "import_error": str(e)
                },
                suggestions=[
                    "Check for missing dependencies",
                    "Verify module structure",
                    "Check Python path configuration"
                ]
            )
        except Exception as e:
            raise BrainsmithError(
                message="Unexpected error during compiler data parsing",
                context={
                    "compiler_data_path": str(self.compiler_data_path),
                    "original_error": str(e)
                },
                suggestions=[
                    "Review compiler data file structure",
                    "Check for runtime dependencies",
                    "Verify file encoding"
                ]
            )

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
            if not DATAFLOW_AVAILABLE:
                raise HardwareKernelGeneratorError("Dataflow framework required but not available")
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

    def _get_hw_custom_op_generator(self):
        """Get HWCustomOp generator instance with lazy initialization."""
        if self._hw_custom_op_generator is None:
            try:
                from .generators.hw_custom_op_generator import HWCustomOpGenerator
                self._hw_custom_op_generator = HWCustomOpGenerator()
            except ImportError as e:
                raise HardwareKernelGeneratorError(f"Could not import HWCustomOpGenerator: {e}")
        return self._hw_custom_op_generator

    def _generate_hw_custom_op(self):
        """
        Generate HWCustomOp using Phase 3 enhanced generator.
        
        Replaces inline generation with direct call to HWCustomOpGenerator.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: RTL data not parsed.")
            
        if not self.dataflow_enabled:
            raise HardwareKernelGeneratorError(
                "HWCustomOp generation requires dataflow framework. "
                "Please ensure brainsmith.dataflow is available."
            )
            
        print("--- Generating HWCustomOp Instance ---")
        
        # Get generator instance
        generator = self._get_hw_custom_op_generator()
        
        # Prepare output path
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"{class_name.lower()}.py"
        
        # Generate using Phase 3 enhanced generator
        try:
            generated_code = generator.generate_hwcustomop(
                hw_kernel=self.hw_kernel_data,
                output_path=output_file,
                class_name=class_name,
                source_file=str(self.rtl_file_path.name)
            )
            
            self.generated_files["hw_custom_op"] = output_file
            print(f"HWCustomOp generation complete. Output: {output_file}")
            return output_file
            
        except Exception as e:
            raise CodeGenerationError(
                message="HWCustomOp generation failed",
                generator_type="HWCustomOpGenerator",
                context={
                    "class_name": class_name,
                    "output_file": str(output_file),
                    "original_error": str(e)
                },
                suggestions=[
                    "Check template syntax and context",
                    "Verify output directory permissions",
                    "Review generator debug output"
                ]
            )
        
    
    def generate_auto_hwcustomop(self, template_path: str, output_path: str) -> str:
        """
        Public method for generating AutoHWCustomOp with Phase 3 enhancements.
        
        Args:
            template_path: Path to Jinja2 template file (for compatibility)
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
            
        try:
            # Get generator and generate
            generator = self._get_hw_custom_op_generator()
            
            # Extract class name from output path
            output_file = Path(output_path)
            class_name = output_file.stem.replace('_', '').title() + 'HWCustomOp'
            
            generated_code = generator.generate_hwcustomop(
                hw_kernel=self.hw_kernel_data,
                output_path=output_file,
                class_name=class_name,
                source_file=str(self.rtl_file_path.name)
            )
            
            print(f"AutoHWCustomOp generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            raise CodeGenerationError(
                message="AutoHWCustomOp generation failed",
                generator_type="HWCustomOpGenerator",
                context={
                    "output_path": output_path,
                    "original_error": str(e)
                },
                suggestions=[
                    "Check template syntax and context",
                    "Verify output directory permissions",
                    "Review generator debug output"
                ]
            )


    def _generate_rtl_backend(self):
        """
        Enhanced RTLBackend generation with dataflow modeling support.
        
        Generates RTLBackend classes with interface-wise code generation capabilities.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate RTLBackend: RTL data not parsed.")
            
        if not self.dataflow_enabled:
            print("Warning: RTLBackend generation without dataflow framework - limited functionality")
            return
            
        if not self.dataflow_interfaces or not self.dataflow_model:
            print("Warning: RTLBackend generation without dataflow model - basic generation only")
            
        print("--- Generating RTLBackend Instance ---")
        
        # Generate RTLBackend with dataflow modeling
        output_path = self._generate_auto_rtlbackend_with_dataflow()
            
        self.generated_files["rtl_backend"] = output_path
        print(f"RTLBackend generation complete. Output: {output_path}")
        
    def _generate_auto_rtlbackend_with_dataflow(self) -> Path:
        """
        Generate RTLBackend with full dataflow modeling support.
        
        Returns:
            Path to generated RTLBackend file
        """
        from jinja2 import Environment, FileSystemLoader
        
        print("Generating enhanced RTLBackend with dataflow modeling")
        
        # Build template context with dataflow information
        from datetime import datetime
        
        template_context = {
            # Kernel metadata
            "kernel_name": self.hw_kernel_data.name,
            "class_name": generate_class_name(self.hw_kernel_data.name),
            "source_file": str(self.rtl_file_path),
            "generation_timestamp": datetime.now().isoformat(),
            
            # RTL Parser data
            "rtl_parameters": self.hw_kernel_data.parameters,
            "rtl_interfaces": self.hw_kernel_data.interfaces,
            "rtl_pragmas": self.hw_kernel_data.pragmas,
            
            # Dataflow framework data
            "dataflow_interfaces": self.dataflow_interfaces or [],
            "dataflow_model": self.dataflow_model,
            
            # Compiler data
            "compiler_data_available": self.compiler_data_module is not None,
        }
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters for template processing
        env.filters['list'] = list
        env.filters['number'] = lambda x: isinstance(x, (int, float))
        
        # Load and render the RTLBackend template
        template = env.get_template("rtl_backend.py.j2")
        generated_code = template.render(**template_context)
        
        # Generate output file with proper class naming
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"{class_name.lower()}_rtlbackend.py"
        
        # Write generated code
        with open(output_file, 'w') as f:
            f.write(generated_code)
        
        print(f"RTLBackend generated: {output_file}")
        return output_file


    def _generate_test_suite(self):
        """
        Generate comprehensive test suite with dataflow modeling support.
        
        Generates test suites for AutoHWCustomOp and RTLBackend validation.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate test suite: RTL data not parsed.")
            
        print("--- Generating Test Suite ---")
        
        # Generate test suite with dataflow modeling
        output_path = self._generate_auto_test_suite_with_dataflow()
            
        self.generated_files["test_suite"] = output_path
        print(f"Test suite generation complete. Output: {output_path}")
        
    def _generate_auto_test_suite_with_dataflow(self) -> Path:
        """
        Generate test suite with full dataflow modeling support.
        
        Returns:
            Path to generated test suite file
        """
        from jinja2 import Environment, FileSystemLoader
        
        print("Generating enhanced test suite with dataflow modeling")
        
        # Build template context with dataflow information
        from datetime import datetime
        
        template_context = {
            # Kernel metadata
            "kernel_name": self.hw_kernel_data.name,
            "class_name": generate_class_name(self.hw_kernel_data.name),
            "source_file": str(self.rtl_file_path),
            "generation_timestamp": datetime.now().isoformat(),
            
            # RTL Parser data
            "rtl_parameters": self.hw_kernel_data.parameters,
            "rtl_interfaces": self.hw_kernel_data.interfaces,
            "rtl_pragmas": self.hw_kernel_data.pragmas,
            
            # Dataflow framework data
            "dataflow_interfaces": self.dataflow_interfaces or [],
            "dataflow_model": self.dataflow_model,
            
            # Compiler data
            "compiler_data_available": self.compiler_data_module is not None,
        }
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters for template processing
        env.filters['list'] = list
        env.filters['number'] = lambda x: isinstance(x, (int, float))
        
        # Load and render the test suite template
        template = env.get_template("test_suite.py.j2")
        generated_code = template.render(**template_context)
        
        # Generate output file with proper class naming
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"test_{class_name.lower()}.py"
        
        # Write generated code
        with open(output_file, 'w') as f:
            f.write(generated_code)
        
        print(f"Test suite generated: {output_file}")
        return output_file

    def _generate_documentation(self):
        """
        Generate documentation with dataflow modeling information.
        
        Creates comprehensive documentation including interface specifications,
        usage examples, and dataflow modeling details.
        """
        if not self.hw_kernel_data:
            raise HardwareKernelGeneratorError("Cannot generate documentation: RTL data not parsed.")
            
        print("--- Generating Documentation ---")
        
        # Generate documentation file
        output_path = self._generate_auto_documentation_with_dataflow()
            
        self.generated_files["documentation"] = output_path
        print(f"Documentation generation complete. Output: {output_path}")
        
    def _generate_auto_documentation_with_dataflow(self) -> Path:
        """
        Generate documentation with dataflow modeling information.
        
        Returns:
            Path to generated documentation file
        """
        class_name = generate_class_name(self.hw_kernel_data.name)
        output_file = self.output_dir / f"{class_name.lower()}_README.md"
        
        # Build documentation content
        doc_content = f"""# {class_name} - Auto-Generated HWCustomOp

## Overview

This document describes the auto-generated HWCustomOp implementation for `{self.hw_kernel_data.name}`.

**Source RTL:** `{self.rtl_file_path}`
**Generated Classes:**
- `{class_name}` - Main HWCustomOp implementation
- `{class_name}RTLBackend` - RTL backend for synthesis
- `Test{class_name}` - Comprehensive test suite

## Interface Specification

"""
        
        if self.dataflow_interfaces:
            doc_content += f"**Total Interfaces:** {len(self.dataflow_interfaces)}\n\n"
            
            # Group interfaces by type
            input_interfaces = [iface for iface in self.dataflow_interfaces
                              if hasattr(iface, 'interface_type') and
                              str(iface.interface_type).endswith('INPUT')]
            output_interfaces = [iface for iface in self.dataflow_interfaces
                               if hasattr(iface, 'interface_type') and
                               str(iface.interface_type).endswith('OUTPUT')]
            weight_interfaces = [iface for iface in self.dataflow_interfaces
                               if hasattr(iface, 'interface_type') and
                               str(iface.interface_type).endswith('WEIGHT')]
            config_interfaces = [iface for iface in self.dataflow_interfaces
                               if hasattr(iface, 'interface_type') and
                               str(iface.interface_type).endswith('CONFIG')]
            
            # Document each interface type
            for interface_group, group_name in [
                (input_interfaces, "Input Interfaces"),
                (output_interfaces, "Output Interfaces"),
                (weight_interfaces, "Weight Interfaces"),
                (config_interfaces, "Configuration Interfaces")
            ]:
                if interface_group:
                    doc_content += f"### {group_name}\n\n"
                    for iface in interface_group:
                        doc_content += f"- **{iface.name}**\n"
                        doc_content += f"  - Type: {iface.interface_type}\n"
                        doc_content += f"  - Dimensions: qDim={iface.qDim}, tDim={iface.tDim}, sDim={iface.sDim}\n"
                        if hasattr(iface.dtype, 'finn_type'):
                            doc_content += f"  - Data Type: {iface.dtype.finn_type}\n"
                        if iface.constraints:
                            doc_content += f"  - Constraints: {iface.constraints}\n"
                        doc_content += "\n"
        else:
            doc_content += "No dataflow interfaces detected.\n\n"
        
        doc_content += f"""
## Usage Example

```python
from {class_name.lower()} import {class_name}
from finn.core.modelwrapper import ModelWrapper

# Create ONNX model with {class_name} node
# ... (model creation code)

# Get the node and create HWCustomOp instance
node = model.get_nodes_by_op_type("{class_name}")[0]
hw_op = {class_name}(node)

# Configure parallelism and datatypes
"""

        if self.dataflow_interfaces:
            for iface in self.dataflow_interfaces:
                if hasattr(iface, 'constraints') and iface.constraints and 'parallelism' in iface.constraints:
                    doc_content += f'hw_op.set_nodeattr("{iface.name}_parallel", 4)\n'
                if hasattr(iface.dtype, 'finn_type'):
                    doc_content += f'hw_op.set_nodeattr("{iface.name}_dtype", "{iface.dtype.finn_type}")\n'

        doc_content += f"""
# Verify node configuration
hw_op.verify_node()

# Get resource estimates
bram_usage = hw_op.bram_estimation()
lut_usage = hw_op.lut_estimation()
dsp_usage = hw_op.dsp_estimation("xcvu9p-flga2104-2-i")

print(f"Resource estimates - BRAM: {{bram_usage}}, LUT: {{lut_usage}}, DSP: {{dsp_usage}}")
```

## Generated Files

- `{class_name.lower()}.py` - Main HWCustomOp implementation
- `{class_name.lower()}_rtlbackend.py` - RTL backend implementation
- `test_{class_name.lower()}.py` - Comprehensive test suite
- `{class_name.lower()}_README.md` - This documentation file

## Resource Estimation

The generated classes include automatic resource estimation based on interface configuration:

- **BRAM Estimation:** Based on weight interface storage requirements and parallelism
- **LUT Estimation:** Based on interface complexity and control logic requirements
- **DSP Estimation:** Based on arithmetic operations and datatype bitwidths

Estimation modes:
- `automatic` - Balanced estimation (default)
- `conservative` - Higher resource estimates for safety margin
- `optimistic` - Lower resource estimates assuming optimal implementation

## Testing

Run the generated test suite:

```bash
pytest test_{class_name.lower()}.py -v
```

The test suite covers:
- Basic functionality and node creation
- Datatype constraint validation
- Parallelism configuration testing
- Resource estimation validation
- RTL backend integration
- End-to-end inference testing (when RTL simulation available)

## Interface-Wise Dataflow Modeling

This implementation uses the Interface-Wise Dataflow Modeling Framework which provides:

- **Unified Computational Model:** Consistent interface abstraction across input, output, weight, and config interfaces
- **Constraint Validation:** Automatic validation of datatype and parallelism constraints
- **Resource Estimation:** Interface-aware resource estimation algorithms
- **Template-Based Generation:** Production-quality code generation from RTL specifications

For more information about the framework, see the main documentation.
"""
        
        # Write documentation file
        with open(output_file, 'w') as f:
            f.write(doc_content)
        
        print(f"Documentation generated: {output_file}")
        return output_file

    def generate_complete_package(self, output_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        Generate complete package of all files for the kernel.
        
        This is the main multi-file generation method that coordinates
        generation of AutoHWCustomOp, RTLBackend, test suite, and documentation.
        
        Args:
            output_dir: Optional override for output directory
            
        Returns:
            Dictionary mapping file types to generated file paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        print("--- Generating Complete Package ---")
        
        # Ensure all prerequisite data is parsed
        if not self.hw_kernel_data:
            self._parse_rtl()
        if not self.compiler_data_module:
            self._parse_compiler_data()
        if not self.dataflow_model and self.dataflow_enabled:
            self._build_dataflow_model()
            
        # Generate all components
        package_files = {}
        
        try:
            # Generate AutoHWCustomOp
            hwcustomop_path = self._generate_hw_custom_op()
            package_files["hwcustomop"] = hwcustomop_path
            
            # Generate RTLBackend
            rtlbackend_path = self._generate_auto_rtlbackend_with_dataflow()
            package_files["rtlbackend"] = rtlbackend_path
            
            # Generate test suite
            test_path = self._generate_auto_test_suite_with_dataflow()
            package_files["test_suite"] = test_path
            
            # Generate documentation
            doc_path = self._generate_auto_documentation_with_dataflow()
            package_files["documentation"] = doc_path
            
            # Generate RTL template if not already done
            if "rtl_template" not in self.generated_files:
                self._generate_rtl_template()
                package_files["rtl_template"] = self.generated_files["rtl_template"]
            
            print(f"--- Complete Package Generated ---")
            print(f"Package contents ({len(package_files)} files):")
            for file_type, file_path in package_files.items():
                print(f"  {file_type}: {file_path}")
            
            # Update generated files registry
            self.generated_files.update(package_files)
            
            return package_files
            
        except Exception as e:
            raise HardwareKernelGeneratorError(f"Complete package generation failed: {e}")


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
            ("generate_test_suite", self._generate_test_suite),
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
            raise HardwareKernelGeneratorError(
                message=f"Unexpected error in phase '{name}': {e}",
                context={'phase': name, 'original_error': str(e)}
            )


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
