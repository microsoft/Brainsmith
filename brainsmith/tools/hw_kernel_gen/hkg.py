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


    def _generate_hw_custom_op(self):
        pass # Commented out while verifying rtl template generation
        # """Generates the HWCustomOp instance file."""
        # if not self.hw_kernel_data or not self.compiler_data_module:
        #      raise HardwareKernelGeneratorError("Cannot generate HWCustomOp: Required data not parsed.")
        # print("--- Generating HWCustomOp Instance ---")
        # # Placeholder: Call the actual generator function
        # output_path = generate_hw_custom_op(self.hw_kernel_data, self.compiler_data_module, self.output_dir)
        # self.generated_files["hw_custom_op"] = output_path
        # print(f"HWCustomOp generation placeholder complete. Output: {output_path}")


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
