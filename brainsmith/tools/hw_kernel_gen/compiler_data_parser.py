############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import ast
import inspect

class CompilerDataParser:
    """
    Parses a Python file (expected to contain compiler-specific data and functions)
    using the AST module to extract function definitions, class methods, and import statements.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.parsed_data = {
            "functions": {}, # Stores extracted function source code
            "class_methods": {}, # Stores extracted class methods {<ClassName>: {<method_name>: <source>}}
            "variables": {}, # For top-level variable assignments if needed later
            "imports_str": "" # Stores all top-level import statements as a string
        }
        self._parse_file()

    def _parse_file(self):
        """
        Reads and parses the Python file, extracting functions, class methods, and imports.
        """
        try:
            with open(self.file_path, "r") as source_file:
                source_code = source_file.read()
            tree = ast.parse(source_code, filename=self.file_path)
            
            import_statements = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_statements.append(ast.get_source_segment(source_code, node))
                elif isinstance(node, ast.FunctionDef):
                    # Store top-level function
                    function_name = node.name
                    self.parsed_data["functions"][function_name] = ast.get_source_segment(source_code, node)
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    self.parsed_data["class_methods"][class_name] = {}
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef): # Method in a class
                            method_name = class_node.name
                            # Get original source, including decorators
                            method_source = ast.get_source_segment(source_code, class_node)
                            self.parsed_data["class_methods"][class_name][method_name] = method_source
            
            self.parsed_data["imports_str"] = "\n".join(import_statements)

        except FileNotFoundError:
            # Handle cases where the compiler_data.py might be optional or not found
            # For now, we can let it raise or log a warning.
            # Depending on requirements, this could be a silent failure if the file is optional.
            print(f"Warning: Compiler data file not found at {self.file_path}")
            # Or raise an error if it\'s mandatory:
            # raise
        except Exception as e:
            print(f"Error parsing compiler data file {self.file_path}: {e}")
            # raise

    def get_function_source(self, function_name: str) -> str | None:
        """
        Retrieves the source code of a top-level function.
        """
        return self.parsed_data["functions"].get(function_name)

    def get_class_method_source(self, class_name: str, method_name: str) -> str | None:
        """
        Retrieves the source code of a method from a specific class.
        """
        if class_name in self.parsed_data["class_methods"]:
            return self.parsed_data["class_methods"][class_name].get(method_name)
        return None

    def get_all_class_methods(self, class_name: str) -> dict | None:
        """
        Retrieves all method sources for a given class.
        """
        return self.parsed_data["class_methods"].get(class_name)

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a dummy compiler_data.py
    dummy_compiler_data_content = """
import os
import sys
from my_module import specific_function, AnotherClass

class MyCompilerData:
    def __init__(self, param1):
        self.param1 = param1
        self.some_data = "initialized"

    def custom_logic_method(self, x, y):
        \\\"\\\"\\\"This is a custom method.\\\"\\\"\\\"
        # Some complex logic
        if x > y:
            return x - y
        else:
            return x + y + self.param1

    def another_method(self):
        return "another method"

def top_level_helper_function(a, b):
    # A helper
    return a * b

# Another import somewhere else
import numpy as np
"""
    dummy_file_path = "/tmp/dummy_compiler_data.py"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_compiler_data_content)

    parser = CompilerDataParser(dummy_file_path)
    print("Parsed Data:", parser.parsed_data)
    
    print("\n--- Extracted Imports ---")
    print(parser.parsed_data.get("imports_str"))

    custom_method_src = parser.get_class_method_source("MyCompilerData", "custom_logic_method")
    if custom_method_src:
        print("\\nSource of MyCompilerData.custom_logic_method:")
        print(custom_method_src)

    helper_func_src = parser.get_function_source("top_level_helper_function")
    if helper_func_src:
        print("\\nSource of top_level_helper_function:")
        print(helper_func_src)
    
    # Clean up dummy file
    import os
    os.remove(dummy_file_path)
