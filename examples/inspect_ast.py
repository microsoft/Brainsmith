# examples/inspect_ast.py
import os
import ctypes
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from tree_sitter import Language, Parser

# --- Configuration ---
# Adjust this path if your grammar file is located elsewhere
# Assumes sv.so is built within the rtl_parser directory
GRAMMAR_PATH = '/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/sv.so'
# Path to the SystemVerilog file to parse
TARGET_SV_FILE = '/home/tafk/dev/brainsmith/examples/thresholding/thresholding_axi.sv'

# --- AST Traversal Function ---
def print_ast(node, indent="", level=0): # Removed max_depth limit
    """Recursively prints the AST structure."""
    # Removed depth check

    node_type = node.type
    node_text = node.text.decode('utf8').strip().replace('\n', '\\n')
    # Limit text length for readability
    if len(node_text) > 80: # Increased limit slightly
        node_text = node_text[:77] + "..."

    print(f"{indent}Type: {node_type:<25} Text: '{node_text}'")

    for i, child in enumerate(node.children):
        print_ast(child, indent + "|  ", level + 1) # Removed max_depth argument

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(GRAMMAR_PATH):
        print(f"Error: Grammar file not found at {GRAMMAR_PATH}")
        print("Please ensure the tree-sitter Verilog grammar is built.")
        exit(1)

    if not os.path.exists(TARGET_SV_FILE):
        print(f"Error: Target SystemVerilog file not found at {TARGET_SV_FILE}")
        exit(1)

    # Read the target SystemVerilog file
    try:
        with open(TARGET_SV_FILE, 'r', encoding='utf8') as f:
            source_code = f.read()
        print(f"Successfully read {TARGET_SV_FILE}")
    except Exception as e:
        print(f"Error reading {TARGET_SV_FILE}: {e}")
        exit(1)

    # 1. Load the shared object
    lib = ctypes.cdll.LoadLibrary(GRAMMAR_PATH)

    # 2. Get language pointer
    lang_ptr = lib.tree_sitter_verilog
    lang_ptr.restype = c_void_p
    lang_ptr = lang_ptr()

    # 3. Create Python capsule
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = (c_void_p, c_char_p, c_void_p)
    capsule = PyCapsule_New(lang_ptr, b"tree_sitter.Language", None)

    # 4. Create parser with language
    language = Language(capsule)
    parser = Parser(language)

    # Parse the source code read from the file
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node

    print("\n--- Abstract Syntax Tree (AST) ---")
    print_ast(root_node) # Removed max_depth argument

    # Check for syntax errors
    if root_node.has_error:
        print("\n--- Syntax Errors Detected ---")
        # Simple BFS to find the first error node
        queue = [root_node]
        found_error = False
        while queue and not found_error:
            current_node = queue.pop(0)
            # Check for ERROR node type or if the node itself has an error flag
            if current_node.type == 'ERROR' or (current_node.has_error and not current_node.children):
                print(f"Error Node found near line {current_node.start_point[0]+1}: Type='{current_node.type}', Text='{current_node.text.decode()}'")
                found_error = True
            elif current_node.has_error: # Check if children might contain error
                 # Add children in standard order for BFS
                 queue.extend(current_node.children)