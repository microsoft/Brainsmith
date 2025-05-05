# examples/inspect_ast.py
import os
import ctypes
from ctypes import c_void_p, c_char_p, py_object, pythonapi
from tree_sitter import Language, Parser

# --- Configuration ---
# Adjust this path if your grammar file is located elsewhere
# Assumes sv.so is built within the rtl_parser directory
GRAMMAR_PATH = '/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/sv.so'

# --- Complex SystemVerilog Example ---
VERILOG_CODE = """
/*
 * Multi-line comment
 * describing the module.
 */
module ComplexModule #(
    parameter int DATA_WIDTH = 32, // Parameter with type and default
    parameter CLK_FREQ_MHZ = 100,
    localparam ADDR_WIDTH = clog2(DATA_WIDTH) // Local parameter
) (
    // ANSI Ports
    input  logic clk, // Single bit input
    input  logic rst_n,
    output logic [DATA_WIDTH-1:0] data_out, // Parametric width output
    input  logic [DATA_WIDTH-1:0] data_in,
    inout  wire  [7:0]          tristate_bus, // Inout port
    output logic                valid_out,
    input                       enable_in, // Implicit type (wire)

    // Interface-like ports (example)
    input  axi_if.master        axi_master_port, // Interface port
    output logic                irq // Another single bit output
);

// Non-ANSI style declarations (less common with ANSI header, but possible)
// These should ideally not be picked up as *new* ports if already in ANSI list
// input logic clk; // Redundant declaration
// output logic valid_out;

// Internal signals
logic [ADDR_WIDTH-1:0] internal_addr;
reg   [DATA_WIDTH-1:0] data_reg;
wire                   internal_enable;

// Assignments and logic (simplified)
assign internal_enable = enable_in && !rst_n;
assign data_out = data_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_reg <= '0;
    end else if (internal_enable) begin
        data_reg <= data_in;
    end
end

// Function for clog2 (simplified, assumes available)
function integer clog2;
    input integer value;
    integer i = 0;
    begin
        while (2**i < value) begin
            i = i + 1;
        end
        return i;
    end
endfunction

endmodule // ComplexModule
"""

# --- AST Traversal Function ---
def print_ast(node, indent="", level=0, max_depth=10):
    """Recursively prints the AST structure."""
    if level > max_depth:
        print(f"{indent}...")
        return

    node_type = node.type
    node_text = node.text.decode('utf8').strip().replace('\n', '\\n')
    # Limit text length for readability
    if len(node_text) > 60:
        node_text = node_text[:57] + "..."

    print(f"{indent}Type: {node_type:<25} Text: '{node_text}'")

    for i, child in enumerate(node.children):
        print_ast(child, indent + "|  ", level + 1, max_depth)

# --- Main Execution ---
if __name__ == "__main__":
    GRAMMAR_PATH = '/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/sv.so'
    if not os.path.exists(GRAMMAR_PATH):
        print(f"Error: Grammar file not found at {GRAMMAR_PATH}")
        print("Please ensure the tree-sitter Verilog grammar is built.")
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

    tree = parser.parse(bytes(VERILOG_CODE, "utf8"))
    root_node = tree.root_node

    print("--- Abstract Syntax Tree (AST) ---")
    print_ast(root_node)

    # Check for syntax errors
    if root_node.has_error:
        print("\n--- Syntax Errors Detected ---")
        # Simple BFS to find the first error node
        queue = [root_node]
        found_error = False
        while queue and not found_error:
            current_node = queue.pop(0)
            if current_node.type == 'ERROR':
                print(f"Error Node found at line {current_node.start_point[0]+1}: Text='{current_node.text.decode()}'")
                found_error = True
            elif current_node.has_error: # Check if children might contain error
                 queue.extend(current_node.children) # Add children in reverse for DFS-like error finding