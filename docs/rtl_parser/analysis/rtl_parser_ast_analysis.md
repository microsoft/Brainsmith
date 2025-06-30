# SystemVerilog AST Analysis (Tree-sitter)

Based on parsing the `ComplexModule` example using `examples/inspect_ast.py`.

**AST Analysis based on `inspect_ast.py` Output:**

1.  **Overall Structure:**
    *   The root node is `source_file`.
    *   Top-level items like comments (`comment`) and module declarations (`module_declaration`) are direct children of `source_file`.

2.  **Module Declaration (`module_declaration`):**
    *   For ANSI-style headers, the primary child is `module_ansi_header`. The `endmodule` keyword is a separate sibling node.
    *   **`module_ansi_header`:** Contains the `module_keyword`, the module name (`simple_identifier`), the parameter list (`parameter_port_list`), the port list (`list_of_port_declarations`), and the closing semicolon.

3.  **Parameters (`parameter_port_list`):**
    *   Starts with `#` and enclosed in `()`.
    *   Contains `parameter_port_declaration` nodes, separated by `,`.
    *   **`parameter_port_declaration`:**
        *   Can contain `parameter_declaration` or `local_parameter_declaration`.
        *   **`parameter_declaration`:** Includes `parameter` keyword, optional `data_type_or_implicit` (which contains `data_type` -> `integer_atom_type` -> `int` in the example), and `list_of_param_assignments`.
        *   **`local_parameter_declaration`:** Similar structure with `localparam` keyword.
        *   **`list_of_param_assignments`:** Contains `param_assignment` nodes.
        *   **`param_assignment`:** Holds the parameter name (`simple_identifier`), `=`, and the value (`constant_param_expression` -> `constant_mintypmax_expression` -> `constant_expression`). The expression itself can be nested (e.g., `clog2(DATA_WIDTH)`).

4.  **ANSI Ports (`list_of_port_declarations`):**
    *   Starts with `(` and ends with `)`.
    *   Contains `ansi_port_declaration` nodes, separated by `,`. Comments are interspersed as `comment` nodes.
    *   **`ansi_port_declaration`:** This is the crucial node for port parsing. Its structure varies:
        *   **Variable Ports (e.g., `logic`, `reg`):**
            *   Child 1: `variable_port_header` (contains direction and type info).
            *   Child 2: Port name (`simple_identifier`).
            *   **`variable_port_header`:**
                *   Child 1: `port_direction` (e.g., `input`, `output`).
                *   Child 2: `variable_port_type` (contains base type and dimension).
                *   **`variable_port_type`:**
                    *   Child 1: `data_type` (e.g., `logic`). Contains nested types like `integer_vector_type`.
                    *   Child 2 (Optional): `packed_dimension` (e.g., `[DATA_WIDTH-1:0]`). **This confirms the dimension is a sibling of the `data_type` within `variable_port_type`**.
        *   **Net Ports (e.g., `wire`):**
            *   Child 1: `net_port_header` (contains direction and type info).
            *   Child 2: Port name (`simple_identifier`).
            *   **`net_port_header`:**
                *   Child 1: `port_direction` (e.g., `inout`).
                *   Child 2: `net_port_type` (contains base type and dimension).
                *   **`net_port_type`:**
                    *   Child 1: `net_type` (e.g., `wire`).
                    *   Child 2: `data_type_or_implicit` -> `implicit_data_type` -> `packed_dimension`. **Here, the dimension is nested differently than for `variable_port_type`**.
        *   **Implicit Type Ports (e.g., `input enable_in`):**
            *   Child 1: `net_port_header` (contains only `port_direction`).
            *   Child 2: Port name (`simple_identifier`). Type defaults to `wire`.
        *   **Interface Ports (`input axi_if.master axi_master_port`):**
            *   Parsed with an `ERROR` node for `.master`. This indicates the grammar might not fully support interface port syntax in this specific context or structure.
            *   Child 1: `net_port_header` (contains `port_direction` and `net_port_type` -> `simple_identifier` for `axi_if`).
            *   Child 2: `ERROR` node containing `.` and `simple_identifier` (`master`).
            *   Child 3: Port name (`simple_identifier` for `axi_master_port`).

5.  **Dimensions (`packed_dimension`):**
    *   Contains `[`, `constant_range` (or similar expression), and `]`.
    *   **`constant_range`:** Contains the MSB expression (`constant_expression`), `:`, and the LSB expression (`constant_expression`). Expressions can be complex (e.g., `DATA_WIDTH-1`).

6.  **Module Body Items:**
    *   Internal signal declarations (`data_declaration`, `net_declaration`) follow a similar pattern to parameters/ports, with `data_type_or_implicit`, optional dimensions, and `list_of_variable_decl_assignments`.
    *   Assignments (`continuous_assign`) contain `assign`, `list_of_net_assignments` -> `net_assignment` (with `net_lvalue`, `=`, `expression`).
    *   Procedural blocks (`always_construct`) have `always_keyword` (`always_ff`), `statement` -> `statement_item` -> `procedural_timing_control_statement` (with `event_control` like `@(...)`) and the block body (`statement_or_null` -> `seq_block`).
    *   Functions (`function_declaration`) have `function`, `function_body_declaration` (containing return type, name, port/variable declarations), `function_statement_or_null`, and `endfunction`.

**Key Takeaways for Parser Logic:**

*   **ANSI Port Width:** The `packed_dimension` node's location depends on the port type (`variable` vs. `net`).
    *   For `variable_port_type`, the dimension is a **sibling** of the `data_type` node.
    *   For `net_port_type`, the dimension seems nested within `data_type_or_implicit` -> `implicit_data_type`.
    *   The parser needs to check both locations.
*   **Data Type:** Similarly, the base data type (`logic`, `wire`, `int`) is nested within `variable_port_type` or `net_port_type`.
*   **Interface Ports:** The grammar used seems to have issues with the `interface_name.modport_name` syntax within the ANSI port list, generating an `ERROR` node. This needs special handling or a grammar update.
*   **Identifiers:** `simple_identifier` is used ubiquitously for names (modules, parameters, ports, variables). Filtering based on context (parent/sibling node types) is essential.
*   **Expressions:** Expressions within dimensions or parameter values are captured as `constant_expression` (or similar) and need to be extracted as text.
