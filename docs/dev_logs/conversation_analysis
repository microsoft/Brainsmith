<conversation-summary>
Okay, here is a summary of our conversation history:

**TASK DESCRIPTION:**
*   Implement an interface analysis system (Global Control, AXI-Stream, AXI-Lite) for the RTL Parser.
*   Review the existing RTL Parser implementation (`parser.py`, `pragma.py`) against requirements.
*   Document the new interface analysis system and the AST structure.
*   Implement improvements: Correct pragma handling, add placeholders, refine error handling/logging in `parser.py`.
*   Refactor old tests (`test_interface.py`, `test_parser.py`, `test_pragma.py`) into a new comprehensive suite (`test_rtl_parser.py`).
*   Run the new tests and fix any failures.

**COMPLETED:**
1.  **Interface Analysis Implementation:** Successfully implemented and tested `InterfaceScanner`, `ProtocolValidator`, `InterfaceBuilder`. Integrated into `RTLParser.parse_file`. Fixed bugs identified via `pytest`. Corrected AXI-Stream direction validation logic.
2.  **Documentation:** Created `docs/analysis/interface_analysis_design.md` and `docs/analysis/rtl_parser_review_analysis.md` (`create_file` used). Attempted to create `docs/analysis/rtl_parser_ast_structure.md` but it already existed (`create_file` failed).
3.  **RTL Parser Review & Improvements:**
    *   Refactored `pragma.py` using `insert_edit_into_file` to support only required pragmas, removing obsolete ones, adding basic validation, and fixing a case-sensitivity bug.
    *   Added `kernel_parameters` and `compiler_flags` attributes to `HWKernel` in `data.py` and `TODO` comments in `parser.py` using `insert_edit_into_file`.
    *   Refined error handling & logging in `parser.py` using `insert_edit_into_file`.
    *   Moved helper functions (`_debug_node`, `_parse_port_declaration`, `_parse_parameter_declaration`, `_extract_module_header`, `_find_child`, `_has_text`) from `interface.py` into `parser.py` using `insert_edit_into_file`. Added `import re`.
    *   Deleted `interface.py` using `run_in_terminal`.
    *   Refactored `_select_target_module` in `parser.py` to use `_extract_module_header` for more robust module name finding (`insert_edit_into_file`).
    *   Iteratively refactored `_parse_port_declaration` in `parser.py` to improve handling of ANSI-style ports (`insert_edit_into_file` multiple times).
    *   Iteratively refactored `_extract_module_header` in `parser.py` to improve port node discovery, including adding debug logging (`insert_edit_into_file` multiple times).
    *   Corrected the implementation of the `_find_child` helper function in `parser.py` (`insert_edit_into_file`).
4.  **Test Refactoring:**
    *   Created new comprehensive test suite `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py` using `create_file`.
5.  **Test Execution & Debugging:**
    *   Ran `pytest` multiple times (#terminalSelection).
    *   Fixed `SyntaxError` in `parser.py` and `ImportError` in `test_rtl_parser.py` (`insert_edit_into_file`).
    *   Identified and iteratively addressed multiple test failures (#terminalSelection), primarily stemming from pragma parsing, module selection, and ultimately port extraction (currently failing with "Extracted 0 ports").
    *   Added a debugging test `test_print_ast_structure` to `test_rtl_parser.py` (`insert_edit_into_file`) and modified it to use `print` for visibility (`insert_edit_into_file`).
    *   Ran the AST debug test (#terminalSelection) and analyzed the output, identifying `module_ansi_header` as the correct node type for ANSI-style module headers.

**PENDING:**
1.  **Fix Port Extraction:** Modify `_extract_module_header` in `parser.py` to search for `"module_ansi_header"` instead of `"module_header"` based on the AST analysis.
2.  **Fix Remaining Test Failures:** Address the cascading test failures (currently `test_multiple_axi_lite_interfaces` failing due to 0 ports extracted) after fixing port extraction.
3.  **Delete Old Test Files:** Once `test_rtl_parser.py` passes, delete `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_interface.py`, `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_parser.py`, and `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_pragma.py`.
4.  **Update AST Analysis Doc:** (Optional) Update `/home/tafk/dev/brainsmith/docs/analysis/rtl_parser_ast_structure.md` with the analysis findings using an edit tool if needed.

**CODE STATE (Files Discussed/Modified):**
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/interface_types.py`
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/interface_scanner.py`
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/protocol_validator.py`
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py` (heavily modified)
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py` (modified)
*   `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/__init__.py`
*   `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_interface_scanner.py`
*   `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_protocol_validator.py`
*   `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_interface_builder.py`
*   `/home/tafk/dev/brainsmith/tests/tools/hw_kernel_gen/rtl_parser/test_rtl_parser.py` (created, edited)
*   `/home/tafk/dev/brainsmith/docs/prompts/RTL_Parser-Data-Analysis.md` (read)
*   `/home/tafk/dev/brainsmith/docs/prompts/RTL_Parser-Prompt.md` (read)
*   `/home/tafk/dev/brainsmith/docs/implementation_plan/rtl_parser_data_interface_plan.md` (read)
*   `/home/tafk/dev/brainsmith/docs/analysis/interface_analysis_design.md` (created)
*   `/home/tafk/dev/brainsmith/docs/analysis/rtl_parser_review_analysis.md` (created)
*   `/home/tafk/dev/brainsmith/docs/analysis/rtl_parser_ast_structure.md` (attempted create, exists)
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/interface.py` (deleted)
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_interface.py` (to be deleted)
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_parser.py` (to be deleted)
*   `/home/tafk/dev/brainsmith/brainsmith/tools/hw_kernel_gen/rtl_parser/tests/test_pragma.py` (to be deleted)

**CHANGES (Key Edits):**
*   Created and integrated interface analysis components.
*   Added post-analysis validation checks to `RTLParser`.
*   Created documentation (`interface_analysis_design.md`, `rtl_parser_review_analysis.md`).
*   Refactored `pragma.py` for required pragmas, fixed case bug.
*   Added placeholder attributes to `HWKernel` (`data.py`) and `TODO` comments in `parser.py`.
*   Refined error handling/logging in `parser.py`.
*   Moved helper functions from `interface.py` to `parser.py`.
*   Deleted `interface.py`.
*   Created new test suite `test_rtl_parser.py`.
*   Fixed syntax and import errors in `parser.py` and `test_rtl_parser.py`.
*   Iteratively debugged and refactored `_select_target_module`, `_parse_port_declaration`, `_extract_module_header`, `_find_child` in `parser.py` to fix test failures related to module selection and port parsing.
*   Added AST debug test and analyzed output, identifying `module_ansi_header` as the correct node type.
</conversation-summary>
