# RTL Parser Design Analysis (May 4, 2025)

## 1. Overall Goal

The RTL Parser component aims to analyze SystemVerilog source files, extract key hardware design information (module parameters, ports, pragmas), identify standard interfaces (AXI-Stream, AXI-Lite, Global Control), validate them, and structure this information into a `HWKernel` object. This object serves as the input for the downstream Hardware Kernel Generator (HKG) process, enabling the automated generation of hardware wrappers and integration logic.

## 2. Component Breakdown

The parser is implemented across several Python modules, each with a specific responsibility:

*   **`data.py`**: Defines the core data structures using `@dataclass`.
    *   `Direction`: Enum for port directions (INPUT, OUTPUT, INOUT).
    *   `Parameter`: Represents a module parameter (name, type, default value). Includes basic validation in `__post_init__`.
    *   `Port`: Represents a module port (name, direction, width string). Includes basic validation in `__post_init__`.
    *   `Pragma`: Represents a parsed `@brainsmith` pragma (type, inputs, line number).
    *   `ModuleSummary`: (Currently empty) Intended to hold summary info before full `HWKernel` creation.
    *   `HWKernel`: (Currently empty) The final container object for all extracted and processed information.
*   **`interface_types.py`**: Defines structures related to interface identification and validation.
    *   `InterfaceType`: Enum for recognized interface types (GLOBAL_CONTROL, AXI_STREAM, AXI_LITE, UNKNOWN).
    *   `ValidationResult`: Simple dataclass to hold validation status (bool) and an optional message.
    *   `PortGroup`: Represents a collection of related `Port` objects tentatively identified as belonging to a specific `InterfaceType`. Holds ports keyed by signal suffix (e.g., `_TDATA`) or full name.
    *   `Interface`: Represents a *validated* interface, containing its name, type, constituent ports, validation result, and metadata.
*   **`pragma.py`**: Handles the parsing and initial processing of `@brainsmith` pragmas found in comments.
    *   `PragmaType`: Enum defining valid pragma types (`TOP_MODULE`, `DATATYPE`, `DERIVED_PARAMETER`).
    *   `PragmaError`: Custom exception for pragma issues.
    *   `PragmaParser`: Class responsible for finding pragma comments in the AST (`parse_comment`) and dispatching to type-specific handlers (`_handle_top_module`, etc.). Handlers are currently placeholders.
    *   `extract_pragmas`: Top-level function to walk the AST and collect all `Pragma` objects using `PragmaParser`.
*   **`interface_scanner.py`**: Scans a list of `Port` objects to group them into potential `PortGroup`s based on naming conventions.
    *   Defines signal name patterns (`GLOBAL_SIGNALS`, `AXI_STREAM_SUFFIXES`, `AXI_LITE_WRITE_SIGNALS`, etc.).
    *   `InterfaceScanner`: Class containing the `scan` method.
    *   `scan`: Iterates through ports, attempting to match them against known patterns (global signals, AXI-Stream prefixes/suffixes, AXI-Lite prefixes). Creates `PortGroup` objects for matched sets. Returns identified groups and any remaining ungrouped ports.
*   **`protocol_validator.py`**: Validates that an identified `PortGroup` adheres to the specific signal requirements of its `InterfaceType`.
    *   `ProtocolValidator`: Class containing validation logic.
    *   `_check_required_signals`: Helper to find missing required signals based on specifications.
    *   `_validate_port_properties`: Helper to check individual port properties (e.g., direction). Currently only checks direction.
    *   `validate_global_signals`, `validate_axi_stream`, `validate_axi_lite`: Type-specific validation methods using the helpers and signal definitions imported from `interface_scanner`.
    *   `validate`: Main entry point that dispatches to the appropriate type-specific validation method based on the `PortGroup`'s `interface_type`.
*   **`interface_builder.py`**: Orchestrates the interface identification and validation process.
    *   `InterfaceBuilder`: Class that uses `InterfaceScanner` and `ProtocolValidator`.
    *   `build_interfaces`: Takes a list of `Port` objects.
        1.  Calls `InterfaceScanner.scan` to get initial `PortGroup`s.
        2.  Iterates through the identified groups.
        3.  Calls `ProtocolValidator.validate` for each group.
        4.  If valid, creates a final `Interface` object (placeholder logic).
        5.  If invalid or the group lacks a name, adds the group's ports to the list of unassigned ports.
        6.  Returns a dictionary of validated `Interface` objects and a list of unassigned `Port`s.
*   **`parser.py`**: The main entry point, responsible for using `tree-sitter` to parse the SystemVerilog file and orchestrate the extraction process.
    *   `ParserError`, `SyntaxError`: Custom exceptions.
    *   `RTLParser`: Main class.
        *   `__init__`: Loads the `tree-sitter` grammar.
        *   `parse_file`: Reads the file, parses it into an AST using `tree-sitter`. Checks for syntax errors. Finds module nodes. Selects the target module (potentially using `TOP_MODULE` pragma). Extracts parameters, ports (using helper methods like `_extract_module_header`, which seems incomplete or needs integration with port parsing logic), and pragmas (using `extract_pragmas`). Placeholder for calling `InterfaceBuilder`. Placeholder for constructing the final `HWKernel` object.
        *   Helper methods (`_find_first_error_node`, `_find_module_nodes`, `_select_target_module`, `_extract_module_header`) for AST traversal and selection.

## 3. Data Flow and Interactions

1.  **Input**: SystemVerilog file path.
2.  **`RTLParser.parse_file`**:
    *   Reads the file content.
    *   Uses `tree-sitter` to generate an Abstract Syntax Tree (AST).
    *   Checks for basic syntax errors (`_find_first_error_node`).
    *   Identifies module declaration nodes (`_find_module_nodes`).
    *   Calls `extract_pragmas` (from `pragma.py`) to walk the AST and collect all `Pragma` objects defined in `data.py`.
    *   Uses pragmas (if `TOP_MODULE` exists) or heuristics to select the target module node (`_select_target_module`).
    *   **(Needs Implementation/Refinement)** Traverses the target module's AST to extract raw port and parameter information, creating `Port` and `Parameter` objects (defined in `data.py`). The existing `_extract_module_header` seems like a starting point but needs integration with detailed port/parameter parsing logic potentially referencing `rtl_parser_ast_analysis.md`.
    *   Creates a list of extracted `Port` objects.
    *   Instantiates `InterfaceBuilder`.
    *   Calls `InterfaceBuilder.build_interfaces`, passing the list of `Port` objects.
3.  **`InterfaceBuilder.build_interfaces`**:
    *   Instantiates `InterfaceScanner`.
    *   Calls `InterfaceScanner.scan`, passing the `Port` list.
4.  **`InterfaceScanner.scan`**:
    *   Iterates through `Port`s, matching names against predefined patterns (`GLOBAL_SIGNALS`, `AXI_STREAM_SUFFIXES`, `AXI_LITE_WRITE_SIGNALS`, etc.).
    *   Groups matched `Port`s into `PortGroup` objects (defined in `interface_types.py`), assigning an `InterfaceType`.
    *   Returns identified `PortGroup`s and remaining unassigned `Port`s.
5.  **`InterfaceBuilder.build_interfaces` (continued)**:
    *   Instantiates `ProtocolValidator`.
    *   Iterates through the received `PortGroup`s.
    *   For each `PortGroup`, calls `ProtocolValidator.validate`.
6.  **`ProtocolValidator.validate`**:
    *   Dispatches to the appropriate validation method (`validate_axi_stream`, etc.) based on `PortGroup.interface_type`.
    *   These methods use signal definitions (imported from `interface_scanner.py`) and helpers (`_check_required_signals`, `_validate_port_properties`) to check for required signals and correct properties (like direction).
    *   Returns a `ValidationResult` (defined in `interface_types.py`).
7.  **`InterfaceBuilder.build_interfaces` (continued)**:
    *   If `ValidationResult.valid` is true, creates an `Interface` object (defined in `interface_types.py`) from the `PortGroup` (logic is currently a placeholder).
    *   If invalid, adds the ports from the `PortGroup` back to the list of unassigned ports.
    *   Returns the dictionary of validated `Interface` objects and the final list of unassigned `Port`s.
8.  **`RTLParser.parse_file` (continued)**:
    *   Receives the validated interfaces and unassigned ports.
    *   **(Needs Implementation)** Populates the `HWKernel` object (defined in `data.py`) with the extracted parameters, validated interfaces, unassigned ports, and processed pragma data.
    *   Returns the populated `HWKernel` object.

## 4. Key Features & Design Choices

*   **Modularity**: Functionality is well-separated into distinct modules (data, parsing, scanning, validation, building).
*   **AST-Based Parsing**: Leverages `tree-sitter` for robust parsing of SystemVerilog syntax, separating parsing from interpretation.
*   **Dataclasses**: Uses Python dataclasses for clear, concise, and type-hinted data structures (`Port`, `Parameter`, `Interface`, etc.). Basic validation is included in `__post_init__`.
*   **Interface Identification**: Employs a two-stage process:
    1.  **Scanning (`InterfaceScanner`)**: Groups ports based on naming conventions (heuristic).
    2.  **Validation (`ProtocolValidator`)**: Rigorously checks if the grouped ports meet the specific protocol requirements (AXI, Global).
*   **Pragma System**: Dedicated module (`pragma.py`) for extracting and potentially processing metadata provided in comments. Handlers are defined but need implementation.
*   **Logging**: Uses Python's `logging` module, allowing for configurable verbosity (though setup seems basic currently).
*   **Error Handling**: Defines custom exceptions (`ParserError`, `SyntaxError`, `PragmaError`) but detailed error reporting and recovery mechanisms need further development.

## 5. Current Status & Completeness

*   **Core Data Structures (`data.py`, `interface_types.py`)**: Mostly defined, though `HWKernel` and `ModuleSummary` are empty placeholders. Basic validation exists in `__post_init__`.
*   **Pragma Parsing (`pragma.py`)**: Framework exists to find pragmas (`extract_pragmas`, `PragmaParser`). `PragmaType` enum is defined. Specific handler logic (`_handle_...`) is missing.
*   **Interface Scanning (`interface_scanner.py`)**: Logic to group ports based on defined AXI/Global patterns seems implemented. Signal definitions are present.
*   **Protocol Validation (`protocol_validator.py`)**: Logic to validate `PortGroup`s against the defined signal requirements (presence, direction) appears implemented.
*   **Interface Building (`interface_builder.py`)**: Orchestrates scanning and validation. The final step of creating the `Interface` object from a validated `PortGroup` is a placeholder. Handling of unassigned ports exists.
*   **Main Parser (`parser.py`)**: Initializes `tree-sitter`, parses the file, finds modules, and calls pragma extraction. Logic for extracting detailed port/parameter information from the AST needs implementation/integration (connecting AST nodes to `Port`/`Parameter` creation). Integration with `InterfaceBuilder` exists but the final step of populating `HWKernel` is missing. Module selection logic based on pragmas is present.

## 6. Extensibility (Based on Code and Analysis Docs)

*   **New Pragma Types**: Add to `PragmaType` enum (`pragma.py`), implement a corresponding `_handle_...` method in `PragmaParser`, and register it in the (currently implicit) handler dispatch logic.
*   **New Interface Types**: Add to `InterfaceType` enum (`interface_types.py`), add corresponding signal patterns/definitions in `interface_scanner.py`, implement a `validate_...` method in `ProtocolValidator`, and update the dispatch logic in `ProtocolValidator.validate`. Update `InterfaceScanner.scan` to recognize the new patterns.
*   **Parameter/Port Enhancements**: Modify `Parameter`/`Port` dataclasses (`data.py`). Update AST parsing logic in `parser.py` to extract any new attributes. Update validation logic if necessary.
*   **Validation Rules**: Modify signal definitions in `interface_scanner.py` or validation logic within `ProtocolValidator` methods.

## 7. Potential Issues / Areas for Improvement

*   **Detailed Port/Parameter Extraction**: The logic in `parser.py` to traverse the AST and create `Port` and `Parameter` objects needs full implementation, likely leveraging insights from `rtl_parser_ast_analysis.md` regarding node types and locations for width, type, direction, etc.
*   **Pragma Handler Implementation**: The specific logic within `_handle_top_module`, `_handle_datatype`, etc. in `pragma.py` needs to be written to process the pragma inputs and potentially store results.
*   **`HWKernel` Population**: The final step in `parser.py` to gather all extracted/processed data (parameters, interfaces, pragmas, unassigned ports) and populate the `HWKernel` object needs implementation.
*   **Error Reporting**: While custom exceptions exist, more granular error reporting (e.g., specific line numbers for validation failures, clearer messages) would be beneficial. Handling syntax errors beyond just finding the first one could be improved.
*   **AST Traversal Helpers**: The helpers mentioned in `rtl_parser_analysis.md` (`_find_child`, `_has_text`, etc.) might need to be implemented or refined within `parser.py` to support robust port/parameter extraction.
*   **Configuration/Metadata**: How metadata (like AXI data widths inferred during scanning/validation) is stored and passed needs clarification (e.g., populating `PortGroup.metadata` and `Interface.metadata`).
*   **Completeness of Signal Definitions**: Ensure the signal definitions in `interface_scanner.py` cover all necessary cases and variations according to the AXI specifications being targeted.
*   **Testing**: Requires comprehensive unit and integration tests covering different SystemVerilog constructs, interface variations, pragmas, and error conditions.
