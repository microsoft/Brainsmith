# RTL Parser Refactoring Plan

## Overview
This plan addresses recommendations 1 & 2 from the structural analysis: breaking up parser.py and reorganizing the pragma system. The goal is to improve modularity, separation of concerns, and extensibility.

## Phase 1: Break Up parser.py (Recommendation 1)

### Step 1.1: Create ast_parser.py
**Purpose**: Extract all tree-sitter AST operations into a dedicated module

1. Create new file: `brainsmith/tools/hw_kernel_gen/rtl_parser/ast_parser.py`
2. Move from parser.py:
   - Tree-sitter initialization logic (lines 73-86)
   - AST parsing methods:
     - `_find_first_error_node()` (lines 356-373)
     - `_find_module_nodes()` (lines 375-387)
     - `_find_child()` (lines 800-806)
     - `_find_children()` (lines 808-815)
     - `_debug_node()` (lines 532-546)
   - Grammar loading and parser setup
3. Create new class `ASTParser` with methods:
   - `__init__(grammar_path, debug)`
   - `parse_source(content: str) -> Tree`
   - `check_syntax_errors(tree: Tree) -> Optional[SyntaxError]`
   - `find_modules(tree: Tree) -> List[Node]`
   - Helper methods for node traversal
4. Update imports in parser.py to use ASTParser

### Step 1.2: Create component_extractor.py
**Purpose**: Extract component extraction logic (parameters, ports, module headers)

1. Create new file: `brainsmith/tools/hw_kernel_gen/rtl_parser/component_extractor.py`
2. Move from parser.py:
   - Component extraction methods:
     - `_extract_module_header()` (lines 464-530)
     - `_parse_parameter_declaration()` (lines 817-956)
     - `_parse_port_declaration()` (lines 604-761)
     - `_extract_width_from_dimension()` (lines 763-798)
     - `_extract_direction()` (lines 548-573)
     - `_find_identifiers_recursive()` (lines 575-602)
3. Create new class `ComponentExtractor` with methods:
   - `extract_module_name(module_node: Node) -> str`
   - `extract_parameters(module_node: Node) -> List[Parameter]`
   - `extract_ports(module_node: Node) -> List[Port]`
   - Helper methods for parsing declarations
4. Update imports in parser.py to use ComponentExtractor

### Step 1.3: Create workflow_orchestrator.py
**Purpose**: Extract high-level workflow coordination

1. Create new file: `brainsmith/tools/hw_kernel_gen/rtl_parser/workflow_orchestrator.py`
2. Move from parser.py:
   - Module selection logic:
     - `_select_target_module()` (lines 389-462)
   - Workflow coordination methods:
     - `_initial_parse()` (lines 100-175)
     - `_extract_kernel_components()` (lines 176-245)
     - `_apply_pragmas_to_kernel()` (lines 958-968)
     - `_apply_autolinking_to_kernel()` (lines 970-1041)
3. Create new class `WorkflowOrchestrator` with methods:
   - `select_module(modules, pragmas, target_name)`
   - `coordinate_parsing(source, target_module) -> ParsedData`
   - `apply_pragmas(kernel_metadata, pragmas)`
   - `apply_autolinking(kernel_metadata)`
4. Update parser.py to use WorkflowOrchestrator

### Step 1.4: Refactor parser.py
**Purpose**: Transform parser.py into a clean facade

1. Keep RTLParser class but simplify to:
   - `__init__()` - Initialize sub-components
   - `parse()` - Main entry point
   - `parse_file()` - File parsing convenience method
2. RTLParser becomes a coordinator that:
   - Uses ASTParser for tree-sitter operations
   - Uses ComponentExtractor for extracting components
   - Uses WorkflowOrchestrator for pipeline coordination
   - Maintains the same public API

## Phase 2: Reorganize Pragma System (Recommendation 2)

### Step 2.1: Create pragmas subdirectory
**Purpose**: Better organization of pragma-related code

1. Create directory: `brainsmith/tools/hw_kernel_gen/rtl_parser/pragmas/`
2. Add `__init__.py` with pragma type exports

### Step 2.2: Move base pragma classes
**Purpose**: Centralize base pragma logic

1. Create `pragmas/base.py`:
   - Move from data.py:
     - `Pragma` base class (lines 155-247)
     - `InterfacePragma` base class (lines 249-322)
     - `ParameterPragma` base class (lines 714-759)
     - `PragmaError` exception (lines 40-42)
   - Keep abstract methods and common functionality

### Step 2.3: Create interface.py
**Purpose**: Group all interface-related pragmas

1. Create `pragmas/interface.py`:
   - Move from data.py:
     - `DatatypePragma` (lines 346-415)
     - `DatatypeParamPragma` (lines 823-913)
     - `WeightPragma` (lines 696-712)
   - Import base classes from `pragmas.base`

### Step 2.4: Create parameter.py
**Purpose**: Group all parameter-related pragmas

1. Create `pragmas/parameter.py`:
   - Move from data.py:
     - `AliasPragma` (lines 761-821)
     - `DerivedParameterPragma` (lines 635-694)
   - Import base classes from `pragmas.base`

### Step 2.5: Create dimension.py
**Purpose**: Group dimension-related pragmas

1. Create `pragmas/dimension.py`:
   - Move from data.py:
     - `BDimPragma` (lines 417-570)
     - `SDimPragma` (lines 572-633)
   - Import base classes from `pragmas.base`

### Step 2.6: Create module.py
**Purpose**: Module-level pragmas

1. Create `pragmas/module.py`:
   - Move from data.py:
     - `TopModulePragma` (lines 326-344)
   - Import base classes from `pragmas.base`

### Step 2.7: Update pragma.py
**Purpose**: Update PragmaHandler to use new structure

1. Update imports to use new pragma locations
2. Ensure pragma_constructors mapping uses new imports
3. Keep PragmaHandler logic unchanged

### Step 2.8: Update data.py
**Purpose**: Clean up after pragma extraction

1. Remove all pragma classes (keep only data structures)
2. Keep:
   - Enums (Direction, PragmaType)
   - Simple data classes (Parameter, Port, PortGroup, ValidationResult)
3. Update module docstring to reflect new scope

## Phase 3: Testing and Validation

### Step 3.1: Update imports
1. Update all imports throughout the codebase
2. Ensure backward compatibility by re-exporting from __init__.py

### Step 3.2: Run existing tests
1. Run all existing RTL parser tests
2. Verify no functionality is broken
3. Check that public API remains unchanged

### Step 3.3: Create unit tests
1. Create tests for new modules:
   - `test_ast_parser.py`
   - `test_component_extractor.py`
   - `test_workflow_orchestrator.py`
2. Test each component in isolation

## Implementation Order

1. **Phase 1 First**: Break up parser.py
   - This is less risky as it's mostly moving code
   - Maintains existing functionality
   - Can be tested incrementally

2. **Phase 2 Second**: Reorganize pragmas
   - Depends on Phase 1 being complete
   - More complex due to interdependencies
   - Requires careful import management

3. **Phase 3 Throughout**: Test after each major step
   - Run tests after each file creation
   - Ensure no regression

## Risk Mitigation

1. **Git branches**: Create feature branch for each phase
2. **Incremental commits**: Commit after each successful step
3. **Backward compatibility**: Maintain public API through re-exports
4. **Testing**: Run full test suite after each phase

## Success Criteria

1. All existing tests pass
2. Parser.py reduced from 1000+ lines to ~200 lines
3. Each new module has single, clear responsibility
4. Pragma system is easier to extend
5. No changes to public API