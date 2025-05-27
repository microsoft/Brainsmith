# Plan: Fix Parameter Comment Association

## Current Issue
The parameter parser is incorrectly associating comments with parameters by looking forward instead of backward, causing documentation comments to be attached to the wrong parameters.

## Proposed Solution

### 1. Reverse Comment Collection Strategy
- Current: Looks for comments after parameters and commas
- New: Look backwards from each parameter to find its documentation

### 2. Comment Priority Order
Implement comment collection in the following priority:

1. Preceding Documentation Comments
   - Look backward from current parameter
   - Collect all contiguous comments until hitting:
     - Previous parameter declaration
     - Parameter list start "("
   - Maintain comment order by inserting at start of list

2. Inline Comments (Fallback)
   - Only check if no preceding comments found
   - Parse parameter's own text for "//" comments
   - Only use first line to avoid false positives

3. Trailing Comments (Last Resort)
   - Only check if no other comments found
   - Check for comment after comma
   - Less preferred since these often belong to next parameter

### 3. Implementation Steps

1. Modify `_get_parameter_comments`:
   ```python
   def _get_parameter_comments(self, param_list: Node, param_idx: int):
       # 1. Look backward for documentation
       # 2. Try inline comments if none found
       # 3. Try trailing comments as last resort
   ```

2. Add Helper Methods:
   ```python
   def _collect_preceding_comments(self, nodes, start_idx)
   def _is_parameter_boundary(self, node)
   ```

3. Update Comment Cleaning:
   - Ensure proper handling of multi-line comments
   - Maintain original order within comment blocks
   - Strip consistent formatting

### 4. Testing Plan

1. Update existing test cases:
   - Add explicit ordering checks
   - Test multi-line comment blocks
   - Verify comment attachment to correct parameters

2. Add new test cases:
   - Mixed comment styles (//, /* */, inline)
   - Comments between parameters
   - Edge cases (empty comments, whitespace)

## Validation

1. Run existing test suite
2. Add debug logging for comment collection
3. Visual inspection of complex parameter blocks
4. Document AST patterns in test cases

## Success Criteria

1. All test cases pass
2. Comments attach to intended parameters
3. Original comment formatting preserved
4. Maintains readable AST traversal logic