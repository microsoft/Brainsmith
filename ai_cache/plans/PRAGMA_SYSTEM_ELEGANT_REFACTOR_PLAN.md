# Pragma System Elegant Refactor Implementation Plan

## Overview

This plan refactors the pragma system to be more tightly coupled and elegantly designed by leveraging the existing object-oriented pragma architecture. The goal is to eliminate duplicated logic and create a clean chain-of-responsibility pattern for pragma application.

## Current Problems

1. **Violation of OOP Design**: `PragmaHandler.create_interface_metadata()` bypasses existing `pragma.apply()` methods
2. **Duplicated Logic**: Logic exists in both pragma classes and PragmaHandler
3. **Inconsistent Data Flow**: Two separate processing paths that can diverge
4. **Missing Integration**: Not leveraging existing `Interface.metadata` population

## Solution Architecture

### Core Design Principles

1. **Single Responsibility**: Each pragma class handles its own application logic
2. **Chain of Responsibility**: Pragmas apply their effects in sequence
3. **Immutable Transformations**: Each pragma returns a new InterfaceMetadata
4. **Composable Effects**: Pragmas can build on each other's effects

### Key Components

1. **Enhanced Pragma Base Class**: Add interface metadata methods
2. **Pragma Chain**: Sequential application of pragma effects
3. **Simplified PragmaHandler**: Orchestration without duplication
4. **Builder Pattern**: Clean interface metadata construction

## Implementation Steps

### Phase 1: Enhance Pragma Base Class

#### Step 1.1: Add Interface Applicability Method
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Add to `Pragma` base class:
```python
def applies_to_interface(self, interface: Interface) -> bool:
    """
    Check if this pragma applies to the given interface.
    
    Base implementation returns False. Subclasses should override.
    
    Args:
        interface: Interface to check against
        
    Returns:
        bool: True if pragma applies to this interface
    """
    return False
```

#### Step 1.2: Add Interface Metadata Application Method
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Add to `Pragma` base class:
```python
def apply_to_interface_metadata(self, interface: Interface, 
                              metadata: InterfaceMetadata) -> InterfaceMetadata:
    """
    Apply pragma effects to InterfaceMetadata.
    
    Base implementation returns metadata unchanged. Subclasses should override
    to implement their specific effects.
    
    Args:
        interface: Interface this pragma applies to
        metadata: Current InterfaceMetadata to modify
        
    Returns:
        InterfaceMetadata: Modified metadata with pragma effects applied
    """
    return metadata
```

### Phase 2: Implement Pragma-Specific Logic

#### Step 2.1: Enhance DatatypePragma
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Implement both methods in `DatatypePragma`:
```python
def applies_to_interface(self, interface: Interface) -> bool:
    """Check if this DATATYPE pragma applies to the interface."""
    if not self.parsed_data:
        return False
    
    pragma_interface_name = self.parsed_data.get('interface_name')
    if not pragma_interface_name:
        return False
    
    return self._interface_names_match(pragma_interface_name, interface.name)

def apply_to_interface_metadata(self, interface: Interface, 
                              metadata: InterfaceMetadata) -> InterfaceMetadata:
    """Apply DATATYPE pragma to modify allowed datatypes."""
    if not self.applies_to_interface(interface):
        return metadata
    
    # Create new datatype constraints based on pragma
    new_constraints = self._create_datatype_constraints()
    
    return InterfaceMetadata(
        name=metadata.name,
        interface_type=metadata.interface_type,
        allowed_datatypes=new_constraints,
        chunking_strategy=metadata.chunking_strategy
    )

def _create_datatype_constraints(self) -> List[DataTypeConstraint]:
    """Create DataTypeConstraint objects from pragma data."""
    # Extract from self.parsed_data and create constraints
    pass

def _interface_names_match(self, pragma_name: str, interface_name: str) -> bool:
    """Check if pragma interface name matches actual interface name."""
    # Implement flexible name matching logic
    pass
```

#### Step 2.2: Enhance BDimPragma
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Implement both methods in `BDimPragma`:
```python
def applies_to_interface(self, interface: Interface) -> bool:
    """Check if this BDIM pragma applies to the interface."""
    if not self.parsed_data:
        return False
    
    pragma_interface_name = self.parsed_data.get('interface_name')
    if not pragma_interface_name:
        return False
    
    return self._interface_names_match(pragma_interface_name, interface.name)

def apply_to_interface_metadata(self, interface: Interface, 
                              metadata: InterfaceMetadata) -> InterfaceMetadata:
    """Apply BDIM pragma to modify chunking strategy."""
    if not self.applies_to_interface(interface):
        return metadata
    
    # Create chunking strategy from pragma data
    new_strategy = self._create_chunking_strategy()
    
    return InterfaceMetadata(
        name=metadata.name,
        interface_type=metadata.interface_type,
        allowed_datatypes=metadata.allowed_datatypes,
        chunking_strategy=new_strategy
    )

def _create_chunking_strategy(self) -> ChunkingStrategy:
    """Create chunking strategy from pragma data."""
    # Process self.parsed_data to create appropriate strategy
    pass
```

#### Step 2.3: Enhance WeightPragma
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Implement both methods in `WeightPragma`:
```python
def applies_to_interface(self, interface: Interface) -> bool:
    """Check if this WEIGHT pragma applies to the interface."""
    if not self.parsed_data:
        return False
    
    interface_names = self.parsed_data.get('interface_names', [])
    for pragma_name in interface_names:
        if self._interface_names_match(pragma_name, interface.name):
            return True
    return False

def apply_to_interface_metadata(self, interface: Interface, 
                              metadata: InterfaceMetadata) -> InterfaceMetadata:
    """Apply WEIGHT pragma to mark interface as weight type."""
    if not self.applies_to_interface(interface):
        return metadata
    
    return InterfaceMetadata(
        name=metadata.name,
        interface_type=InterfaceType.WEIGHT,  # Override type
        allowed_datatypes=metadata.allowed_datatypes,
        chunking_strategy=metadata.chunking_strategy
    )
```

### Phase 3: Refactor PragmaHandler

#### Step 3.1: Simplify create_interface_metadata
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py`

Replace the current complex implementation with:
```python
def create_interface_metadata(self, interface: Interface, pragmas: List[Pragma]) -> InterfaceMetadata:
    """
    Create InterfaceMetadata using pragma chain-of-responsibility pattern.
    
    This method creates base metadata from the interface structure, then
    applies each relevant pragma in sequence to build the final metadata.
    
    Args:
        interface: Parsed Interface object from interface builder
        pragmas: All pragmas found in the RTL
        
    Returns:
        InterfaceMetadata: Complete metadata with all pragma effects applied
    """
    logger.debug(f"Creating InterfaceMetadata for interface: {interface.name}")
    
    # Start with base metadata from interface structure
    metadata = self._create_base_interface_metadata(interface)
    
    # Apply each relevant pragma in sequence
    for pragma in pragmas:
        try:
            if pragma.applies_to_interface(interface):
                logger.debug(f"Applying {pragma.type.value} pragma to {interface.name}")
                metadata = pragma.apply_to_interface_metadata(interface, metadata)
        except Exception as e:
            logger.warning(f"Failed to apply {pragma.type.value} pragma to {interface.name}: {e}")
    
    return metadata

def _create_base_interface_metadata(self, interface: Interface) -> InterfaceMetadata:
    """Create base InterfaceMetadata from interface structure."""
    # Extract base datatype constraints from interface ports
    allowed_datatypes = self._extract_base_datatype_constraints(interface)
    
    return InterfaceMetadata(
        name=interface.name,
        interface_type=interface.type,
        allowed_datatypes=allowed_datatypes,
        chunking_strategy=DefaultChunkingStrategy()
    )
```

#### Step 3.2: Remove Duplicated Methods
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py`

Remove these methods (logic moved to pragma classes):
- `_apply_datatype_pragma()`
- `_apply_chunking_pragma()`
- `_apply_weight_pragma()`
- `_pragma_applies_to_interface()`
- `_interface_names_match()`

Keep only:
- `_extract_base_datatype_constraints()` (base interface analysis)

### Phase 4: Add Shared Helper Methods

#### Step 4.1: Add Interface Name Matching Mixin
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Add before pragma classes:
```python
class InterfaceNameMatcher:
    """Mixin providing interface name matching utilities."""
    
    @staticmethod
    def _interface_names_match(pragma_name: str, interface_name: str) -> bool:
        """Check if pragma interface name matches actual interface name."""
        # Exact match
        if pragma_name == interface_name:
            return True
        
        # Prefix match (e.g., "in0" matches "in0_V_data_V")
        if interface_name.startswith(pragma_name):
            return True
        
        # Reverse prefix match (e.g., "in0_V_data_V" matches "in0")
        if pragma_name.startswith(interface_name):
            return True
        
        # Base name matching (remove common suffixes)
        pragma_base = pragma_name.replace('_V_data_V', '').replace('_data', '')
        interface_base = interface_name.replace('_V_data_V', '').replace('_data', '')
        
        return pragma_base == interface_base
```

#### Step 4.2: Update Pragma Classes to Use Mixin
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Update class definitions:
```python
class DatatypePragma(Pragma, InterfaceNameMatcher):
    # ... existing code

class BDimPragma(Pragma, InterfaceNameMatcher):
    # ... existing code

class WeightPragma(Pragma, InterfaceNameMatcher):
    # ... existing code
```

### Phase 5: Testing and Validation

#### Step 5.1: Create Unit Tests
**File**: `tests/tools/hw_kernel_gen/rtl_parser/test_pragma_refactor.py`

Create comprehensive tests for:
- Each pragma's `applies_to_interface()` method
- Each pragma's `apply_to_interface_metadata()` method
- Pragma chain application
- Edge cases and error handling

#### Step 5.2: Integration Testing
**File**: `test_pragma_system_integration.py`

Test complete pragma system with:
- Multiple pragmas affecting same interface
- Pragmas with different priorities
- Real SystemVerilog examples
- Performance comparison

### Phase 6: Clean Up Legacy Code

#### Step 6.1: Remove Old Apply Methods
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`

Update existing `apply()` methods to use new interface metadata methods:
```python
def apply(self, **kwargs) -> Any:
    """Legacy apply method for backward compatibility."""
    interfaces = kwargs.get('interfaces')
    if interfaces:
        for interface in interfaces.values():
            if self.applies_to_interface(interface):
                # Apply to interface.metadata for compatibility
                # This ensures existing tests still pass
                pass
```

#### Step 6.2: Update RTL Parser
**File**: `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`

Ensure `_apply_pragmas()` method works with new architecture.

## Benefits of This Refactor

### 1. **Elegant Object-Oriented Design**
- Each pragma class is responsible for its own logic
- Clean separation of concerns
- Easy to extend with new pragma types

### 2. **Elimination of Code Duplication**
- Single source of truth for pragma logic
- Consistent behavior across the system
- Easier maintenance

### 3. **Improved Testability**
- Each pragma can be tested independently
- Clear interfaces for testing
- Easy to mock and verify behavior

### 4. **Better Error Handling**
- Isolated error handling per pragma
- Graceful degradation when pragmas fail
- Better debugging information

### 5. **Composable Architecture**
- Pragmas can build on each other's effects
- Order-dependent processing when needed
- Easy to add pragma combinations

## Risk Mitigation

### 1. **Backward Compatibility**
- Keep existing `apply()` methods during transition
- Gradual migration of dependent code
- Comprehensive testing of edge cases

### 2. **Performance Considerations**
- Minimal overhead from new architecture
- Efficient pragma filtering
- Lazy evaluation where possible

### 3. **Error Recovery**
- Each pragma application is isolated
- Failures don't break entire pipeline
- Clear error reporting and logging

## Success Criteria

1. **All existing tests pass** with new architecture
2. **No performance regression** in pragma processing
3. **Simplified PragmaHandler** with reduced complexity
4. **Clean pragma class implementations** following OOP principles
5. **Comprehensive test coverage** for new methods
6. **Documentation updated** to reflect new architecture

## Timeline Estimate

- **Phase 1**: 2-3 hours (base class enhancement)
- **Phase 2**: 4-6 hours (pragma-specific implementations)
- **Phase 3**: 2-3 hours (PragmaHandler refactor)
- **Phase 4**: 1-2 hours (shared utilities)
- **Phase 5**: 3-4 hours (testing)
- **Phase 6**: 1-2 hours (cleanup)

**Total**: 13-20 hours over 3-5 development sessions