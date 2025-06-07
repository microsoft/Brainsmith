"""
Unit tests for enhanced AutoHWCustomOp with per-interface chunking strategies.

Tests the strategy-based chunking architecture where each interface
has its own chunking strategy instead of a global override system.
"""

import pytest
from unittest.mock import Mock, MagicMock
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import (
    InterfaceMetadata, InterfaceMetadataCollection, DataTypeConstraint, DataflowInterfaceType
)
from brainsmith.dataflow.core.chunking_strategy import (
    default_chunking, index_chunking, last_dim_chunking, DefaultChunkingStrategy, IndexBasedChunkingStrategy
)


class TestEnhancedAutoHWCustomOp:
    """Test enhanced AutoHWCustomOp with per-interface chunking strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock ONNX node
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_tensor", "weight_tensor"]
        self.mock_onnx_node.output = ["output_tensor"]
        
        # Create test interface metadata with default strategies
        self.input_metadata = InterfaceMetadata(
            name="in0_V_data_V",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8),
                DataTypeConstraint(finn_type="INT8", bit_width=8, signed=True)
            ],
            chunking_strategy=default_chunking()
        )
        
        self.output_metadata = InterfaceMetadata(
            name="out_V_data_V",
            interface_type=DataflowInterfaceType.OUTPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8)
            ],
            chunking_strategy=default_chunking()
        )
        
        self.interface_metadata_list = [self.input_metadata, self.output_metadata]
    
    def test_two_phase_initialization_new_approach(self):
        """Test two-phase initialization with interface metadata and strategies."""
        # Create AutoHWCustomOp with interface metadata (new approach)
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=self.interface_metadata_list
        )
        
        # Model should not be built initially
        assert not op._model_built
        assert op._dataflow_model is None
        
        # Interface metadata should be accessible
        assert len(op.interface_metadata.interfaces) == 2
        assert op.interface_metadata.get_by_name("in0_V_data_V") == self.input_metadata
        assert op.interface_metadata.get_by_name("out_V_data_V") == self.output_metadata
    
    def test_interface_metadata_property_access(self):
        """Test accessing interface metadata through property."""
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=self.interface_metadata_list
        )
        
        # Test interface metadata access
        metadata = op.interface_metadata
        assert isinstance(metadata, InterfaceMetadataCollection)
        assert len(metadata.interfaces) == 2
        assert metadata.get_by_name("in0_V_data_V") == self.input_metadata
    
    def test_lazy_dataflow_model_building(self):
        """Test that DataflowModel is built lazily on first access."""
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=self.interface_metadata_list
        )
        
        # Initially not built
        assert not op._model_built
        
        # Mock the build method to avoid actual tensor chunking
        mock_model = Mock()
        mock_model.input_interfaces = []
        mock_model.output_interfaces = []
        mock_model.weight_interfaces = []
        mock_model.config_interfaces = []
        
        op._build_dataflow_model = Mock()
        op._dataflow_model = mock_model
        op._model_built = True
        
        # Access dataflow_model property - should return the model
        model = op.dataflow_model
        assert model == mock_model
    
    def test_interface_metadata_collection_access(self):
        """Test access to interface metadata collection."""
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=self.interface_metadata_list
        )
        
        collection = op._interface_metadata_collection
        
        # Test collection properties
        assert len(collection.interfaces) == 2
        assert collection.get_by_name("in0_V_data_V") == self.input_metadata
        assert collection.get_by_name("out_V_data_V") == self.output_metadata
        
        # Test type-based access
        input_interfaces = collection.get_input_interfaces()
        output_interfaces = collection.get_output_interfaces()
        
        assert len(input_interfaces) == 1
        assert len(output_interfaces) == 1
        assert input_interfaces[0] == self.input_metadata
        assert output_interfaces[0] == self.output_metadata
    
    def test_per_interface_chunking_strategies(self):
        """Test per-interface chunking strategies (replaces override system)."""
        # Create interfaces with different chunking strategies
        default_interface = InterfaceMetadata(
            name="default_input",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=default_chunking()
        )
        
        custom_interface = InterfaceMetadata(
            name="custom_input",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=index_chunking(-1, [16])
        )
        
        convenience_interface = InterfaceMetadata(
            name="conv_input",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=last_dim_chunking(32)
        )
        
        # Test with AutoHWCustomOp
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=[default_interface, custom_interface, convenience_interface]
        )
        
        # Verify each interface has its own strategy
        interfaces = op.interface_metadata.interfaces
        assert isinstance(interfaces[0].chunking_strategy, DefaultChunkingStrategy)
        assert isinstance(interfaces[1].chunking_strategy, IndexBasedChunkingStrategy)
        assert isinstance(interfaces[2].chunking_strategy, IndexBasedChunkingStrategy)
        
        # Verify strategies have correct parameters
        assert interfaces[1].chunking_strategy.start_index == -1
        assert interfaces[1].chunking_strategy.shape == [16]
        assert interfaces[2].chunking_strategy.start_index == -1
        assert interfaces[2].chunking_strategy.shape == [32]
    
    def test_datatype_constraint_validation(self):
        """Test datatype constraint validation."""
        constraint = DataTypeConstraint(
            finn_type="UINT8",
            bit_width=8,
            signed=False
        )
        
        # Test exact match validation
        assert constraint.validates("UINT8")
        assert not constraint.validates("INT8")
        assert not constraint.validates("UINT16")
    
    def test_interface_metadata_datatype_validation(self):
        """Test interface metadata datatype validation."""
        # Test valid datatype
        assert self.input_metadata.validates_datatype("UINT8")
        assert self.input_metadata.validates_datatype("INT8")
        
        # Test invalid datatype
        assert not self.input_metadata.validates_datatype("UINT16")
        assert not self.input_metadata.validates_datatype("FLOAT32")
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata is properly validated."""
        # Test duplicate interface names should raise error
        duplicate_metadata = [
            self.input_metadata,
            self.input_metadata  # Duplicate
        ]
        
        with pytest.raises(ValueError, match="Duplicate interface names"):
            AutoHWCustomOp(
                onnx_node=self.mock_onnx_node,
                interface_metadata=duplicate_metadata
            )
    
    def test_model_invalidation(self):
        """Test model invalidation functionality."""
        op = AutoHWCustomOp(
            onnx_node=self.mock_onnx_node,
            interface_metadata=self.interface_metadata_list
        )
        
        # Mock build process
        op._build_dataflow_model = Mock()
        mock_model = Mock()
        op._dataflow_model = mock_model
        op._model_built = True
        
        # Invalidate model
        op._invalidate_dataflow_model()
        assert not op._model_built
        assert op._dataflow_model is None
        
        # Access again should trigger rebuild
        op._build_dataflow_model = Mock()
        op._ensure_dataflow_model_built()
        assert op._build_dataflow_model.called
    
    def test_required_interface_metadata(self):
        """Test that interface metadata is required."""
        # Should raise error if no interface metadata provided
        with pytest.raises(TypeError):
            AutoHWCustomOp(onnx_node=self.mock_onnx_node)  # Missing required parameter


class TestDataTypeConstraint:
    """Test DataTypeConstraint functionality."""
    
    def test_constraint_creation(self):
        """Test creating datatype constraints."""
        constraint = DataTypeConstraint(
            finn_type="UINT8",
            bit_width=8,
            signed=False
        )
        
        assert constraint.finn_type == "UINT8"
        assert constraint.bit_width == 8
        assert not constraint.signed
    
    def test_constraint_validation_errors(self):
        """Test constraint validation errors."""
        with pytest.raises(ValueError, match="Bit width must be positive"):
            DataTypeConstraint(finn_type="UINT8", bit_width=0)
    
    def test_from_dict_creation(self):
        """Test creating constraint from dictionary."""
        constraint_dict = {
            "finn_type": "INT16",
            "bit_width": 16,
            "signed": True
        }
        
        constraint = DataTypeConstraint.from_dict(constraint_dict)
        assert constraint.finn_type == "INT16"
        assert constraint.bit_width == 16
        assert constraint.signed


class TestInterfaceMetadata:
    """Test InterfaceMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating interface metadata with chunking strategy."""
        metadata = InterfaceMetadata(
            name="test_interface",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=index_chunking(-1, [16])
        )
        
        assert metadata.name == "test_interface"
        assert metadata.interface_type == DataflowInterfaceType.INPUT
        assert len(metadata.allowed_datatypes) == 1
        assert isinstance(metadata.chunking_strategy, IndexBasedChunkingStrategy)
    
    def test_metadata_validation_errors(self):
        """Test metadata validation errors."""
        with pytest.raises(ValueError, match="Interface name cannot be empty"):
            InterfaceMetadata(
                name="",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)]
            )
    
    def test_default_datatype(self):
        """Test getting default datatype."""
        metadata = InterfaceMetadata(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            allowed_datatypes=[
                DataTypeConstraint(finn_type="UINT8", bit_width=8),
                DataTypeConstraint(finn_type="UINT16", bit_width=16)
            ]
        )
        
        default_dtype = metadata.get_default_datatype()
        assert default_dtype.finn_type == "UINT8"  # First one is default


class TestChunkingStrategies:
    """Test chunking strategy functionality."""
    
    def test_default_strategy(self):
        """Test default chunking strategy."""
        strategy = default_chunking()
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # Default strategy should preserve tensor shape
        assert qDim == [1, 1, 1, 1]
        assert tDim == [1, 8, 32, 32]
    
    def test_index_based_strategy(self):
        """Test index-based chunking strategy."""
        strategy = index_chunking(-1, [16])
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # Should chunk last dimension
        assert qDim[3] == 2  # 32 // 16
        assert tDim[3] == 16
    
    def test_convenience_functions(self):
        """Test convenience chunking functions."""
        # Test last_dim_chunking
        strategy = last_dim_chunking(8)
        assert strategy.start_index == -1
        assert strategy.shape == [8]
        
        # Test spatial_chunking  
        from brainsmith.dataflow.core.chunking_strategy import spatial_chunking
        strategy = spatial_chunking(16, 16)
        assert strategy.start_index == 2
        assert strategy.shape == [16, 16]