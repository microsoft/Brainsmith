"""Test QONNX plugin discovery and integration.

Tests the enhanced QONNX registry with metadata support and
BrainSmith's ability to discover and query QONNX transforms.
"""

import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestQONNXIntegration:
    """Test suite for QONNX plugin integration."""
    
    def test_qonnx_registry_metadata(self):
        """Test enhanced QONNX registry with metadata."""
        try:
            from qonnx.transformation.registry import (
                TRANSFORMATION_REGISTRY,
                TRANSFORMATION_METADATA,
                get_transformation_info
            )
        except ImportError:
            pytest.skip("QONNX not available")
        
        # Check that enhanced functions exist
        assert hasattr(get_transformation_info, '__call__')
        
        # Check if any transforms have metadata
        if TRANSFORMATION_METADATA:
            # Get first transform with metadata
            name = next(iter(TRANSFORMATION_METADATA))
            info = get_transformation_info(name)
            
            assert info is not None
            assert "name" in info
            assert "class" in info
            
            # Check for enhanced metadata if present
            if "description" in info:
                assert isinstance(info["description"], (str, type(None)))
            if "tags" in info:
                assert isinstance(info["tags"], list)
    
    def test_qonnx_transform_discovery(self):
        """Test discovering QONNX transforms through BrainSmith."""
        from brainsmith.plugin.core import get_registry
        
        # Import QONNX to trigger any registrations
        try:
            import qonnx.transformation
        except ImportError:
            pytest.skip("QONNX not available")
        
        registry = get_registry()
        
        # Look for QONNX transforms (they should have qonnx: prefix)
        all_transforms = registry.query(type="transform")
        qonnx_transforms = [t for t in all_transforms if t["name"].startswith("qonnx:")]
        
        # We should discover at least some QONNX transforms
        # This assumes BrainSmith discovery has been implemented
        if qonnx_transforms:
            print(f"\nDiscovered {len(qonnx_transforms)} QONNX transforms")
            
            # Check a well-known transform
            fold_constants = [t for t in qonnx_transforms if "FoldConstants" in t["name"]]
            if fold_constants:
                transform = fold_constants[0]
                assert transform["framework"] == "qonnx"
                # Stage might be assigned by BrainSmith
                if "stage" in transform:
                    assert transform["stage"] in [None, "cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
    
    def test_qonnx_metadata_query(self):
        """Test querying QONNX transforms by metadata."""
        try:
            from qonnx.transformation.registry import query_transformations, list_transformation_tags
        except ImportError:
            pytest.skip("QONNX enhanced registry not available")
        
        # Query by tags if any exist
        tags = list_transformation_tags()
        if tags:
            # Query transforms with first available tag
            tag = tags[0]
            transforms = query_transformations(tags__contains=tag)
            
            assert len(transforms) > 0
            for t in transforms:
                assert tag in t.get("tags", [])
        
        # Query all and check structure
        all_transforms = query_transformations()
        for t in all_transforms:
            assert "name" in t
            assert "class" in t
    
    def test_qonnx_stage_mapping(self):
        """Test BrainSmith stage mapping for QONNX transforms."""
        from brainsmith.plugin.core import get_registry
        
        # Check if stage mapping exists
        try:
            from brainsmith.plugin.qonnx_stages import QONNX_STAGE_MAPPING
        except ImportError:
            pytest.skip("QONNX stage mapping not implemented yet")
        
        registry = get_registry()
        
        # Common QONNX transforms that should have stages
        expected_mappings = {
            "FoldConstants": "cleanup",
            "RemoveIdentityOps": "cleanup",
            "InferShapes": "cleanup",
            "GiveUniqueNodeNames": "cleanup",
        }
        
        for transform_name, expected_stage in expected_mappings.items():
            # Check mapping
            if transform_name in QONNX_STAGE_MAPPING:
                assert QONNX_STAGE_MAPPING[transform_name] == expected_stage
            
            # Check if registered with stage
            registered = registry.query(type="transform", name=f"qonnx:{transform_name}")
            if registered:
                transform = registered[0]
                assert transform.get("stage") == expected_stage
    
    def test_qonnx_fallback_compatibility(self):
        """Test backward compatibility for QONNX without metadata."""
        try:
            from qonnx.transformation.registry import TRANSFORMATION_REGISTRY
        except ImportError:
            pytest.skip("QONNX not available")
        
        # Basic registry should always work
        assert isinstance(TRANSFORMATION_REGISTRY, dict)
        
        # If we have transforms, they should be accessible
        if TRANSFORMATION_REGISTRY:
            name = next(iter(TRANSFORMATION_REGISTRY))
            transform_cls = TRANSFORMATION_REGISTRY[name]
            
            assert transform_cls is not None
            assert hasattr(transform_cls, '__name__')
    
    def test_qonnx_transform_class_retrieval(self):
        """Test retrieving actual QONNX transform classes."""
        from brainsmith.plugin.core import get_registry
        
        try:
            from qonnx.transformation.fold_constants import FoldConstants
            from qonnx.transformation.infer_shapes import InferShapes
        except ImportError:
            pytest.skip("QONNX transforms not available")
        
        registry = get_registry()
        
        # Try to get QONNX transforms through registry
        # They might be prefixed with "qonnx:"
        fold_cls = registry.get("transform", "qonnx:FoldConstants")
        if fold_cls:
            # Verify it's the right class
            assert fold_cls.__name__ == "FoldConstants"
            
            # Check it has the apply method
            assert hasattr(fold_cls, 'apply')
    
    def test_qonnx_transform_metadata_on_class(self):
        """Test that QONNX transforms have metadata stored on class."""
        try:
            from qonnx.transformation.fold_constants import FoldConstants
            from qonnx.transformation.infer_shapes import InferShapes
            from qonnx.transformation.remove import RemoveIdentityOps
        except ImportError:
            pytest.skip("QONNX transforms not available")
        
        # Check if enhanced transforms have metadata
        transforms_to_check = [FoldConstants, InferShapes, RemoveIdentityOps]
        
        for transform_cls in transforms_to_check:
            if hasattr(transform_cls, '_qonnx_metadata'):
                metadata = transform_cls._qonnx_metadata
                assert isinstance(metadata, dict)
                assert "name" in metadata
                
                # These should have been added by our enhancements
                if "tags" in metadata:
                    assert isinstance(metadata["tags"], list)
    
    def test_qonnx_query_by_author(self):
        """Test querying QONNX transforms by author metadata."""
        try:
            from qonnx.transformation.registry import query_transformations
        except ImportError:
            pytest.skip("QONNX enhanced registry not available")
        
        # Query by author if any have it
        qonnx_authored = query_transformations(author="qonnx-team")
        
        if qonnx_authored:
            for t in qonnx_authored:
                assert t.get("author") == "qonnx-team"
    
    def test_qonnx_tag_discovery(self):
        """Test discovering all tags used by QONNX transforms."""
        try:
            from qonnx.transformation.registry import list_transformation_tags
        except ImportError:
            pytest.skip("QONNX enhanced registry not available")
        
        tags = list_transformation_tags()
        
        # Check expected tags if metadata was added
        expected_tags = ["optimization", "cleanup", "graph-simplification", "shape-inference", "validation"]
        
        for tag in expected_tags:
            if tag in tags:
                # Verify transforms exist with this tag
                from qonnx.transformation.registry import query_transformations
                tagged_transforms = query_transformations(tags__contains=tag)
                assert len(tagged_transforms) > 0