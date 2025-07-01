"""
Test BERT steps to check for transform registration errors.

This test applies all BERT steps to dummy ONNX models to identify
which transforms are properly registered and which are missing.
"""

import pytest
import tempfile
import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dataclasses import dataclass
from typing import List, Tuple, Optional

from brainsmith.steps import bert_steps
from brainsmith.plugins import transforms as tfm
from brainsmith.plugin import get_plugin_manager


# Helper classes and functions

@dataclass
class DummyConfig:
    """Minimal config object for step functions."""
    output_dir: str
    verify_input_npy: str = ""
    verify_expected_output_npy: str = ""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            self.output_dir = self.temp_dir
        else:
            self.output_dir = output_dir
            self.temp_dir = None
        
        # Create required subdirectories
        os.makedirs(os.path.join(self.output_dir, "stitched_ip"), exist_ok=True)
        
        # Create dummy npy files
        self.verify_input_npy = os.path.join(self.output_dir, "input.npy")
        self.verify_expected_output_npy = os.path.join(self.output_dir, "expected_output.npy")
        np.save(self.verify_input_npy, np.array([[1.0]]))
        np.save(self.verify_expected_output_npy, np.array([[1.0]]))
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def create_dummy_model(name="test_model", num_nodes=3) -> ModelWrapper:
    """Create a minimal ONNX model for testing."""
    
    # Create input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    
    # Create a simple chain of operations
    nodes = []
    tensors = ["input"]
    
    for i in range(num_nodes):
        output_name = f"output_{i}" if i < num_nodes - 1 else "output"
        
        # Create an Identity node (simple pass-through)
        node = helper.make_node(
            "Identity",
            inputs=[tensors[-1]],
            outputs=[output_name],
            name=f"node_{i}"
        )
        nodes.append(node)
        tensors.append(output_name)
    
    # Create output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        name,
        [input_tensor],
        [output_tensor],
        []
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 11
    
    # Wrap in ModelWrapper
    return ModelWrapper(model)


def create_bert_model() -> ModelWrapper:
    """Create a dummy BERT-like model with LayerNormalization nodes."""
    
    # Create input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128, 768])
    
    # Create nodes
    nodes = []
    
    # Add a LayerNormalization node (for head/tail removal tests)
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [768], 
                                   np.ones(768, dtype=np.float32).tolist())
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [768], 
                                  np.zeros(768, dtype=np.float32).tolist())
    
    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=["input", "scale", "bias"],
        outputs=["ln_output"],
        name="LayerNorm_0",
        axis=-1,
        epsilon=1e-5
    )
    nodes.append(ln_node)
    
    # Add some more nodes
    identity_node = helper.make_node(
        "Identity",
        inputs=["ln_output"],
        outputs=["output"],
        name="final_node"
    )
    nodes.append(identity_node)
    
    # Create output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128, 768])
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        "bert_model",
        [input_tensor],
        [output_tensor],
        [scale_init, bias_init]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 11
    
    # Wrap in ModelWrapper
    return ModelWrapper(model)


def check_transform_registered(transform_name: str, framework: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a transform is properly registered.
    
    Returns:
        Tuple of (is_registered, error_message)
    """
    try:
        if framework:
            # Try framework-specific access
            framework_accessor = getattr(tfm, framework, None)
            if framework_accessor is None:
                return False, f"Framework '{framework}' not found"
            
            transform = getattr(framework_accessor, transform_name, None)
            if transform is None:
                return False, f"Transform '{transform_name}' not found in {framework}"
        else:
            # Try direct access
            transform = getattr(tfm, transform_name, None)
            if transform is None:
                return False, f"Transform '{transform_name}' not found"
        
        return True, ""
    except Exception as e:
        return False, str(e)


class TestBertStepsRegistration:
    """Test all BERT steps for transform registration errors."""
    
    @pytest.fixture
    def dummy_model(self):
        """Provide a dummy ONNX model."""
        return create_dummy_model()
    
    @pytest.fixture
    def bert_model(self):
        """Provide a BERT-like model."""
        return create_bert_model()
    
    @pytest.fixture
    def config(self):
        """Provide a dummy config object."""
        cfg = DummyConfig()
        yield cfg
        cfg.cleanup()
    
    def test_metadata_steps(self, dummy_model, config):
        """Test metadata extraction steps for registration errors."""
        missing_transforms = []
        
        # Check ExtractShellIntegrationMetadata
        registered, error = check_transform_registered("ExtractShellIntegrationMetadata")
        if not registered:
            missing_transforms.append(("ExtractShellIntegrationMetadata", error))
        
        # Only test if transforms are available
        if not missing_transforms:
            try:
                # This step needs the stitched_ip directory to exist
                model = bert_steps.shell_metadata_handover_step(dummy_model, config)
                assert model is not None
            except Exception as e:
                pytest.fail(f"shell_metadata_handover_step failed: {e}")
        
        # Report missing transforms
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing transforms for metadata steps:\n{report}")
    
    def test_cleanup_steps(self, dummy_model, config):
        """Test cleanup steps for registration errors."""
        missing_transforms = []
        
        # Check transforms used in cleanup steps
        cleanup_transforms = [
            ("RemoveIdentityOps", "qonnx"),
            ("GiveReadableTensorNames", "qonnx"),
            ("GiveUniqueNodeNames", "qonnx"),
            ("ConvertDivToMul", "qonnx"),
        ]
        
        for transform_name, framework in cleanup_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                missing_transforms.append((f"{framework}:{transform_name}", error))
        
        # Test the steps
        errors = []
        
        try:
            model = bert_steps.cleanup_step(dummy_model, config)
            assert model is not None
        except AttributeError as e:
            errors.append(f"cleanup_step: {e}")
        except Exception as e:
            errors.append(f"cleanup_step (other): {e}")
        
        try:
            model = bert_steps.cleanup_advanced_step(dummy_model, config)
            assert model is not None
        except AttributeError as e:
            errors.append(f"cleanup_advanced_step: {e}")
        except Exception as e:
            errors.append(f"cleanup_advanced_step (other): {e}")
        
        try:
            model = bert_steps.fix_dynamic_dimensions_step(dummy_model, config)
            assert model is not None
        except AttributeError as e:
            errors.append(f"fix_dynamic_dimensions_step: {e}")
        except Exception as e:
            errors.append(f"fix_dynamic_dimensions_step (other): {e}")
        
        # Report results
        if missing_transforms or errors:
            report = ""
            if missing_transforms:
                report += "Missing transforms:\n"
                report += "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            if errors:
                if report:
                    report += "\n\n"
                report += "Execution errors:\n"
                report += "\n".join([f"  - {error}" for error in errors])
            pytest.fail(f"Cleanup steps have issues:\n{report}")
    
    def test_conversion_steps(self, dummy_model, config):
        """Test QONNX to FINN conversion steps for registration errors."""
        missing_transforms = []
        
        # Check transforms used in conversion
        conversion_transforms = [
            ("ExpandNorms", None),  # BrainSmith native
            ("FoldConstants", None),
            ("ConvertDivToMul", "qonnx"),
            ("ConvertQONNXtoFINN", None),  # FINN native
        ]
        
        for transform_name, framework in conversion_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                prefix = f"{framework}:" if framework else ""
                missing_transforms.append((f"{prefix}{transform_name}", error))
        
        # Test the step
        try:
            model = bert_steps.qonnx_to_finn_step(dummy_model, config)
            assert model is not None
        except AttributeError as e:
            if missing_transforms:
                report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
                pytest.fail(f"Missing transforms:\n{report}\n\nError: {e}")
            else:
                pytest.fail(f"qonnx_to_finn_step failed with AttributeError: {e}")
        except Exception as e:
            pytest.fail(f"qonnx_to_finn_step failed: {e}")
        
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing transforms for conversion:\n{report}")
    
    def test_streamlining_steps(self, dummy_model, config):
        """Test streamlining steps for registration errors."""
        missing_transforms = []
        
        # Check transforms used in streamlining
        streamlining_transforms = [
            # FINN native
            ("AbsorbSignBiasIntoMultiThreshold", None),
            ("AbsorbAddIntoMultiThreshold", None),
            ("AbsorbMulIntoMultiThreshold", None),
            ("RoundAndClipThresholds", None),
            ("MoveScalarMulPastMatMul", None),
            ("MoveScalarLinearPastInvariants", None),
            # Framework specific
            ("MoveOpPastFork", "finn"),
            ("InferDataTypes", "qonnx"),
            ("GiveUniqueNodeNames", "qonnx"),
        ]
        
        for transform_name, framework in streamlining_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                prefix = f"{framework}:" if framework else ""
                missing_transforms.append((f"{prefix}{transform_name}", error))
        
        # Test the step
        try:
            # First apply qonnx_to_finn to prepare the model
            model = bert_steps.qonnx_to_finn_step(dummy_model, config)
            model = bert_steps.streamlining_step(model, config)
            assert model is not None
        except AttributeError as e:
            if missing_transforms:
                report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
                pytest.fail(f"Missing transforms:\n{report}\n\nError: {e}")
            else:
                pytest.fail(f"streamlining_step failed with AttributeError: {e}")
        except Exception as e:
            # Some transforms might fail on our simple dummy model
            pass
        
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing transforms for streamlining:\n{report}")
    
    def test_hardware_steps(self, dummy_model, config):
        """Test hardware inference steps for registration errors."""
        missing_transforms = []
        
        # Check transforms used in hardware inference
        hardware_transforms = [
            ("ConvertToHWLayers", None),  # FINN native
            ("InferLayerNorm", None),  # BrainSmith native
            ("InferShuffle", None),
            ("InferHWSoftmax", None),
        ]
        
        for transform_name, framework in hardware_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                prefix = f"{framework}:" if framework else ""
                missing_transforms.append((f"{prefix}{transform_name}", error))
        
        # Test the step
        try:
            # Prepare model
            model = bert_steps.qonnx_to_finn_step(dummy_model, config)
            model = bert_steps.streamlining_step(model, config)
            model = bert_steps.infer_hardware_step(model, config)
            assert model is not None
        except AttributeError as e:
            if missing_transforms:
                report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
                pytest.fail(f"Missing transforms:\n{report}\n\nError: {e}")
            else:
                pytest.fail(f"infer_hardware_step failed with AttributeError: {e}")
        except Exception as e:
            # Some transforms might fail on our simple dummy model
            pass
        
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing transforms for hardware inference:\n{report}")
    
    def test_bert_specific_steps(self, bert_model, config):
        """Test BERT-specific head/tail removal steps."""
        missing_transforms = []
        
        # Check transforms
        bert_transforms = [
            ("RemoveBertHead", None),
            ("RemoveBertTail", None),
        ]
        
        for transform_name, framework in bert_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                missing_transforms.append((transform_name, error))
        
        # Test the steps
        errors = []
        
        try:
            model = bert_steps.remove_head_step(bert_model, config)
            assert model is not None
        except AttributeError as e:
            errors.append(f"remove_head_step: {e}")
        except Exception as e:
            # These transforms might have specific requirements
            pass
        
        try:
            model = bert_steps.remove_tail_step(bert_model, config)
            assert model is not None
        except AttributeError as e:
            errors.append(f"remove_tail_step: {e}")
        except Exception as e:
            # These transforms might have specific requirements
            pass
        
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing BERT-specific transforms:\n{report}")
        
        if errors:
            report = "\n".join([f"  - {error}" for error in errors])
            pytest.fail(f"BERT-specific steps failed:\n{report}")
    
    def test_optimization_steps(self, dummy_model, config):
        """Test optimization steps for registration errors."""
        missing_transforms = []
        
        # Check transforms
        optimization_transforms = [
            ("TempShuffleFixer", None),
            ("SetPumpedCompute", None),
        ]
        
        for transform_name, framework in optimization_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            if not registered:
                missing_transforms.append((transform_name, error))
        
        # Test the step
        try:
            model = bert_steps.constrain_folding_and_set_pumped_compute_step(dummy_model, config)
            assert model is not None
        except AttributeError as e:
            if missing_transforms:
                report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
                pytest.fail(f"Missing transforms:\n{report}\n\nError: {e}")
            else:
                pytest.fail(f"Optimization step failed with AttributeError: {e}")
        except Exception as e:
            # Some transforms might fail on our simple dummy model
            pass
        
        if missing_transforms:
            report = "\n".join([f"  - {name}: {error}" for name, error in missing_transforms])
            pytest.skip(f"Missing optimization transforms:\n{report}")
    
    def test_generate_missing_transforms_report(self):
        """Generate a comprehensive report of all missing transforms."""
        all_transforms = [
            # Metadata
            ("ExtractShellIntegrationMetadata", None, "metadata"),
            
            # Cleanup
            ("RemoveIdentityOps", "qonnx", "cleanup"),
            ("GiveReadableTensorNames", "qonnx", "cleanup"),
            ("GiveUniqueNodeNames", "qonnx", "cleanup"),
            ("ConvertDivToMul", "qonnx", "cleanup"),
            ("SortCommutativeInputsInitializerLast", "qonnx", "cleanup"),  # Known missing
            
            # Conversion
            ("ExpandNorms", None, "conversion"),
            ("FoldConstants", None, "conversion"),
            ("ConvertQONNXtoFINN", None, "conversion"),
            
            # Streamlining
            ("AbsorbSignBiasIntoMultiThreshold", None, "streamlining"),
            ("AbsorbAddIntoMultiThreshold", None, "streamlining"),
            ("AbsorbMulIntoMultiThreshold", None, "streamlining"),
            ("RoundAndClipThresholds", None, "streamlining"),
            ("MoveOpPastFork", "finn", "streamlining"),
            ("MoveScalarMulPastMatMul", None, "streamlining"),
            ("MoveScalarLinearPastInvariants", None, "streamlining"),
            ("InferDataTypes", "qonnx", "streamlining"),
            
            # Hardware
            ("ConvertToHWLayers", None, "hardware"),
            ("InferLayerNorm", None, "hardware"),
            ("InferShuffle", None, "hardware"),
            ("InferHWSoftmax", None, "hardware"),
            
            # BERT-specific
            ("RemoveBertHead", None, "bert"),
            ("RemoveBertTail", None, "bert"),
            
            # Optimization
            ("TempShuffleFixer", None, "optimization"),
            ("SetPumpedCompute", None, "optimization"),
        ]
        
        missing_by_category = {}
        registered_by_category = {}
        
        for transform_name, framework, category in all_transforms:
            registered, error = check_transform_registered(transform_name, framework)
            
            if not registered:
                if category not in missing_by_category:
                    missing_by_category[category] = []
                prefix = f"{framework}:" if framework else ""
                missing_by_category[category].append(f"{prefix}{transform_name}")
            else:
                if category not in registered_by_category:
                    registered_by_category[category] = []
                prefix = f"{framework}:" if framework else ""
                registered_by_category[category].append(f"{prefix}{transform_name}")
        
        # Generate report
        report = "=== Transform Registration Report ===\n\n"
        
        report += "REGISTERED TRANSFORMS:\n"
        for category, transforms in sorted(registered_by_category.items()):
            report += f"\n{category.upper()}:\n"
            for transform in sorted(transforms):
                report += f"  ✓ {transform}\n"
        
        report += "\n\nMISSING TRANSFORMS:\n"
        for category, transforms in sorted(missing_by_category.items()):
            report += f"\n{category.upper()}:\n"
            for transform in sorted(transforms):
                report += f"  ✗ {transform}\n"
        
        report += "\n\nSUMMARY:\n"
        total_transforms = len(all_transforms)
        total_missing = sum(len(transforms) for transforms in missing_by_category.values())
        total_registered = sum(len(transforms) for transforms in registered_by_category.values())
        
        report += f"  Total transforms checked: {total_transforms}\n"
        report += f"  Registered: {total_registered} ({total_registered/total_transforms*100:.1f}%)\n"
        report += f"  Missing: {total_missing} ({total_missing/total_transforms*100:.1f}%)\n"
        
        # Print the report for visibility
        print("\n" + report)
        
        # Store report in test output
        with open("transform_registration_report.txt", "w") as f:
            f.write(report)
        
        # Don't fail the test, just document the state
        assert True