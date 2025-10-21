"""Parity tests for computational kernels (MVAU, VVAU, etc.)

This mixin provides tests for kernels that perform computation with
weights, thresholds, and accumulators.

**Applicable to**:
- MatrixVectorActivation (MVAU) - Matrix-vector multiplication with activation
- VectorVectorActivation (VVAU) - Depthwise convolution with activation
- Future computational kernels with weights and accumulation

**NOT applicable to**:
- Thresholding - Has thresholds but no weights (subset of tests apply)
- Data movement ops (Shuffle, FIFO, etc.) - No computation

**Usage**:
```python
from tests.parity.base_parity_test import ParityTestBase
from tests.parity.computational_parity_test import ComputationalParityMixin

class TestVVAUParity(ParityTestBase, ComputationalParityMixin):
    '''VVAU gets all 32 tests (25 base + 7 computational)'''
    pass
```

**Test Coverage**:
- Memory sizing (calc_wmem, calc_tmem)
- Accumulator datatype inference
- Hardware tensor transformations (weights, thresholds)
- File generation (weight files, parameter files)

Total: 7 additional tests beyond ParityTestBase
"""

import numpy as np
import pytest
import tempfile
import os


class ComputationalParityMixin:
    """Parity tests for computational kernels with weights and thresholds.

    This mixin adds 7 tests for computational operations:
    1. test_calc_wmem_parity - Weight memory depth
    2. test_calc_tmem_parity - Threshold memory depth
    3. test_accumulator_datatype_parity - Accumulator bit-width
    4. test_hw_compatible_weight_tensor_parity - Weight tensor layout
    5. test_hw_compatible_threshold_tensor_parity - Threshold tensor layout
    6. test_make_weight_file_parity - HLS weight file generation
    7. test_generate_params_parity - Parameter file generation

    Requires:
    - Inheriting class must also inherit from ParityTestBase
    - setup_manual_op() and setup_auto_op() must be implemented
    """

    @pytest.mark.parity
    @pytest.mark.computational
    def test_calc_wmem_parity(self):
        """Test weight memory depth calculation matches.

        Weight memory depth (WMEM) determines the size of weight storage
        in hardware. Critical for FPGA resource allocation.

        Formula: WMEM = total_weights / PE
        Where PE is the parallelization factor.

        Requirements:
            - Both backends must implement calc_wmem()
            - Memory depths must match exactly

        Raises:
            pytest.skip: If method not implemented (e.g., ops without weights)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "calc_wmem"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement calc_wmem")
        if not hasattr(auto_op, "calc_wmem"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement calc_wmem")

        manual_wmem = manual_op.calc_wmem()
        auto_wmem = auto_op.calc_wmem()

        assert manual_wmem == auto_wmem, (
            f"Weight memory depth mismatch:\n"
            f"  Manual: {manual_wmem:,} entries\n"
            f"  Auto:   {auto_wmem:,} entries"
        )

    @pytest.mark.parity
    @pytest.mark.computational
    def test_calc_tmem_parity(self):
        """Test threshold memory depth calculation matches.

        Threshold memory depth (TMEM) determines the size of activation
        threshold storage in hardware.

        Formula: TMEM = total_thresholds / PE
        Where PE is the parallelization factor.

        Requirements:
            - Both backends must implement calc_tmem()
            - Memory depths must match exactly

        Raises:
            pytest.skip: If method not implemented (e.g., ops without thresholds)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "calc_tmem"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement calc_tmem")
        if not hasattr(auto_op, "calc_tmem"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement calc_tmem")

        manual_tmem = manual_op.calc_tmem()
        auto_tmem = auto_op.calc_tmem()

        assert manual_tmem == auto_tmem, (
            f"Threshold memory depth mismatch:\n"
            f"  Manual: {manual_tmem:,} entries\n"
            f"  Auto:   {auto_tmem:,} entries"
        )

    @pytest.mark.parity
    @pytest.mark.computational
    def test_accumulator_datatype_parity(self):
        """Test accumulator datatype matches between backends.

        The accumulator datatype determines the bit-width used for
        intermediate accumulation during MAC operations. Critical for
        preventing overflow and maintaining numerical precision.

        Requirements:
            - Both backends must implement get_accumulator_datatype()
            - Datatypes must match exactly

        Raises:
            pytest.skip: If method not implemented (e.g., non-computational ops)
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "get_accumulator_datatype"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement get_accumulator_datatype")
        if not hasattr(auto_op, "get_accumulator_datatype"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement get_accumulator_datatype")

        manual_acc_dt = manual_op.get_accumulator_datatype()
        auto_acc_dt = auto_op.get_accumulator_datatype()

        assert manual_acc_dt == auto_acc_dt, (
            f"Accumulator datatype mismatch:\n"
            f"  Manual: {manual_acc_dt.name} ({manual_acc_dt.bitwidth()} bits)\n"
            f"  Auto:   {auto_acc_dt.name} ({auto_acc_dt.bitwidth()} bits)"
        )

    @pytest.mark.parity
    @pytest.mark.computational
    def test_hw_compatible_weight_tensor_parity(self):
        """Test weight tensor hardware transformation matches.

        Weights must be reshaped and interleaved for parallel hardware
        execution (PE parallelism). This test validates that both backends
        produce identical hardware-compatible weight layouts.

        Transformation: (K, C) → (PE, WMEM, SIMD)
        Where weights are interleaved across PE units.

        Requirements:
            - Both backends must implement get_hw_compatible_weight_tensor()
            - Transformed tensors must match exactly (same shape, same values)

        Raises:
            pytest.skip: If method not implemented or no weight tensor
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "get_hw_compatible_weight_tensor"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement get_hw_compatible_weight_tensor")
        if not hasattr(auto_op, "get_hw_compatible_weight_tensor"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement get_hw_compatible_weight_tensor")

        # Get original weight tensor from model
        # Assume weight is second input (index 1)
        if len(manual_op.onnx_node.input) < 2:
            pytest.skip(f"{manual_op.__class__.__name__} has no weight input")

        weight_tensor_name = manual_op.onnx_node.input[1]
        manual_orig_weights = manual_model.get_initializer(weight_tensor_name)
        auto_orig_weights = auto_model.get_initializer(weight_tensor_name)

        if manual_orig_weights is None or auto_orig_weights is None:
            pytest.skip(f"Weight tensor '{weight_tensor_name}' not found in model initializers")

        # Transform weights for hardware
        manual_hw_weights = manual_op.get_hw_compatible_weight_tensor(manual_orig_weights)
        auto_hw_weights = auto_op.get_hw_compatible_weight_tensor(auto_orig_weights)

        # Verify shapes match
        assert manual_hw_weights.shape == auto_hw_weights.shape, (
            f"Hardware-compatible weight tensor shape mismatch:\n"
            f"  Manual: {manual_hw_weights.shape}\n"
            f"  Auto:   {auto_hw_weights.shape}"
        )

        # Verify values match
        np.testing.assert_array_equal(
            manual_hw_weights,
            auto_hw_weights,
            err_msg="Hardware-compatible weight tensors differ"
        )

    @pytest.mark.parity
    @pytest.mark.computational
    def test_hw_compatible_threshold_tensor_parity(self):
        """Test threshold tensor hardware transformation matches.

        Thresholds must be reshaped for parallel hardware execution (PE
        parallelism). This test validates that both backends produce
        identical hardware-compatible threshold layouts.

        Transformation: Reshape and interleave thresholds across PE units

        Requirements:
            - Both backends must implement get_hw_compatible_threshold_tensor()
            - Transformed tensors must match exactly (same shape, same values)

        Raises:
            pytest.skip: If method not implemented or no activation mode
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "get_hw_compatible_threshold_tensor"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement get_hw_compatible_threshold_tensor")
        if not hasattr(auto_op, "get_hw_compatible_threshold_tensor"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement get_hw_compatible_threshold_tensor")

        # Check if in noActivation mode (no thresholds)
        if manual_op.get_nodeattr("noActivation") == 1:
            pytest.skip("Op is in noActivation mode, no thresholds to test")

        # Get original threshold tensor from model
        # Thresholds are typically third input (index 2) for MVAU/VVAU
        if len(manual_op.onnx_node.input) < 3:
            pytest.skip(f"{manual_op.__class__.__name__} has no threshold input")

        threshold_tensor_name = manual_op.onnx_node.input[2]
        manual_orig_thresholds = manual_model.get_initializer(threshold_tensor_name)
        auto_orig_thresholds = auto_model.get_initializer(threshold_tensor_name)

        if manual_orig_thresholds is None or auto_orig_thresholds is None:
            pytest.skip(f"Threshold tensor '{threshold_tensor_name}' not found in model initializers")

        # Transform thresholds for hardware
        manual_hw_thresholds = manual_op.get_hw_compatible_threshold_tensor(manual_orig_thresholds)
        auto_hw_thresholds = auto_op.get_hw_compatible_threshold_tensor(auto_orig_thresholds)

        # Verify shapes match
        assert manual_hw_thresholds.shape == auto_hw_thresholds.shape, (
            f"Hardware-compatible threshold tensor shape mismatch:\n"
            f"  Manual: {manual_hw_thresholds.shape}\n"
            f"  Auto:   {auto_hw_thresholds.shape}"
        )

        # Verify values match
        np.testing.assert_array_equal(
            manual_hw_thresholds,
            auto_hw_thresholds,
            err_msg="Hardware-compatible threshold tensors differ"
        )

    @pytest.mark.parity
    @pytest.mark.computational
    @pytest.mark.slow
    def test_make_weight_file_parity(self):
        """Test HLS weight file generation produces identical files.

        Weight files (.dat or .h) are generated for HLS code generation.
        This test validates that both backends produce bit-identical files.

        File formats:
        - "decoupled": .dat format for streaming weights
        - "embedded": .h format for embedded weights

        Requirements:
            - Both backends must implement make_weight_file()
            - Generated file contents must match exactly

        Raises:
            pytest.skip: If method not implemented or not HLS backend
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "make_weight_file"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement make_weight_file")
        if not hasattr(auto_op, "make_weight_file"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement make_weight_file")

        # Get weights
        if len(manual_op.onnx_node.input) < 2:
            pytest.skip(f"{manual_op.__class__.__name__} has no weight input")

        weight_tensor_name = manual_op.onnx_node.input[1]
        manual_weights = manual_model.get_initializer(weight_tensor_name)
        auto_weights = auto_model.get_initializer(weight_tensor_name)

        if manual_weights is None or auto_weights is None:
            pytest.skip(f"Weight tensor '{weight_tensor_name}' not found")

        # Get hardware-compatible weights
        manual_hw_weights = manual_op.get_hw_compatible_weight_tensor(manual_weights)
        auto_hw_weights = auto_op.get_hw_compatible_weight_tensor(auto_weights)

        # Test both decoupled and embedded modes
        for weight_file_mode in ["decoupled", "embedded"]:
            # Generate weight files
            with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as manual_f:
                manual_file = manual_f.name
                manual_op.make_weight_file(manual_hw_weights, weight_file_mode, manual_file)

            with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as auto_f:
                auto_file = auto_f.name
                auto_op.make_weight_file(auto_hw_weights, weight_file_mode, auto_file)

            # Read and compare file contents
            try:
                with open(manual_file, 'r') as f:
                    manual_content = f.read()
                with open(auto_file, 'r') as f:
                    auto_content = f.read()

                assert manual_content == auto_content, (
                    f"Weight file contents differ for mode '{weight_file_mode}':\n"
                    f"  Manual file: {len(manual_content)} chars\n"
                    f"  Auto file:   {len(auto_content)} chars"
                )
            finally:
                # Cleanup
                os.unlink(manual_file)
                os.unlink(auto_file)

    @pytest.mark.parity
    @pytest.mark.computational
    def test_generate_params_parity(self):
        """Test parameter file generation produces identical files.

        Parameter files contain kernel configuration for RTL/HLS synthesis.
        This test validates that both backends produce identical parameter sets.

        Requirements:
            - Both backends must implement generate_params()
            - Generated parameter files must match exactly

        Raises:
            pytest.skip: If method not implemented
        """
        manual_op, manual_model = self.setup_manual_op()
        auto_op, auto_model = self.setup_auto_op()

        # Check if method exists
        if not hasattr(manual_op, "generate_params"):
            pytest.skip(f"{manual_op.__class__.__name__} does not implement generate_params")
        if not hasattr(auto_op, "generate_params"):
            pytest.skip(f"{auto_op.__class__.__name__} does not implement generate_params")

        # Create temporary directories for parameter generation
        manual_tmpdir = tempfile.mkdtemp(prefix="params_manual_")
        auto_tmpdir = tempfile.mkdtemp(prefix="params_auto_")

        try:
            # Generate parameters
            manual_op.generate_params(manual_model, manual_tmpdir)
            auto_op.generate_params(auto_model, auto_tmpdir)

            # List all generated files
            manual_files = sorted(os.listdir(manual_tmpdir))
            auto_files = sorted(os.listdir(auto_tmpdir))

            # Verify same files were generated
            assert manual_files == auto_files, (
                f"Generated parameter files differ:\n"
                f"  Manual: {manual_files}\n"
                f"  Auto:   {auto_files}"
            )

            # Compare contents of each file
            for filename in manual_files:
                manual_path = os.path.join(manual_tmpdir, filename)
                auto_path = os.path.join(auto_tmpdir, filename)

                with open(manual_path, 'r') as f:
                    manual_content = f.read()
                with open(auto_path, 'r') as f:
                    auto_content = f.read()

                assert manual_content == auto_content, (
                    f"Parameter file '{filename}' contents differ:\n"
                    f"  Manual size: {len(manual_content)} bytes\n"
                    f"  Auto size:   {len(auto_content)} bytes"
                )

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(manual_tmpdir, ignore_errors=True)
            shutil.rmtree(auto_tmpdir, ignore_errors=True)
