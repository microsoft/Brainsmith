"""Session-scoped model cache for computational reuse.

Caches expensive artifacts across test depths:
- Stage 1 Models: f(test_case.test_id)
- Golden References: f(test_case.test_id)
- Stage 2 Models: f(test_case.test_id)
- Stage 3 Models: f(test_case.test_id, platform.fpgapart)

Scope: session (shared across all tests)
Population: Lazy (built on first access)
Storage: In-memory dictionaries
"""

from collections.abc import Callable

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper


class ModelCache:
    """Session-scoped cache for test artifacts.

    Provides get-or-build pattern for all cached artifacts.
    Thread-safe for parallel test execution.
    """

    def __init__(self):
        # Cache dictionaries
        self._stage1_models: dict[str, ModelWrapper] = {}
        self._stage2_models: dict[str, tuple] = {}
        self._stage3_models: dict[tuple[str, str], tuple] = {}
        self._golden_outputs: dict[str, dict[str, np.ndarray]] = {}
        self._test_inputs: dict[str, dict[str, np.ndarray]] = {}

        # Statistics (for debugging)
        self.stats = {
            "stage1_hits": 0,
            "stage1_misses": 0,
            "stage2_hits": 0,
            "stage2_misses": 0,
            "stage3_hits": 0,
            "stage3_misses": 0,
            "golden_hits": 0,
            "golden_misses": 0,
        }

    def get_stage1_model(self, test_id: str, builder: Callable[[], ModelWrapper]) -> ModelWrapper:
        """Get or build Stage 1 ONNX model.

        Args:
            test_id: Unique test case identifier (cache key)
            builder: Callable that builds the model if cache miss

        Returns:
            Stage 1 ONNX model with QONNX annotations
        """
        if test_id in self._stage1_models:
            self.stats["stage1_hits"] += 1
            return self._stage1_models[test_id]

        self.stats["stage1_misses"] += 1
        model = builder()
        self._stage1_models[test_id] = model
        return model

    def get_stage2_model(self, test_id: str, builder: Callable[[], tuple]) -> tuple:
        """Get or build Stage 2 Kernel model.

        Args:
            test_id: Unique test case identifier (cache key)
            builder: Callable that builds (op, model) if cache miss

        Returns:
            (kernel_op, model) tuple for Stage 2
        """
        if test_id in self._stage2_models:
            self.stats["stage2_hits"] += 1
            return self._stage2_models[test_id]

        self.stats["stage2_misses"] += 1
        op, model = builder()
        self._stage2_models[test_id] = (op, model)
        return op, model

    def get_stage3_model(self, test_id: str, fpgapart: str, builder: Callable[[], tuple]) -> tuple:
        """Get or build Stage 3 Backend model.

        Args:
            test_id: Unique test case identifier
            fpgapart: FPGA part number (platform identifier)
            builder: Callable that builds (op, model) if cache miss

        Returns:
            (backend_op, model) tuple for Stage 3
        """
        cache_key = (test_id, fpgapart)

        if cache_key in self._stage3_models:
            self.stats["stage3_hits"] += 1
            return self._stage3_models[cache_key]

        self.stats["stage3_misses"] += 1
        op, model = builder()
        self._stage3_models[cache_key] = (op, model)
        return op, model

    def get_golden_outputs(
        self, test_id: str, builder: Callable[[], dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Get or compute golden reference outputs.

        Args:
            test_id: Unique test case identifier (cache key)
            builder: Callable that computes golden if cache miss

        Returns:
            Dict mapping output names to numpy arrays
        """
        if test_id in self._golden_outputs:
            self.stats["golden_hits"] += 1
            return self._golden_outputs[test_id]

        self.stats["golden_misses"] += 1
        golden = builder()
        self._golden_outputs[test_id] = golden
        return golden

    def get_test_inputs(
        self, test_id: str, builder: Callable[[], dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Get or generate test inputs.

        Args:
            test_id: Unique test case identifier (cache key)
            builder: Callable that generates inputs if cache miss

        Returns:
            Dict mapping input names to numpy arrays
        """
        if test_id in self._test_inputs:
            return self._test_inputs[test_id]

        inputs = builder()
        self._test_inputs[test_id] = inputs
        return inputs

    def print_stats(self):
        """Print cache statistics (useful for optimization analysis)."""
        total_accesses = sum(v for k, v in self.stats.items() if "hits" in k or "misses" in k)
        total_hits = sum(v for k, v in self.stats.items() if "hits" in k)

        print("\n" + "=" * 60)
        print("Model Cache Statistics")
        print("=" * 60)
        print(f"Stage 1: {self.stats['stage1_hits']} hits, {self.stats['stage1_misses']} misses")
        print(f"Stage 2: {self.stats['stage2_hits']} hits, {self.stats['stage2_misses']} misses")
        print(f"Stage 3: {self.stats['stage3_hits']} hits, {self.stats['stage3_misses']} misses")
        print(f"Golden:  {self.stats['golden_hits']} hits, {self.stats['golden_misses']} misses")
        if total_accesses > 0:
            print(
                f"\nTotal: {total_hits}/{total_accesses} hits ({100*total_hits/total_accesses:.1f}%)"
            )
        print("=" * 60 + "\n")
