"""Test configuration via composition (v4.0).

Compositional test configuration separating concerns into reusable sub-configs.
Each sub-config owns a distinct concern: model structure, design parameters,
platform configuration, or validation criteria.

Architecture:
    ModelStructure: What we're testing (operation, shapes, dtypes)
    DesignParameters: How we configure it (PE, SIMD, backend variants)
    PlatformConfig: Where we run it (FPGA part)
    ValidationConfig: How we validate it (tolerances)

Usage:
    # Minimal configuration
    config = KernelTestConfig(
        test_id="add_int8_baseline",
        model=ModelStructure(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]}
        )
    )

    # Full configuration with reusable sub-configs
    ZYNQ_7020 = PlatformConfig(fpgapart="xc7z020clg400-1")
    PE8_DESIGN = DesignParameters(input_streams={0: 8})
    TIGHT_VALIDATION = ValidationConfig(
        tolerance_python={"rtol": 1e-10, "atol": 1e-12}
    )

    config = KernelTestConfig(
        test_id="add_int8_pe8_cppsim",
        model=ModelStructure("Add", shapes, dtypes),
        design=PE8_DESIGN,
        platform=ZYNQ_7020,
        validation=TIGHT_VALIDATION
    )

    # Reuse sub-configs across multiple tests
    configs = [
        KernelTestConfig(
            test_id=f"add_{dtype}_{size}",
            model=ModelStructure("Add", shapes[size], dtypes[dtype]),
            platform=ZYNQ_7020,
            validation=TIGHT_VALIDATION
        )
        for dtype in ["int8", "int16"]
        for size in ["small", "large"]
    ]

v4.0 Changes from v3.0:
- Composition over monolithic structure
- Reusable sub-configs (share platform/validation across tests)
- Immutability where appropriate (frozen dataclasses)
- No auto-generation (explicit test_id required)
- No factory methods (dataclasses are simple enough)
- backend_variants moved to DesignParameters (it's a design choice)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qonnx.core.datatype import DataType


@dataclass(frozen=True)
class ModelStructure:
    """What we're testing - the model topology.

    Immutable after creation. Validates that shapes and dtypes are consistent.

    Attributes:
        operation: ONNX operation name (e.g., "Add", "MatMul", "Conv")
        input_shapes: Dict mapping input names to shapes
        input_dtypes: Dict mapping input names to DataTypes

    Example:
        model = ModelStructure(
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]}
        )
    """
    operation: str
    input_shapes: Dict[str, Tuple[int, ...]]
    input_dtypes: Dict[str, DataType]

    def __post_init__(self):
        """Validate that shapes and dtypes have matching keys."""
        if set(self.input_shapes.keys()) != set(self.input_dtypes.keys()):
            raise ValueError(
                f"Shape keys {set(self.input_shapes.keys())} must match "
                f"dtype keys {set(self.input_dtypes.keys())}"
            )


@dataclass
class DesignParameters:
    """How we configure the implementation - DSE dimensions.

    Mutable to allow DSE exploration. Empty instance = baseline configuration.

    Attributes:
        input_streams: Stream parallelism for inputs (e.g., {0: 8} sets PE=8)
        output_streams: Stream parallelism for outputs (e.g., {0: 16})
        dimensions: Other DSE dimensions (e.g., {"SIMD": 32, "mem_mode": "internal"})
        backend_variants: Backend variants to test (e.g., ("hls", "rtl"))

    Stage Availability:
        - Stage 2 (Kernel): input_streams, output_streams, dimensions (SIMD, etc.)
        - Stage 3 (Backend): dimensions (mem_mode, ram_style, resType)

    Example:
        # Baseline (no parallelization)
        baseline = DesignParameters()

        # PE parallelization
        pe8 = DesignParameters(input_streams={0: 8})

        # Full configuration
        full = DesignParameters(
            input_streams={0: 8},
            output_streams={0: 16},
            dimensions={"SIMD": 32, "mem_mode": "internal"},
            backend_variants=("hls", "rtl")
        )
    """
    input_streams: Dict[int, int] = field(default_factory=dict)
    output_streams: Dict[int, int] = field(default_factory=dict)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    backend_variants: Tuple[str, ...] = ("hls",)


@dataclass(frozen=True)
class PlatformConfig:
    """Where we run - hardware platform.

    Immutable platform identity. fpgapart=None means software execution only.

    Attributes:
        fpgapart: FPGA part number (e.g., "xc7z020clg400-1") or None

    Future extensions:
        - xilinx_version: Vivado version
        - device_family: Zynq, UltraScale+, etc.
        - board: Target board name

    Example:
        # Software only (no backend testing)
        software = PlatformConfig()

        # Zynq 7020 hardware
        zynq = PlatformConfig(fpgapart="xc7z020clg400-1")

        # UltraScale+
        ultrascale = PlatformConfig(fpgapart="xczu9eg-ffvb1156-2-e")
    """
    fpgapart: Optional[str] = None


@dataclass(frozen=True)
class ValidationConfig:
    """How we validate - numerical tolerances for each execution mode.

    Immutable test criteria. Provides sensible defaults for each execution mode.

    Attributes:
        tolerance_python: Tolerance for Python execution
        tolerance_cppsim: Tolerance for C++ simulation
        tolerance_rtlsim: Tolerance for RTL simulation (defaults to cppsim)

    Example:
        # Default tolerances
        default = ValidationConfig()

        # Tight tolerances for exact operations
        tight = ValidationConfig(
            tolerance_python={"rtol": 1e-10, "atol": 1e-12},
            tolerance_cppsim={"rtol": 1e-7, "atol": 1e-9}
        )

        # Relaxed tolerances for approximate operations
        relaxed = ValidationConfig(
            tolerance_python={"rtol": 1e-5, "atol": 1e-6},
            tolerance_cppsim={"rtol": 1e-3, "atol": 1e-4}
        )
    """
    tolerance_python: Dict[str, float] = field(
        default_factory=lambda: {"rtol": 1e-7, "atol": 1e-9}
    )
    tolerance_cppsim: Dict[str, float] = field(
        default_factory=lambda: {"rtol": 1e-5, "atol": 1e-6}
    )
    tolerance_rtlsim: Optional[Dict[str, float]] = None

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Get RTL tolerance, defaulting to cppsim if not specified."""
        return self.tolerance_rtlsim or self.tolerance_cppsim


@dataclass
class KernelTestConfig:
    """Composed test configuration - each concern separated.

    Composes sub-configs to build complete test configuration. No inheritance,
    pure composition.

    Required:
        test_id: Unique test identifier for pytest
        model: ModelStructure defining the operation

    Optional (sensible defaults):
        design: DesignParameters (default: baseline)
        platform: PlatformConfig (default: software only)
        validation: ValidationConfig (default: standard tolerances)
        marks: List of pytest marks

    Example:
        # Minimal test - software only, no parallelization
        config = KernelTestConfig(
            test_id="add_int8_baseline",
            model=ModelStructure(
                operation="Add",
                input_shapes={"input": (1, 64), "param": (1, 64)},
                input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]}
            )
        )

        # Full test with backend, parallelization, custom tolerances
        config = KernelTestConfig(
            test_id="add_int8_pe8_cppsim",
            model=ModelStructure("Add", shapes, dtypes),
            design=DesignParameters(input_streams={0: 8}),
            platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            validation=ValidationConfig(tolerance_python={"rtol": 1e-10, "atol": 1e-12})
        )

        # Reuse sub-configs
        ZYNQ = PlatformConfig(fpgapart="xc7z020clg400-1")
        configs = [
            KernelTestConfig(test_id=f"test_{i}", model=m, platform=ZYNQ)
            for i, m in enumerate(models)
        ]
    """
    test_id: str
    model: ModelStructure
    design: DesignParameters = field(default_factory=DesignParameters)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    marks: List[Any] = field(default_factory=list)

    # ========================================================================
    # Compatibility accessors for framework code
    # ========================================================================

    @property
    def operation(self) -> str:
        """Access model.operation directly."""
        return self.model.operation

    @property
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Access model.input_shapes directly."""
        return self.model.input_shapes

    @property
    def input_dtypes(self) -> Dict[str, DataType]:
        """Access model.input_dtypes directly."""
        return self.model.input_dtypes

    @property
    def input_streams(self) -> Dict[int, int]:
        """Access design.input_streams directly."""
        return self.design.input_streams

    @property
    def output_streams(self) -> Dict[int, int]:
        """Access design.output_streams directly."""
        return self.design.output_streams

    @property
    def dse_dimensions(self) -> Dict[str, Any]:
        """Access design.dimensions directly."""
        return self.design.dimensions

    @property
    def backend_variants(self) -> Tuple[str, ...]:
        """Access design.backend_variants directly."""
        return self.design.backend_variants

    @property
    def fpgapart(self) -> Optional[str]:
        """Access platform.fpgapart directly."""
        return self.platform.fpgapart

    @property
    def tolerance_python(self) -> Dict[str, float]:
        """Access validation.tolerance_python directly."""
        return self.validation.tolerance_python

    @property
    def tolerance_cppsim(self) -> Dict[str, float]:
        """Access validation.tolerance_cppsim directly."""
        return self.validation.tolerance_cppsim

    @property
    def tolerance_rtlsim(self) -> Optional[Dict[str, float]]:
        """Access validation.tolerance_rtlsim directly."""
        return self.validation.tolerance_rtlsim

    # ========================================================================
    # Framework delegation methods (match v3.0 API)
    # ========================================================================

    def get_tolerance_python(self) -> Dict[str, float]:
        """Get Python execution tolerance (always returns dict, never None)."""
        return self.validation.tolerance_python

    def get_tolerance_cppsim(self) -> Dict[str, float]:
        """Get C++ simulation tolerance (always returns dict, never None)."""
        return self.validation.tolerance_cppsim

    def get_tolerance_rtlsim(self) -> Dict[str, float]:
        """Get RTL simulation tolerance (defaults to cppsim)."""
        return self.validation.get_tolerance_rtlsim()

    def get_fpgapart(self) -> Optional[str]:
        """Get FPGA part string (None if software-only testing)."""
        return self.platform.fpgapart

    def has_backend_testing(self) -> bool:
        """Check if backend testing is enabled (fpgapart configured)."""
        return self.platform.fpgapart is not None
