"""
Interfaces for Phase 2, including the BuildRunnerInterface for Phase 3 integration.
"""

from abc import ABC, abstractmethod
import random
import time
from datetime import datetime

from .data_structures import BuildConfig, BuildResult, BuildStatus
from ..phase1.data_structures import BuildMetrics


class BuildRunnerInterface(ABC):
    """
    Abstract interface for build runners.
    
    This interface defines how Phase 2 interacts with Phase 3 build runners.
    Different implementations can target different backends (FINN, etc.).
    """
    
    @abstractmethod
    def run(self, config: BuildConfig) -> BuildResult:
        """
        Execute a build with the given configuration.
        
        Args:
            config: The build configuration to execute
            
        Returns:
            BuildResult with status, metrics, and artifacts
        """
        pass


class MockBuildRunner(BuildRunnerInterface):
    """
    Mock build runner for testing Phase 2 without Phase 3.
    
    Simulates build execution with configurable success rates and
    generates fake metrics for testing.
    """
    
    def __init__(
        self, 
        success_rate: float = 0.8,
        min_duration: float = 1.0,
        max_duration: float = 5.0,
        simulate_delay: bool = False
    ):
        """
        Initialize the mock build runner.
        
        Args:
            success_rate: Probability of build success (0.0 to 1.0)
            min_duration: Minimum simulated build duration in seconds
            max_duration: Maximum simulated build duration in seconds
            simulate_delay: Whether to actually sleep during builds
        """
        self.success_rate = success_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.simulate_delay = simulate_delay
    
    def run(self, config: BuildConfig) -> BuildResult:
        """Execute a mock build."""
        result = BuildResult(
            config_id=config.id,
            status=BuildStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # Simulate build duration
        duration = random.uniform(self.min_duration, self.max_duration)
        if self.simulate_delay:
            time.sleep(duration)
        
        # Determine success/failure
        if random.random() < self.success_rate:
            # Successful build - generate fake metrics
            result.metrics = self._generate_fake_metrics(config)
            result.complete(BuildStatus.SUCCESS)
            
            # Add fake artifacts
            result.artifacts = {
                "bitstream": f"/tmp/builds/{config.id}/design.bit",
                "reports": f"/tmp/builds/{config.id}/reports/",
                "logs": f"/tmp/builds/{config.id}/logs/",
            }
        else:
            # Failed build
            error_messages = [
                "Timing constraints not met",
                "Resource utilization exceeded",
                "Synthesis failed",
                "Place and route failed",
                "Invalid configuration",
            ]
            result.complete(
                BuildStatus.FAILED,
                error_message=random.choice(error_messages)
            )
        
        # Add fake logs
        result.logs = {
            "synthesis": f"Mock synthesis log for {config.id}\n" * 10,
            "implementation": f"Mock implementation log for {config.id}\n" * 20,
        }
        
        return result
    
    def _generate_fake_metrics(self, config: BuildConfig) -> BuildMetrics:
        """Generate realistic-looking fake metrics."""
        # Base metrics with some randomness
        base_throughput = 1000.0
        base_latency = 10.0
        base_lut = 0.5
        
        # Adjust based on configuration
        # More kernels = lower throughput but higher resource usage
        kernel_factor = len(config.kernels) * 0.1
        base_throughput /= (1 + kernel_factor)
        base_lut *= (1 + kernel_factor * 0.5)
        
        # More transforms = higher latency
        transform_factor = len(config.transforms) * 0.05
        base_latency *= (1 + transform_factor)
        
        # Add randomness
        throughput = base_throughput * random.uniform(0.8, 1.2)
        latency = base_latency * random.uniform(0.9, 1.1)
        
        return BuildMetrics(
            throughput=throughput,
            latency=latency,
            clock_frequency=random.uniform(200, 300),  # MHz
            lut_utilization=min(base_lut * random.uniform(0.9, 1.1), 0.95),
            dsp_utilization=random.uniform(0.3, 0.7),
            bram_utilization=random.uniform(0.2, 0.6),
            total_power=random.uniform(5, 15),  # Watts
            accuracy=random.uniform(0.95, 0.99),  # Model accuracy
            custom={"mock": True}
        )