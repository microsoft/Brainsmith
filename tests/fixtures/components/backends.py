"""Real test backend components."""

from brainsmith.registry import backend


@backend(name="TestKernel_hls", target_kernel="test:TestKernel", language="hls")
class TestKernel_hls:
    """HLS backend for TestKernel."""

    def __init__(self, node):
        self.node = node

    def generate(self):
        """Generate HLS code."""
        return "// HLS implementation"


@backend(name="TestKernel_rtl", target_kernel="test:TestKernel", language="rtl")
class TestKernel_rtl:
    """RTL backend for TestKernel."""

    def __init__(self, node):
        self.node = node

    def generate(self):
        """Generate RTL code."""
        return "-- VHDL implementation"


@backend(name="TestKernel2_hls", target_kernel="test:TestKernel2", language="hls")
class TestKernel2_hls:
    """HLS backend for TestKernel2."""

    def __init__(self, node):
        self.node = node

    def generate(self):
        """Generate HLS code."""
        return "// HLS implementation for TestKernel2"


@backend(name="AnotherTestKernel_hls", target_kernel="test:AnotherTestKernel", language="hls")
class AnotherTestKernel_hls:
    """HLS backend for AnotherTestKernel."""

    def __init__(self, node):
        self.node = node
