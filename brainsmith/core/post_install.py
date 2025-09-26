"""Post-install hook for Poetry - displays setup instructions."""

import os
import sys
from pathlib import Path


def main():
    """Main entry point for post-install hook."""
    print("\n" + "=" * 60)
    print("🎉 Brainsmith installation complete!")
    print("=" * 60)
    
    print("\nTo set up optional dependencies, run:")
    print("  brainsmith setup all        # Install all dependencies")
    print("  brainsmith setup cppsim     # C++ simulation support")
    print("  brainsmith setup xsim       # Xilinx simulation support (requires Vivado)")
    print("  brainsmith setup boards     # Download board definition files")
    
    print("\nTo check setup status:")
    print("  brainsmith setup check")
    
    print("\nTo get started:")
    print("  smith --help                # Operational commands (DSE, kernels)")
    print("  brainsmith --help           # Configuration and setup")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()