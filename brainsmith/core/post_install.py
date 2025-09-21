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
    print("  smith setup all        # Install all dependencies")
    print("  smith setup cppsim     # C++ simulation support")
    print("  smith setup xsim       # Xilinx simulation support (requires Vivado)")
    print("  smith setup boards     # Download board definition files")
    
    print("\nTo check setup status:")
    print("  smith setup check")
    
    print("\nTo get started:")
    print("  smith --help")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()