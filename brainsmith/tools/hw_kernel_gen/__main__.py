"""
Main entry point for running HWKG as a module.

Enables running: python -m brainsmith.tools.hw_kernel_gen

This implementation follows the Interface-Wise Dataflow Modeling axioms
and provides a simple-by-default, powerful-when-needed approach to hardware
kernel generation.
"""

import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main())