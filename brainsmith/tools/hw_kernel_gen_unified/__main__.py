"""
Main entry point for running unified HWKG as a module.

Enables running: python -m brainsmith.tools.hw_kernel_gen_unified

This unified implementation follows the Interface-Wise Dataflow Modeling axioms
and provides a simple-by-default, powerful-when-needed approach to hardware
kernel generation.
"""

from .cli import main

if __name__ == '__main__':
    main()