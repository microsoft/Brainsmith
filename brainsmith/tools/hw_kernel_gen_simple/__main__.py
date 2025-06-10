"""
Main entry point for running HWKG as a module.

Enables running: python -m brainsmith.tools.hw_kernel_gen_simple
"""

from .cli import main

if __name__ == '__main__':
    main()