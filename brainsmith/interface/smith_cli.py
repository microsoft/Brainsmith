"""Smith operational CLI wrapper.

This provides the standalone 'smith' command that inherits user defaults
and delegates to the refactored CLI.
"""

import sys
from typing import List, Optional

from .context import ApplicationContext
from .cli import cli as original_smith_cli


def run_smith(args: List[str], context: Optional[ApplicationContext] = None) -> None:
    """Run smith CLI with given arguments and optional inherited context.
    
    Args:
        args: Command-line arguments to pass to smith
        context: Optional ApplicationContext from brainsmith
    """
    # If no context provided, create one with user defaults
    if context is None:
        context = ApplicationContext()
    
    # Ensure configuration is loaded  
    if context.config is None:
        context.load_configuration()
    
    # Export environment for the session
    config = context.get_effective_config()
    config.export_to_environment(verbose=context.verbose)
    
    # Convert args list back to sys.argv format
    # Insert 'smith' as program name if not present
    if not args or args[0] != 'smith':
        args = ['smith'] + list(args)
    
    # Replace sys.argv temporarily
    original_argv = sys.argv
    try:
        sys.argv = args
        
        # Pass context through Click's obj
        original_smith_cli(obj=context, standalone_mode=False)
        
    except SystemExit as e:
        # Re-raise to preserve exit code
        raise
    except Exception:
        # Let the CLI handle exceptions
        raise
    finally:
        # Restore original argv
        sys.argv = original_argv


def main() -> None:
    """Main entry point for standalone smith command."""
    # Run with user defaults, passing all arguments
    try:
        run_smith(sys.argv)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()