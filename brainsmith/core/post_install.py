"""Post-install hook for Poetry to set up simulation dependencies."""

import os
import sys
from pathlib import Path


def main():
    """Main entry point for post-install hook."""
    # Check if --no-sim flag was passed
    NO_SIM = "--no-sim" in sys.argv or os.environ.get("BRAINSMITH_NO_SIM") == "1"
    
    if not NO_SIM:
        print("\n=== Setting up Brainsmith simulation dependencies ===")
        
        # Import the simulation setup
        try:
            from brainsmith.core.plugins.simulation import SimulationSetup
            
            sim_setup = SimulationSetup()
            
            # Try to set up C++ simulation
            print("\nSetting up C++ simulation...")
            if sim_setup.setup_cppsim():
                print("✓ C++ simulation ready")
            else:
                print("✗ C++ simulation setup failed (missing prerequisites?)")
            
            # Try to set up RTL simulation if Vivado is available
            if os.environ.get("XILINX_VIVADO"):
                print("\nSetting up RTL simulation...")
                if sim_setup.setup_rtlsim():
                    print("✓ RTL simulation ready")
                else:
                    print("✗ RTL simulation setup failed")
            else:
                print("\nSkipping RTL simulation (Vivado not found)")
            
            print("\n✓ Simulation setup complete!")
            print("\nTo skip simulation setup in the future, use:")
            print("  export BRAINSMITH_NO_SIM=1")
            print("  poetry install")
            
        except Exception as e:
            print(f"\nWarning: Could not set up simulation dependencies: {e}")
            print("This is not critical - you can set them up later if needed.")
    else:
        print("\nSkipping simulation setup (BRAINSMITH_NO_SIM=1)")


if __name__ == "__main__":
    main()