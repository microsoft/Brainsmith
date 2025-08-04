#!/bin/bash
# Script to run Kernel Integrator demos

echo "🚀 Kernel Integrator Demos"
echo "==================================================="
echo ""
echo "This script will guide you through running the demos."
echo ""

# Function to run a demo
run_demo() {
    local demo_num=$1
    local demo_name=$2
    local demo_file=$3
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Demo $demo_num: $demo_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    read -p "Press Enter to run Demo $demo_num (or 's' to skip): " choice
    
    if [[ "$choice" != "s" ]]; then
        python $demo_file
        echo ""
        read -p "Press Enter to continue..."
    fi
}

# Main menu
while true; do
    echo ""
    echo "Available Demos:"
    echo "1. RTL to FINN in 30 Seconds"
    echo "2. RTL Parser Interactive Explorer" 
    echo "A. Run all demos"
    echo "Q. Quit"
    echo ""
    read -p "Select option (1/2/A/Q): " option
    
    case $option in
        1)
            run_demo 1 "RTL to FINN in 30 Seconds" "demo_01_rtl_to_finn.py"
            ;;
        2)
            echo ""
            echo "Demo 2 requires interactive file selection."
            echo "Running with the master demo runner..."
            python run_all_demos.py --demo 2
            ;;
        A|a)
            echo ""
            echo "Running all demos in sequence..."
            run_demo 1 "RTL to FINN in 30 Seconds" "demo_01_rtl_to_finn.py"
            
            echo ""
            echo "Running Demo 2 with example file..."
            python demo_02_rtl_parser.py --file ../tests/tools/kernel_integrator/rtl_parser/demo_rtl/01_basic_module.sv
            ;;
        Q|q)
            echo "Exiting demo runner."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done