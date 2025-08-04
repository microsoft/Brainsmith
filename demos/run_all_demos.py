#!/usr/bin/env python3
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Master Demo Runner for Kernel Integrator & Kernel Modeling System

This script provides a unified interface to run all demos, perfect for
presentations and comprehensive demonstrations.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    import questionary
    import typer
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Some dependencies not available.")
    print("Install with: pip install rich questionary typer")

# Demo registry
DEMOS = [
    {
        "number": 1,
        "name": "RTL to FINN in 30 Seconds",
        "file": "demo_01_rtl_to_finn.py",
        "duration": "2-3 min",
        "status": "ready",
        "description": "Live conversion from SystemVerilog to FINN HWCustomOp"
    },
    {
        "number": 2,
        "name": "RTL Parser Interactive Explorer",
        "file": "demo_02_rtl_parser.py",
        "duration": "3-4 min",
        "status": "ready",
        "description": "Rich visualization of RTL parsing and pragma extraction"
    }
]


class DemoRunner:
    """Main demo runner class."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.demos_dir = Path(__file__).parent
        self.results = []
    
    def run_interactive(self):
        """Run demos in interactive mode with menu."""
        while True:
            self._show_menu()
            
            if RICH_AVAILABLE and questionary:
                choices = []
                for demo in DEMOS:
                    status = "✅" if demo['status'] == 'ready' else "🚧"
                    choice = f"{demo['number']:2d}. {status} {demo['name']} ({demo['duration']})"
                    choices.append(choice)
                
                choices.extend([
                    "─" * 50,
                    "A. Run all ready demos",
                    "R. Generate report",
                    "Q. Quit"
                ])
                
                selection = questionary.select(
                    "Select a demo to run:",
                    choices=choices
                ).ask()
                
                if not selection:
                    break
                
                if selection.startswith("Q"):
                    break
                elif selection.startswith("A"):
                    self._run_all_demos()
                elif selection.startswith("R"):
                    self._generate_report()
                elif "─" not in selection:
                    # Extract demo number
                    demo_num = int(selection.split('.')[0])
                    self._run_single_demo(demo_num)
            else:
                # Fallback menu
                print("\nSelect option:")
                print("1-10: Run specific demo")
                print("A: Run all ready demos")
                print("R: Generate report")
                print("Q: Quit")
                
                choice = input("\nChoice: ").strip().upper()
                
                if choice == 'Q':
                    break
                elif choice == 'A':
                    self._run_all_demos()
                elif choice == 'R':
                    self._generate_report()
                elif choice.isdigit():
                    self._run_single_demo(int(choice))
    
    def run_presentation(self, delay: int = 5):
        """Run all ready demos in presentation mode."""
        ready_demos = [d for d in DEMOS if d['status'] == 'ready']
        
        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[bold cyan]Presentation Mode[/bold cyan]\n\n"
                    f"Running {len(ready_demos)} demos with {delay}s delay between each.",
                    title="🎥 Demo Presentation",
                    border_style="cyan"
                )
            )
        else:
            print("\n🎥 PRESENTATION MODE")
            print(f"Running {len(ready_demos)} demos")
        
        for i, demo in enumerate(ready_demos, 1):
            if i > 1:
                print(f"\n⏱️  Next demo in {delay} seconds...")
                time.sleep(delay)
            
            self._run_demo(demo, presentation_mode=True)
        
        print("\n✅ Presentation complete!")
    
    def _show_menu(self):
        """Display the demo menu."""
        if RICH_AVAILABLE:
            # Create fancy table
            table = Table(title="🚀 Kernel Integrator Demo Suite", show_header=True)
            table.add_column("#", style="cyan", width=3)
            table.add_column("Status", width=6)
            table.add_column("Demo Name", style="yellow")
            table.add_column("Duration", style="green")
            table.add_column("Description", style="dim")
            
            for demo in DEMOS:
                status = "✅" if demo['status'] == 'ready' else "🚧"
                table.add_row(
                    str(demo['number']),
                    status,
                    demo['name'],
                    demo['duration'],
                    demo['description']
                )
            
            self.console.print(table)
        else:
            print("\n" + "=" * 70)
            print("KERNEL INTEGRATOR DEMO SUITE")
            print("=" * 70)
            
            for demo in DEMOS:
                status = "[Ready]" if demo['status'] == 'ready' else "[Planned]"
                print(f"{demo['number']:2d}. {status:9} {demo['name']:<30} ({demo['duration']})")
    
    def _run_single_demo(self, number: int):
        """Run a single demo by number."""
        demo = next((d for d in DEMOS if d['number'] == number), None)
        
        if not demo:
            print(f"❌ Demo {number} not found")
            return
        
        if demo['status'] != 'ready':
            print(f"🚧 Demo {number} is not yet implemented")
            return
        
        self._run_demo(demo)
    
    def _run_demo(self, demo: Dict, presentation_mode: bool = False):
        """Execute a demo."""
        demo_file = self.demos_dir / demo['file']
        
        if not demo_file.exists():
            print(f"❌ Demo file not found: {demo_file}")
            return
        
        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[bold]Demo {demo['number']}: {demo['name']}[/bold]\n"
                    f"Duration: {demo['duration']}\n"
                    f"{demo['description']}",
                    title="🎯 Starting Demo",
                    border_style="green"
                )
            )
        else:
            print(f"\n🎯 Starting Demo {demo['number']}: {demo['name']}")
            print("-" * 50)
        
        # Run the demo
        start_time = time.time()
        
        args = ["python", str(demo_file)]
        if presentation_mode:
            args.append("--non-interactive")
        
        result = subprocess.run(args)
        
        elapsed = time.time() - start_time
        
        # Record result
        self.results.append({
            "demo": demo['name'],
            "number": demo['number'],
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed
        })
        
        if result.returncode == 0:
            print(f"\n✅ Demo completed in {elapsed:.1f} seconds")
        else:
            print(f"\n❌ Demo failed with code {result.returncode}")
        
        if not presentation_mode:
            input("\nPress Enter to continue...")
    
    def _run_all_demos(self):
        """Run all ready demos."""
        ready_demos = [d for d in DEMOS if d['status'] == 'ready']
        
        print(f"\n🚀 Running {len(ready_demos)} ready demos...")
        
        for demo in ready_demos:
            self._run_demo(demo)
    
    def _generate_report(self):
        """Generate a summary report of demo runs."""
        if not self.results:
            print("No demos have been run yet.")
            return
        
        report_file = Path("demo_outputs/demo_run_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_demos": len(DEMOS),
            "ready_demos": len([d for d in DEMOS if d['status'] == 'ready']),
            "demos_run": len(self.results),
            "results": self.results,
            "total_time": sum(r['elapsed_seconds'] for r in self.results)
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_file}")
        
        if RICH_AVAILABLE:
            # Show summary table
            summary = Table(title="Demo Run Summary")
            summary.add_column("Demo", style="cyan")
            summary.add_column("Status", style="green")
            summary.add_column("Time", justify="right")
            
            for result in self.results:
                status = "✅ Success" if result['success'] else "❌ Failed"
                time_str = f"{result['elapsed_seconds']:.1f}s"
                summary.add_row(result['demo'], status, time_str)
            
            summary.add_row("", "", "")
            summary.add_row(
                "[bold]Total[/bold]",
                f"{len(self.results)} demos",
                f"{report['total_time']:.1f}s"
            )
            
            self.console.print(summary)


def main():
    """Main entry point with CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Kernel Integrator demos"
    )
    parser.add_argument(
        '--presentation',
        action='store_true',
        help='Run in presentation mode (auto-advance)'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=5,
        help='Delay between demos in presentation mode (seconds)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate report after running'
    )
    parser.add_argument(
        '--demo',
        type=int,
        help='Run specific demo number'
    )
    
    args = parser.parse_args()
    
    runner = DemoRunner()
    
    if args.demo:
        runner._run_single_demo(args.demo)
    elif args.presentation:
        runner.run_presentation(args.delay)
        if args.report:
            runner._generate_report()
    else:
        runner.run_interactive()


if __name__ == '__main__':
    main()