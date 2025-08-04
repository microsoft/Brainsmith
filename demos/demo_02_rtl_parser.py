#!/usr/bin/env python3
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Demo 2: RTL Parser Interactive Explorer

This demo provides an enhanced wrapper around the RTL parser demo with
interactive file selection, rich visualizations, and multi-format export.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos.common.utils import (
    create_demo_header, highlight_code, save_demo_output,
    wait_for_input, Timer
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.progress import track
    import questionary
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' and 'questionary' not available.")
    print("Install with: pip install rich questionary")


class RTLParserExplorer:
    """Interactive RTL Parser Explorer."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.demo_rtl_dir = Path("tests/tools/kernel_integrator/rtl_parser/demo_rtl")
        self.parser_demo = Path("tests/tools/kernel_integrator/rtl_parser/rtl_parser_demo.py")
        self.output_dir = Path("demo_outputs/rtl_parser")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Run the interactive explorer."""
        create_demo_header(
            "Demo 2: RTL Parser Interactive Explorer",
            "Explore SystemVerilog parsing with rich visualizations"
        )
        
        # Check if demo files exist
        if not self.demo_rtl_dir.exists():
            print(f"Error: Demo RTL directory not found: {self.demo_rtl_dir}")
            return
            
        # Get list of demo files
        demo_files = sorted(self.demo_rtl_dir.glob("*.sv"))
        if not demo_files:
            print(f"No .sv files found in {self.demo_rtl_dir}")
            return
        
        # Interactive file selection
        selected_file = self._select_file(demo_files)
        if not selected_file:
            return
        
        # Show file preview
        self._preview_file(selected_file)
        
        # Parse and display results
        self._parse_and_display(selected_file)
        
        # Export options
        self._export_results(selected_file)
        
        # Show pragma statistics
        self._show_pragma_statistics(selected_file)
        
    def _select_file(self, files: List[Path]) -> Optional[Path]:
        """Interactive file selection."""
        if RICH_AVAILABLE and questionary:
            # Create choices with descriptions
            choices = []
            for i, file in enumerate(files, 1):
                # Extract description from filename
                name = file.stem
                parts = name.split('_', 2)
                if len(parts) >= 3:
                    desc = parts[2].replace('_', ' ').title()
                else:
                    desc = name
                
                choice = f"{i:02d}. {file.name} - {desc}"
                choices.append(choice)
            
            # Add custom file option
            choices.append("99. Browse for custom file...")
            
            selected = questionary.select(
                "Select an RTL file to explore:",
                choices=choices
            ).ask()
            
            if not selected:
                return None
            
            if "99." in selected:
                # Browse for custom file
                custom_path = questionary.path(
                    "Enter path to SystemVerilog file:",
                    validate=lambda x: x.endswith('.sv') or x.endswith('.v')
                ).ask()
                return Path(custom_path) if custom_path else None
            else:
                # Extract index from selection
                idx = int(selected.split('.')[0]) - 1
                return files[idx]
        else:
            # Fallback to simple selection
            print("\nAvailable RTL files:")
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file.name}")
            
            try:
                choice = int(input("\nSelect file number: "))
                if 1 <= choice <= len(files):
                    return files[choice - 1]
            except ValueError:
                pass
            
            return None
    
    def _preview_file(self, file: Path):
        """Show file preview with pragma highlights."""
        print(f"\n📄 File Preview: {file.name}")
        print("=" * 60)
        
        with open(file, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Find pragma lines and extract types
        pragma_info = []
        pragma_types = set()
        
        for i, line in enumerate(lines, 1):
            if '@brainsmith' in line:
                pragma_info.append(i)
                # Extract pragma type
                parts = line.split()
                for j, part in enumerate(parts):
                    if part == '@brainsmith' and j + 1 < len(parts):
                        pragma_types.add(parts[j + 1])
                        break
        
        # Show file stats
        if RICH_AVAILABLE:
            stats_table = Table(show_header=False, box=None)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            stats_table.add_row("Total Lines", str(len(lines)))
            stats_table.add_row("File Size", f"{len(content):,} bytes")
            stats_table.add_row("Pragma Count", str(len(pragma_info)))
            stats_table.add_row("Pragma Types", ", ".join(sorted(pragma_types)))
            
            self.console.print(Panel(stats_table, title="File Statistics", 
                                   border_style="blue"))
        else:
            print(f"Total Lines: {len(lines)}")
            print(f"File Size: {len(content):,} bytes")
            print(f"Pragma Count: {len(pragma_info)}")
            print(f"Pragma Types: {', '.join(sorted(pragma_types))}")
        
        # Show first 30 lines with highlights
        print("\nFirst 30 lines:")
        print("-" * 60)
        
        preview_lines = lines[:30]
        preview_content = '\n'.join(preview_lines)
        
        # Adjust pragma line numbers for preview
        preview_pragma_lines = [i for i in pragma_info if i <= 30]
        
        highlight_code(preview_content, "systemverilog", 
                      highlight_lines=preview_pragma_lines)
        
        if len(lines) > 30:
            print(f"\n... ({len(lines) - 30} more lines)")
    
    def _parse_and_display(self, file: Path):
        """Parse file and display results."""
        wait_for_input("\nPress Enter to parse the file...")
        
        print("\n🔍 Parsing RTL File")
        print("=" * 60)
        
        # Run parser demo directly
        cmd_str = f"python {self.parser_demo} {file} --format console"
        
        with Timer("RTL Parsing") as timer:
            process = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True
            )
        
        if process.returncode == 0:
            print(f"✅ Parsed successfully in {timer.get_elapsed_ms():.2f}ms")
            
            # Display output (already formatted by rtl_parser_demo.py)
            print("\n" + process.stdout)
        else:
            print(f"❌ Parsing failed: {process.stderr}")
    
    def _export_results(self, file: Path):
        """Export parsing results in multiple formats."""
        if not RICH_AVAILABLE:
            return
        
        export = questionary.confirm(
            "\nWould you like to export the results?"
        ).ask()
        
        if not export:
            return
        
        formats = questionary.checkbox(
            "Select export formats:",
            choices=["JSON", "Markdown", "All"]
        ).ask()
        
        if not formats:
            return
        
        print("\n📤 Exporting Results")
        print("=" * 60)
        
        exported_files = []
        
        # Export each format
        if "JSON" in formats or "All" in formats:
            output_file = self.output_dir / f"{file.stem}_metadata.json"
            cmd = f"python {self.parser_demo} {file} --format json --output {output_file}"
            
            if subprocess.run(cmd, shell=True).returncode == 0:
                exported_files.append(("JSON", output_file))
        
        if "Markdown" in formats or "All" in formats:
            output_file = self.output_dir / f"{file.stem}_metadata.md"
            cmd = f"python {self.parser_demo} {file} --format markdown --output {output_file}"
            
            if subprocess.run(cmd, shell=True).returncode == 0:
                exported_files.append(("Markdown", output_file))
        
        # Show exported files
        if exported_files:
            export_table = Table(title="Exported Files")
            export_table.add_column("Format", style="cyan")
            export_table.add_column("File", style="green")
            export_table.add_column("Size", justify="right")
            
            for fmt, path in exported_files:
                size = f"{path.stat().st_size:,} bytes"
                export_table.add_row(fmt, path.name, size)
            
            self.console.print(export_table)
    
    def _show_pragma_statistics(self, file: Path):
        """Display detailed pragma statistics."""
        print("\n📊 Pragma Analysis Dashboard")
        print("=" * 60)
        
        # Parse file to get pragma stats
        with open(file, 'r') as f:
            content = f.read()
        
        # Analyze pragmas
        pragma_stats = self._analyze_pragmas(content)
        
        if RICH_AVAILABLE:
            # Create visual dashboard
            # Top section: Overview
            overview = Table(title="Pragma Overview", show_header=False)
            overview.add_column("Type", style="cyan")
            overview.add_column("Count", style="yellow", justify="right")
            
            for ptype, count in sorted(pragma_stats['by_type'].items()):
                overview.add_row(ptype, str(count))
            
            # Middle section: Interface coverage
            interface_tree = Tree("Interface Pragmas")
            for iface, pragmas in pragma_stats['by_interface'].items():
                iface_branch = interface_tree.add(f"[bold]{iface}[/bold]")
                for pragma in pragmas:
                    iface_branch.add(f"[dim]{pragma}[/dim]")
            
            # Bottom section: Complexity metrics
            complexity = Panel(
                f"Complexity Score: {pragma_stats['complexity_score']}/10\n"
                f"Coverage: {pragma_stats['coverage']}%\n"
                f"Advanced Features: {', '.join(pragma_stats['advanced_features'])}",
                title="Complexity Analysis",
                border_style="magenta"
            )
            
            # Display dashboard
            self.console.print(Columns([overview, interface_tree]))
            self.console.print(complexity)
        else:
            # Fallback display
            print("\nPragma Types:")
            for ptype, count in sorted(pragma_stats['by_type'].items()):
                print(f"  {ptype}: {count}")
            
            print(f"\nComplexity Score: {pragma_stats['complexity_score']}/10")
            print(f"Coverage: {pragma_stats['coverage']}%")
    
    def _analyze_pragmas(self, content: str) -> Dict[str, Any]:
        """Analyze pragma usage in content."""
        lines = content.splitlines()
        
        stats = {
            'by_type': {},
            'by_interface': {},
            'advanced_features': [],
            'complexity_score': 0,
            'coverage': 0
        }
        
        # Count pragma types
        for line in lines:
            if '@brainsmith' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == '@brainsmith' and i + 1 < len(parts):
                        pragma_type = parts[i + 1]
                        stats['by_type'][pragma_type] = stats['by_type'].get(pragma_type, 0) + 1
                        
                        # Extract interface if present
                        if i + 2 < len(parts) and pragma_type in ['BDIM', 'SDIM', 'DATATYPE']:
                            interface = parts[i + 2]
                            if interface not in stats['by_interface']:
                                stats['by_interface'][interface] = []
                            stats['by_interface'][interface].append(pragma_type)
        
        # Calculate complexity
        if 'RELATIONSHIP' in stats['by_type']:
            stats['advanced_features'].append('Relationships')
            stats['complexity_score'] += 2
        
        if 'DERIVED_PARAMETER' in stats['by_type']:
            stats['advanced_features'].append('Derived Parameters')
            stats['complexity_score'] += 2
        
        if 'ALIAS' in stats['by_type']:
            stats['advanced_features'].append('Aliases')
            stats['complexity_score'] += 1
        
        if len(stats['by_interface']) > 3:
            stats['advanced_features'].append('Multi-Interface')
            stats['complexity_score'] += 2
        
        # Basic complexity from pragma count
        total_pragmas = sum(stats['by_type'].values())
        stats['complexity_score'] += min(3, total_pragmas // 5)
        
        # Coverage estimate
        expected_pragmas = len(stats['by_interface']) * 3  # BDIM, SDIM, DATATYPE per interface
        stats['coverage'] = min(100, (total_pragmas / max(expected_pragmas, 1)) * 100)
        
        return stats
    
    def _create_summary_report(self):
        """Create a summary report of all explorations."""
        # This would aggregate results from multiple file explorations
        pass


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo 2: RTL Parser Interactive Explorer"
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Specific RTL file to explore'
    )
    
    args = parser.parse_args()
    
    explorer = RTLParserExplorer()
    
    if args.file:
        # Direct file mode
        explorer._preview_file(args.file)
        explorer._parse_and_display(args.file)
        explorer._show_pragma_statistics(args.file)
    else:
        # Interactive mode
        explorer.run()


if __name__ == '__main__':
    main()