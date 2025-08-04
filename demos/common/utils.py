############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Shared utilities for demos."""

import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")


class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed else 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def load_rtl_file(file_path: Path) -> Dict[str, Any]:
    """Load RTL file and return metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"RTL file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Count pragmas
    pragma_count = content.count('@brainsmith')
    
    # Extract module name
    import re
    module_match = re.search(r'module\s+(\w+)', content)
    module_name = module_match.group(1) if module_match else "unknown"
    
    return {
        'path': file_path,
        'name': file_path.name,
        'size': file_path.stat().st_size,
        'content': content,
        'lines': len(content.splitlines()),
        'pragma_count': pragma_count,
        'module_name': module_name
    }


def create_demo_header(title: str, subtitle: str = "") -> None:
    """Create a formatted demo header."""
    if RICH_AVAILABLE:
        console = Console()
        
        header_text = Text(title, style="bold cyan")
        if subtitle:
            header_text.append("\n" + subtitle, style="dim")
        
        panel = Panel(
            header_text,
            expand=True,
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(panel)
    else:
        print("=" * 80)
        print(f"{title:^80}")
        if subtitle:
            print(f"{subtitle:^80}")
        print("=" * 80)
    print()


def create_progress_bar(description: str = "Processing") -> Optional[Any]:
    """Create a progress bar context manager."""
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=Console(),
            transient=True
        )
    return None


def highlight_code(code: str, language: str = "systemverilog", 
                  line_numbers: bool = True, highlight_lines: Optional[List[int]] = None,
                  start_line: int = 1) -> None:
    """Display syntax-highlighted code."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Convert to appropriate language name for rich
        lang_map = {
            'systemverilog': 'verilog',
            'sv': 'verilog',
            'py': 'python'
        }
        language = lang_map.get(language.lower(), language)
        
        syntax = Syntax(
            code,
            language,
            line_numbers=line_numbers,
            highlight_lines=highlight_lines or [],
            start_line=start_line
        )
        console.print(syntax)
    else:
        # Fallback to simple print
        lines = code.splitlines()
        for i, line in enumerate(lines, start_line):
            if highlight_lines and i in highlight_lines:
                print(f">>> {i:4d}: {line}")
            else:
                print(f"    {i:4d}: {line}")


def save_demo_output(content: str, filename: str, output_dir: Path = Path("demo_outputs")) -> Path:
    """Save demo output to file."""
    output_dir.mkdir(exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{timestamp}_{filename}"
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    return output_file


def create_comparison_table(before_data: Dict[str, Any], after_data: Dict[str, Any], 
                           title: str = "Before/After Comparison") -> None:
    """Create a comparison table showing before/after metrics."""
    if RICH_AVAILABLE:
        console = Console()
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Before", style="red")
        table.add_column("After", style="green")
        table.add_column("Improvement", justify="right", style="bold yellow")
        
        # Calculate improvements
        for key in before_data:
            if key in after_data:
                before_val = before_data[key]
                after_val = after_data[key]
                
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    if before_val > 0:
                        improvement = ((before_val - after_val) / before_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                else:
                    improvement_str = "-"
                
                table.add_row(
                    key,
                    str(before_val),
                    str(after_val),
                    improvement_str
                )
        
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 60)
        print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Improvement':>10}")
        print("-" * 60)
        
        for key in before_data:
            if key in after_data:
                before_val = before_data[key]
                after_val = after_data[key]
                
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    if before_val > 0:
                        improvement = ((before_val - after_val) / before_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        improvement_str = "N/A"
                else:
                    improvement_str = "-"
                
                print(f"{key:<20} {str(before_val):>15} {str(after_val):>15} {improvement_str:>10}")


def wait_for_input(prompt: str = "Press Enter to continue...") -> None:
    """Wait for user input with a prompt."""
    import sys
    
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        # Non-interactive mode, skip the wait
        return
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[dim]{prompt}[/dim]")
    else:
        print(f"\n{prompt}")
    
    try:
        input()
    except EOFError:
        # Handle non-interactive environments gracefully
        pass


def run_command_with_output(command: str, description: str = "Running command") -> tuple[bool, str]:
    """Run a shell command and capture output."""
    import subprocess
    
    if RICH_AVAILABLE:
        console = Console()
        with console.status(f"[bold green]{description}..."):
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True, result.stdout
            except subprocess.CalledProcessError as e:
                return False, e.stderr
    else:
        print(f"{description}...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr