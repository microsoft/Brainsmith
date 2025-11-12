#!/usr/bin/env python3
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Parameter Exploration Demo with ASCII Visualizations.

This demo showcases the parameter exploration capabilities of the two-phase
kernel construction system. It creates a simple LayerNorm model, explores
all valid parallelization parameters, and displays results with ASCII
visualizations.

Features:
- Systematic exploration of all valid configurations
- ASCII visualizations (grids, histograms, tables, heatmaps)
- Performance analysis with simulated metrics
- ANSI color output for better readability

Usage:
    python examples/parameter_exploration_demo.py

Requirements:
    - Brainsmith with LayerNorm kernel
    - ONNX, QONNX

Output:
    - Console output with ASCII visualizations
    - Demonstrates parameter space size and exploration capabilities
"""

import sys
import time

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from brainsmith.dataflow.utils import iter_valid_configurations

# ============================================================================
# ANSI Color Codes
# ============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


# ============================================================================
# Model Creation
# ============================================================================


def create_layernorm_model(hidden_size: int = 64) -> ModelWrapper:
    """Create a simple ONNX model with LayerNorm for demonstration.

    Args:
        hidden_size: Hidden dimension size (default 64 for rich divisor set)
                     64 has divisors: 1, 2, 4, 8, 16, 32, 64

    Returns:
        ModelWrapper containing the LayerNorm model
    """
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, hidden_size])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, hidden_size])

    node = helper.make_node(
        "LayerNorm",
        inputs=["input"],
        outputs=["output"],
        name="layernorm_demo",
        domain="brainsmith.kernels",
        SIMD=1,  # Will be explored
        epsilon=1e-5,
        input0Datatype="FLOAT32",
        output0Datatype="FLOAT32",
    )

    graph = helper.make_graph([node], "layernorm_demo_graph", [input_tensor], [output_tensor])

    model = helper.make_model(graph)
    return ModelWrapper(model)


# ============================================================================
# ASCII Visualization Functions
# ============================================================================


def print_header(title: str, width: int = 80):
    """Print a fancy header banner."""
    c = Colors
    border = "═" * (width - 4)
    print(f"\n{c.BOLD}{c.CYAN}╔═{border}═╗{c.RESET}")
    print(f"{c.BOLD}{c.CYAN}║ {title.center(width - 4)} ║{c.RESET}")
    print(f"{c.BOLD}{c.CYAN}╚═{border}═╝{c.RESET}\n")


def print_section(title: str):
    """Print a section header."""
    c = Colors
    print(f"\n{c.BOLD}{c.YELLOW}▶ {title}{c.RESET}")
    print(f"{c.DIM}{'─' * (len(title) + 2)}{c.RESET}")


def print_distribution_histogram(
    param_name: str, value_counts: dict[int, int], max_width: int = 50
):
    """Print ASCII histogram of parameter value distribution.

    Args:
        param_name: Parameter name (e.g., "SIMD")
        value_counts: Dict mapping parameter values to counts
        max_width: Maximum width of bars in characters
    """
    c = Colors

    if not value_counts:
        print(f"{c.DIM}  No data{c.RESET}")
        return

    max_count = max(value_counts.values())

    print(f"\n{c.BOLD}{param_name} Distribution:{c.RESET}")

    for value in sorted(value_counts.keys()):
        count = value_counts[value]
        bar_width = int((count / max_count) * max_width)
        bar = "▓" * bar_width

        # Color code based on value (higher = greener)
        if value >= max(value_counts.keys()) * 0.7:
            color = c.BRIGHT_GREEN
        elif value >= max(value_counts.keys()) * 0.4:
            color = c.BRIGHT_YELLOW
        else:
            color = c.BRIGHT_RED

        print(f"  {value:4d} {color}{bar}{c.RESET} {count:,}")


def print_parameter_grid(configs: list[dict[str, int]], param1: str, param2: str = None):
    """Print 2D grid visualization of parameter space.

    Args:
        configs: List of configuration dictionaries
        param1: First parameter name
        param2: Second parameter name (if None, shows 1D)
    """
    c = Colors

    if not configs:
        print(f"{c.DIM}  No configurations{c.RESET}")
        return

    # Get unique values for param1
    values1 = sorted(set(cfg[param1] for cfg in configs if param1 in cfg))

    if param2 and any(param2 in cfg for cfg in configs):
        # 2D grid
        values2 = sorted(set(cfg[param2] for cfg in configs if param2 in cfg))

        print(f"\n{c.BOLD}Parameter Space: {param1} × {param2}{c.RESET}")
        print(f"{c.DIM}  {param2:>6s}", end="")
        for v2 in values2:
            print(f" {v2:4d}", end="")
        print(f"{c.RESET}")
        print(f"{c.DIM}  {param1:>6s} ┌{'─' * (5 * len(values2))}┐{c.RESET}")

        for v1 in values1:
            print(f"{c.DIM}  {v1:6d} │{c.RESET}", end="")
            for v2 in values2:
                # Check if this combination exists
                exists = any(cfg.get(param1) == v1 and cfg.get(param2) == v2 for cfg in configs)
                if exists:
                    print(f"{c.GREEN} ●  {c.RESET}", end="")
                else:
                    print(f"{c.DIM} ·  {c.RESET}", end="")
            print(f"{c.DIM}│{c.RESET}")
        print(f"{c.DIM}         └{'─' * (5 * len(values2))}┘{c.RESET}")
    else:
        # 1D visualization
        print(f"\n{c.BOLD}Parameter Space: {param1}{c.RESET}")
        print(f"{c.DIM}  Values: ", end="")
        for v in values1:
            print(f"{c.GREEN}{v}{c.RESET}", end=" ")
        print()


def print_performance_heatmap(
    configs: list[dict[str, int]], scores: list[float], param_name: str = "SIMD"
):
    """Print performance heatmap (simulated performance scores).

    Args:
        configs: List of configurations
        scores: Performance scores for each config
        param_name: Parameter to use for heatmap
    """
    c = Colors

    if not configs or not scores:
        return

    # Group configs by parameter value
    value_scores: dict[int, list[float]] = {}
    for cfg, score in zip(configs, scores):
        value = cfg.get(param_name)
        if value is not None:
            if value not in value_scores:
                value_scores[value] = []
            value_scores[value].append(score)

    # Calculate average scores
    avg_scores = {v: sum(s) / len(s) for v, s in value_scores.items()}

    if not avg_scores:
        return

    print(f"\n{c.BOLD}Performance Heatmap (by {param_name}):{c.RESET}")

    max_score = max(avg_scores.values())
    min_score = min(avg_scores.values())
    score_range = max_score - min_score if max_score > min_score else 1

    for value in sorted(avg_scores.keys()):
        avg_score = avg_scores[value]

        # Normalize to 0-1
        normalized = (avg_score - min_score) / score_range

        # Color code: green (good) to red (bad)
        if normalized > 0.7:
            color = c.BG_GREEN + c.BLACK
            indicator = "██████"
        elif normalized > 0.4:
            color = c.BG_YELLOW + c.BLACK
            indicator = "████  "
        else:
            color = c.BG_RED + c.BLACK
            indicator = "██    "

        print(f"  {param_name}={value:4d}  {color}{indicator}{c.RESET}  " f"Score: {avg_score:.2f}")


def print_table(headers: list[str], rows: list[list], title: str = None):
    """Print ASCII table.

    Args:
        headers: Column headers
        rows: Table rows (list of lists)
        title: Optional table title
    """
    c = Colors

    if not rows:
        print(f"{c.DIM}  No data{c.RESET}")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print title
    if title:
        print(f"\n{c.BOLD}{title}{c.RESET}")

    # Print top border
    print(f"{c.DIM}┌", end="")
    for i, width in enumerate(col_widths):
        print("─" * (width + 2), end="")
        if i < len(col_widths) - 1:
            print("┬", end="")
    print(f"┐{c.RESET}")

    # Print headers
    print(f"{c.DIM}│{c.RESET}", end="")
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        print(f" {c.BOLD}{header.center(width)}{c.RESET} {c.DIM}│{c.RESET}", end="")
    print()

    # Print header separator
    print(f"{c.DIM}├", end="")
    for i, width in enumerate(col_widths):
        print("─" * (width + 2), end="")
        if i < len(col_widths) - 1:
            print("┼", end="")
    print(f"┤{c.RESET}")

    # Print rows
    for row in rows:
        print(f"{c.DIM}│{c.RESET}", end="")
        for cell, width in zip(row, col_widths):
            # Right-align numbers, left-align strings
            cell_str = str(cell)
            if isinstance(cell, int | float):
                formatted = cell_str.rjust(width)
            else:
                formatted = cell_str.ljust(width)
            print(f" {formatted} {c.DIM}│{c.RESET}", end="")
        print()

    # Print bottom border
    print(f"{c.DIM}└", end="")
    for i, width in enumerate(col_widths):
        print("─" * (width + 2), end="")
        if i < len(col_widths) - 1:
            print("┴", end="")
    print(f"┘{c.RESET}")


def print_progress_bar(current: int, total: int, width: int = 50):
    """Print a progress bar.

    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar in characters
    """
    c = Colors

    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)

    print(f"\r  {c.CYAN}[{bar}]{c.RESET} {current}/{total} ({percent*100:.1f}%)", end="")
    sys.stdout.flush()


# ============================================================================
# Main Demo
# ============================================================================


def simulate_performance_score(config: dict[str, int]) -> float:
    """Simulate a performance score for demonstration.

    Higher SIMD generally gives better performance (lower latency),
    but this is a simplified simulation.

    Args:
        config: Configuration dict

    Returns:
        Simulated performance score (higher is better)
    """
    simd = config.get("SIMD", 1)
    # Simulate: higher parallelism = better performance, with diminishing returns
    return 100.0 * (1.0 - (1.0 / (simd**0.5)))


def main():
    """Run parameter exploration demo."""
    c = Colors

    # Print header
    print_header("Parameter Exploration Demo - LayerNorm", width=80)

    print(f"{c.DIM}This demo explores all valid parallelization parameters for a")
    print(f"LayerNorm kernel and displays results with ASCII visualizations.{c.RESET}")

    # Step 1: Create model
    print_section("Step 1: Creating LayerNorm Model")

    hidden_size = 64  # Use 64 for rich divisor set (1,2,4,8,16,32,64)
    print(f"  Hidden size: {c.BOLD}{hidden_size}{c.RESET}")
    print(f"  {c.DIM}(64 chosen for rich divisor set: 1, 2, 4, 8, 16, 32, 64){c.RESET}")

    model_w = create_layernorm_model(hidden_size)
    node = model_w.graph.node[0]
    kernel_op = getCustomOp(node)

    print(f"  Kernel type: {c.BOLD}{node.op_type}{c.RESET}")
    print(f"  Node name: {c.BOLD}{node.name}{c.RESET}")
    print(f"  {c.GREEN}✓{c.RESET} Model created successfully")

    # Step 2: Get valid ranges
    print_section("Step 2: Computing Valid Parameter Ranges")

    valid_ranges = kernel_op.get_valid_ranges(model_w)

    if not valid_ranges:
        print(f"  {c.YELLOW}⚠{c.RESET} No parallelization parameters found")
        return

    print(f"  Parameters found: {c.BOLD}{', '.join(valid_ranges.keys())}{c.RESET}")

    for param_name, param_values in valid_ranges.items():
        print(
            f"    {param_name}: {len(param_values)} values "
            f"(range: {min(param_values)} - {max(param_values)})"
        )

    # Calculate total configs
    total_configs = 1
    for param_values in valid_ranges.values():
        total_configs *= len(param_values)

    print(f"  Total configurations: {c.BOLD}{c.BRIGHT_CYAN}{total_configs:,}{c.RESET}")

    # Step 3: Explore configurations
    print_section("Step 3: Exploring Configurations")

    configs = []
    scores = []

    print(f"\n  {c.DIM}Exploring {total_configs:,} configurations...{c.RESET}\n")

    start_time = time.time()

    for i, config in enumerate(iter_valid_configurations(kernel_op, model_w)):
        configs.append(config)

        # Simulate performance score
        score = simulate_performance_score(config)
        scores.append(score)

        # Update progress bar every 5 configs or at the end
        if (i + 1) % 5 == 0 or (i + 1) == total_configs:
            print_progress_bar(i + 1, total_configs)

    elapsed = time.time() - start_time

    print(
        f"\n\n  {c.GREEN}✓{c.RESET} Explored {len(configs):,} configurations in "
        f"{c.BOLD}{elapsed:.3f}s{c.RESET}"
    )
    print(f"  Average time per config: {c.BOLD}{elapsed*1000/len(configs):.2f}ms{c.RESET}")

    # Step 4: Visualizations
    print_section("Step 4: Parameter Space Visualization")

    # Distribution histogram
    if "SIMD" in valid_ranges:
        simd_counts = {}
        for cfg in configs:
            simd = cfg.get("SIMD")
            simd_counts[simd] = simd_counts.get(simd, 0) + 1

        print_distribution_histogram("SIMD", simd_counts)

    # Parameter grid
    param_names = list(valid_ranges.keys())
    if len(param_names) >= 2:
        print_parameter_grid(configs, param_names[0], param_names[1])
    elif len(param_names) == 1:
        print_parameter_grid(configs, param_names[0])

    # Performance heatmap
    if scores:
        print_performance_heatmap(configs, scores, param_names[0] if param_names else "SIMD")

    # Step 5: Top configurations
    print_section("Step 5: Top Configurations")

    # Sort by score (descending)
    sorted_configs = sorted(zip(configs, scores), key=lambda x: x[1], reverse=True)

    # Build table data
    top_n = min(10, len(sorted_configs))
    headers = ["Rank"] + list(valid_ranges.keys()) + ["Score"]
    rows = []

    for i, (cfg, score) in enumerate(sorted_configs[:top_n]):
        row = [i + 1] + [cfg[param] for param in valid_ranges.keys()] + [f"{score:.2f}"]
        rows.append(row)

    print_table(headers, rows, title=f"Top {top_n} Configurations (by simulated performance)")

    # Step 6: Summary
    print_section("Summary")

    print(f"  Total configurations explored: {c.BOLD}{len(configs):,}{c.RESET}")
    print(f"  Exploration time: {c.BOLD}{elapsed:.3f}s{c.RESET}")
    print(f"  Configurations per second: {c.BOLD}{len(configs)/elapsed:.1f}{c.RESET}")

    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        print("  Performance scores:")
        print(f"    Best:    {c.BRIGHT_GREEN}{max_score:.2f}{c.RESET}")
        print(f"    Average: {c.BRIGHT_YELLOW}{avg_score:.2f}{c.RESET}")
        print(f"    Worst:   {c.BRIGHT_RED}{min_score:.2f}{c.RESET}")

    print(f"\n{c.BOLD}{c.GREEN}Demo completed successfully!{c.RESET}\n")

    print(f"{c.DIM}This demo used the two-phase kernel construction system to:")
    print("  • Systematically explore all valid configurations")
    print("  • Avoid invalid configuration attempts")
    print("  • Provide rich visualizations of the design space")
    print("\nFor more information, see:")
    print("  • docs/two_phase_architecture.md - System architecture")
    print(f"  • docs/two_phase_user_guide.md - Usage guide{c.RESET}\n")


if __name__ == "__main__":
    main()
