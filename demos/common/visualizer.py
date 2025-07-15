############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Visualization components for demos."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
    from matplotlib.lines import Line2D
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Install with: pip install networkx")


def create_interface_diagram(interfaces: List[Dict[str, Any]], 
                            title: str = "Interface Diagram",
                            output_path: Optional[Path] = None) -> Optional[Path]:
    """Create a visual diagram of kernel interfaces."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Color scheme for different interface types
    colors = {
        'input': '#3498db',      # Blue
        'output': '#2ecc71',     # Green  
        'weight': '#e74c3c',     # Red
        'config': '#f39c12',     # Orange
        'status': '#9b59b6'      # Purple
    }
    
    # Calculate layout
    n_interfaces = len(interfaces)
    if n_interfaces == 0:
        ax.text(0.5, 0.5, "No interfaces", ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Arrange interfaces in a circular layout
        center_x, center_y = 0.5, 0.5
        radius = 0.3
        
        # Draw kernel box in center
        kernel_box = FancyBboxPatch(
            (center_x - 0.1, center_y - 0.1), 0.2, 0.2,
            boxstyle="round,pad=0.02",
            facecolor='#ecf0f1',
            edgecolor='#34495e',
            linewidth=2
        )
        ax.add_patch(kernel_box)
        ax.text(center_x, center_y, "Kernel", ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Draw interfaces around the kernel
        for i, interface in enumerate(interfaces):
            angle = 2 * np.pi * i / n_interfaces
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Get interface properties
            iface_name = interface.get('name', f'Interface{i}')
            iface_type = interface.get('type', 'unknown')
            iface_width = interface.get('width', 32)
            
            # Draw interface box
            color = colors.get(iface_type, '#95a5a6')
            iface_box = FancyBboxPatch(
                (x - 0.08, y - 0.04), 0.16, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor='#2c3e50',
                alpha=0.8
            )
            ax.add_patch(iface_box)
            
            # Add interface label
            ax.text(x, y, iface_name, ha='center', va='center',
                   fontsize=10, color='white', fontweight='bold')
            
            # Add width annotation
            ax.text(x, y - 0.06, f"{iface_width} bits", ha='center', va='center',
                   fontsize=8, color='#7f8c8d')
            
            # Draw connection line
            line = Line2D([center_x, x], [center_y, y], 
                         color=color, alpha=0.5, linewidth=2)
            ax.add_line(line)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['input'], 
               markersize=10, label='Input'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['output'],
               markersize=10, label='Output'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['weight'],
               markersize=10, label='Weight'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['config'],
               markersize=10, label='Config')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.tight_layout()
    return output_path




def create_comparison_chart(metrics: Dict[str, Tuple[float, float]],
                           title: str = "Performance Comparison",
                           output_path: Optional[Path] = None) -> Optional[Path]:
    """Create a bar chart comparing before/after metrics."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Prepare data
    labels = list(metrics.keys())
    before_values = [metrics[k][0] for k in labels]
    after_values = [metrics[k][1] for k in labels]
    
    # Create bar positions
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, before_values, width, label='Before', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='After', color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Add improvement percentages
    for i, (before, after) in enumerate(zip(before_values, after_values)):
        if before > 0:
            improvement = ((before - after) / before) * 100
            y_pos = max(before, after) + 5
            ax.text(i, y_pos, f"{improvement:.0f}%", ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#27ae60')
    
    # Customize chart
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.tight_layout()
    return output_path


def create_pragma_visualization(pragmas: List[Dict[str, Any]],
                               code_snippet: str,
                               title: str = "Pragma Effects",
                               output_path: Optional[Path] = None) -> Optional[Path]:
    """Visualize pragma annotations on code."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Left side: Code with pragma highlights
    ax1.set_title("SystemVerilog with Pragmas", fontsize=12)
    ax1.axis('off')
    
    # Display code with line numbers
    lines = code_snippet.splitlines()
    y_start = 0.95
    line_height = 0.02
    
    for i, line in enumerate(lines[:40]):  # Limit to 40 lines
        y_pos = y_start - i * line_height
        
        # Check if this line has a pragma
        is_pragma = '@brainsmith' in line
        
        if is_pragma:
            # Highlight pragma lines
            rect = Rectangle((0.02, y_pos - line_height/2), 0.96, line_height,
                           facecolor='#fff3cd', edgecolor='none', alpha=0.5)
            ax1.add_patch(rect)
            color = '#856404'
            weight = 'bold'
        else:
            color = '#212529'
            weight = 'normal'
        
        # Add line number and text
        ax1.text(0.03, y_pos, f"{i+1:3d}", fontsize=8, color='#6c757d',
                fontfamily='monospace', va='center')
        ax1.text(0.08, y_pos, line[:80], fontsize=8, color=color,
                fontfamily='monospace', va='center', weight=weight)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Right side: Pragma summary
    ax2.set_title("Extracted Metadata", fontsize=12)
    ax2.axis('off')
    
    # Group pragmas by type
    pragma_groups = {}
    for pragma in pragmas:
        ptype = pragma.get('type', 'unknown')
        if ptype not in pragma_groups:
            pragma_groups[ptype] = []
        pragma_groups[ptype].append(pragma)
    
    y_pos = 0.95
    for ptype, plist in pragma_groups.items():
        # Type header
        ax2.text(0.05, y_pos, f"{ptype} ({len(plist)})", fontsize=10,
                fontweight='bold', color='#0066cc')
        y_pos -= 0.03
        
        # List pragmas
        for pragma in plist[:5]:  # Limit display
            details = pragma.get('details', str(pragma))
            ax2.text(0.1, y_pos, f"• {details}", fontsize=8, color='#495057')
            y_pos -= 0.025
        
        if len(plist) > 5:
            ax2.text(0.1, y_pos, f"... and {len(plist)-5} more", fontsize=8,
                    style='italic', color='#6c757d')
            y_pos -= 0.025
        
        y_pos -= 0.02
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.tight_layout()
    return output_path


def export_visualization(fig: Any, output_path: Path, formats: List[str] = ['png']) -> List[Path]:
    """Export visualization in multiple formats."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    
    output_files = []
    
    for fmt in formats:
        output_file = output_path.with_suffix(f'.{fmt}')
        
        if fmt in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            plt.savefig(output_file, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white')
            output_files.append(output_file)
        elif fmt == 'json':
            # For data export
            # Note: This would need the actual data, not the matplotlib figure
            pass
    
    return output_files


