"""
Data Export Functions - North Star Aligned

Simple functions for exporting data to external analysis tools.
Consolidates functionality from analysis and metrics modules while eliminating enterprise patterns.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

from .types import BuildMetrics, DataSummary, ComparisonResult, MetricsList

logger = logging.getLogger(__name__)


def export_for_analysis(
    data: Union[BuildMetrics, MetricsList, DataSummary, ComparisonResult],
    format: str = 'pandas',
    filepath: Optional[str] = None
) -> Any:
    """
    Unified export function for external analysis tools.
    
    Consolidates analysis.expose_analysis_data() and metrics.export_metrics()
    functionality into a single, simple interface.
    
    Args:
        data: Data to export (BuildMetrics, list, summary, or comparison)
        format: Export format ('pandas', 'csv', 'json', 'dict', 'scipy')
        filepath: Optional file path for saving
        
    Returns:
        Exported data in requested format
        
    Example:
        # Export to pandas for analysis
        df = export_for_analysis(metrics_list, 'pandas')
        df.plot(x='pe_count', y='throughput_ops_sec', kind='scatter')
        
        # Export to CSV for Excel
        export_for_analysis(summary, 'csv', 'results.csv')
        
        # Export to JSON for web tools
        json_data = export_for_analysis(metrics, 'json')
    """
    if format == 'pandas':
        return to_pandas(data, filepath)
    elif format == 'csv':
        return to_csv(data, filepath)
    elif format == 'json':
        return to_json(data, filepath)
    elif format == 'dict':
        return _to_dict(data)
    elif format == 'scipy':
        return _to_scipy(data)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def to_pandas(
    data: Union[BuildMetrics, MetricsList, DataSummary],
    filepath: Optional[str] = None
):
    """
    Export data to pandas DataFrame for analysis.
    
    Consolidates analysis.pandas_adapter() and metrics.export_to_pandas() 
    with enhanced data flattening.
    
    Args:
        data: Data to export
        filepath: Optional CSV file path for saving DataFrame
        
    Returns:
        pandas.DataFrame or None if pandas not available
        
    Example:
        df = to_pandas(metrics_list)
        summary = df.describe()
        correlation = df.corr()
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not available - cannot export to DataFrame")
        return None
    
    # Convert to list of dictionaries
    data_dicts = _prepare_data_for_export(data)
    
    if not data_dicts:
        logger.warning("No data to export")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data_dicts)
    
    # Flatten nested columns if needed
    df = _flatten_dataframe_columns(df)
    
    # Save to CSV if filepath provided
    if filepath:
        df.to_csv(filepath, index=False)
        logger.info(f"DataFrame exported to {filepath}")
    
    logger.info(f"Created pandas DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


def to_csv(
    data: Union[BuildMetrics, MetricsList, DataSummary],
    filepath: Optional[str] = None
) -> str:
    """
    Export data to CSV format.
    
    Args:
        data: Data to export
        filepath: Optional file path for saving
        
    Returns:
        CSV string data
    """
    import csv
    import io
    
    # Convert to list of dictionaries
    data_dicts = _prepare_data_for_export(data)
    
    if not data_dicts:
        logger.warning("No data to export")
        return ""
    
    # Flatten dictionaries for CSV
    flattened_data = []
    for data_dict in data_dicts:
        flattened = _flatten_dict(data_dict)
        flattened_data.append(flattened)
    
    # Get all possible columns
    all_columns = set()
    for data in flattened_data:
        all_columns.update(data.keys())
    all_columns = sorted(all_columns)
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_columns)
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for data in flattened_data:
        # Fill missing columns with empty strings
        row = {col: data.get(col, '') for col in all_columns}
        writer.writerow(row)
    
    csv_content = output.getvalue()
    output.close()
    
    # Save to file if filepath provided
    if filepath:
        with open(filepath, 'w') as f:
            f.write(csv_content)
        logger.info(f"CSV exported to {filepath}")
    
    logger.info(f"Generated CSV: {len(flattened_data)} rows, {len(all_columns)} columns")
    return csv_content


def to_json(
    data: Union[BuildMetrics, MetricsList, DataSummary, ComparisonResult],
    filepath: Optional[str] = None
) -> str:
    """
    Export data to JSON format.
    
    Args:
        data: Data to export
        filepath: Optional file path for saving
        
    Returns:
        JSON string data
    """
    # Convert to list of dictionaries
    data_dicts = _prepare_data_for_export(data)
    
    # Create JSON structure
    json_data = {
        'export_timestamp': _get_current_timestamp(),
        'data_count': len(data_dicts) if isinstance(data_dicts, list) else 1,
        'data': data_dicts
    }
    
    # Convert to JSON string
    json_string = json.dumps(json_data, indent=2, default=str)
    
    # Save to file if filepath provided
    if filepath:
        with open(filepath, 'w') as f:
            f.write(json_string)
        logger.info(f"JSON exported to {filepath}")
    
    logger.info(f"Generated JSON: {len(data_dicts) if isinstance(data_dicts, list) else 1} records")
    return json_string


def create_report(
    data: Union[BuildMetrics, MetricsList, DataSummary],
    format: str = 'markdown',
    filepath: Optional[str] = None
) -> str:
    """
    Generate a data report for documentation.
    
    Args:
        data: Data to report
        format: Report format ('markdown', 'html', 'text')
        filepath: Optional file path for saving
        
    Returns:
        Report string in requested format
    """
    if format == 'markdown':
        report = _generate_markdown_report(data)
    elif format == 'html':
        report = _generate_html_report(data)
    elif format == 'text':
        report = _generate_text_report(data)
    else:
        raise ValueError(f"Unsupported report format: {format}")
    
    # Save to file if filepath provided
    if filepath:
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Report exported to {filepath}")
    
    return report


# Private helper functions

def _prepare_data_for_export(data: Any) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Prepare data for export by converting to dictionaries."""
    if isinstance(data, BuildMetrics):
        return [data.to_dict()]
    elif isinstance(data, (DataSummary, ComparisonResult)):
        return [data.to_dict()]
    elif isinstance(data, list):
        return [item.to_dict() if hasattr(item, 'to_dict') else item for item in data]
    else:
        logger.warning(f"Unknown data type: {type(data)}")
        return []


def _to_dict(data: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Export data to dictionary format."""
    if isinstance(data, BuildMetrics):
        return [data.to_dict()]  # Always return list for consistency
    elif isinstance(data, (DataSummary, ComparisonResult)):
        return [data.to_dict()]  # Always return list for consistency
    elif isinstance(data, list):
        return [item.to_dict() if hasattr(item, 'to_dict') else item for item in data]
    else:
        return []


def _to_scipy(data: MetricsList) -> Dict[str, Any]:
    """
    Export data in format optimized for scipy statistical analysis.
    
    Consolidates analysis.scipy_adapter() and metrics.export_for_scipy().
    
    Returns:
        Dictionary with numpy arrays for statistical functions
        
    Example:
        scipy_data = export_for_analysis(metrics_list, 'scipy')
        correlation = scipy.stats.pearsonr(scipy_data['throughput'], scipy_data['lut_utilization'])
    """
    try:
        import numpy as np
        
        if not isinstance(data, list):
            data = [data] if hasattr(data, 'to_dict') else []
        
        successful = [m for m in data if hasattr(m, 'is_successful') and m.is_successful()]
        
        scipy_data = {
            'throughput': np.array([m.performance.throughput_ops_sec for m in successful 
                                   if m.performance.throughput_ops_sec is not None]),
            'latency': np.array([m.performance.latency_ms for m in successful 
                                if m.performance.latency_ms is not None]),
            'lut_utilization': np.array([m.resources.lut_utilization_percent for m in successful 
                                        if m.resources.lut_utilization_percent is not None]),
            'dsp_utilization': np.array([m.resources.dsp_utilization_percent for m in successful 
                                        if m.resources.dsp_utilization_percent is not None]),
            'efficiency_scores': np.array([m.get_efficiency_score() for m in successful 
                                          if m.get_efficiency_score() is not None])
        }
        
        # Add metadata
        scipy_data['metadata'] = {
            'sample_size': len(successful),
            'total_samples': len(data),
            'metric_names': list(scipy_data.keys())
        }
        
        return scipy_data
        
    except ImportError:
        logger.warning("NumPy not available - returning plain lists")
        return _to_matplotlib(data)


def _to_matplotlib(data: MetricsList) -> Dict[str, List[float]]:
    """
    Export data in format optimized for matplotlib plotting.
    
    Returns:
        Dictionary with lists of values for direct plotting
        
    Example:
        plot_data = export_for_analysis(metrics_list, 'matplotlib')
        plt.scatter(plot_data['throughput'], plot_data['lut_utilization'])
    """
    if not isinstance(data, list):
        data = [data] if hasattr(data, 'to_dict') else []
    
    plot_data = {
        'throughput': [],
        'latency': [],
        'lut_utilization': [],
        'dsp_utilization': [],
        'accuracy': [],
        'build_time': []
    }
    
    for m in data:
        if hasattr(m, 'is_successful') and m.is_successful():
            plot_data['throughput'].append(m.performance.throughput_ops_sec or 0)
            plot_data['latency'].append(m.performance.latency_ms or 0)
            plot_data['lut_utilization'].append(m.resources.lut_utilization_percent or 0)
            plot_data['dsp_utilization'].append(m.resources.dsp_utilization_percent or 0)
            plot_data['accuracy'].append(m.quality.accuracy_percent or 0)
            plot_data['build_time'].append(m.build.build_time_seconds)
    
    return plot_data


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _flatten_dataframe_columns(df):
    """Flatten DataFrame columns that contain dictionaries."""
    try:
        import pandas as pd
        
        # Check for columns that contain dictionaries
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if any values are dictionaries
                sample_vals = df[col].dropna().head()
                if len(sample_vals) > 0 and isinstance(sample_vals.iloc[0], dict):
                    # Expand dictionary columns
                    expanded = pd.json_normalize(df[col])
                    expanded.columns = [f"{col}_{subcol}" for subcol in expanded.columns]
                    
                    # Drop original column and add expanded columns
                    df = df.drop(columns=[col])
                    for expanded_col in expanded.columns:
                        df[expanded_col] = expanded[expanded_col]
        
        return df
        
    except ImportError:
        return df


def _generate_markdown_report(data: Any) -> str:
    """Generate markdown format report."""
    lines = []
    lines.append("# FPGA Data Report")
    lines.append("")
    lines.append(f"Generated at: {_get_current_timestamp()}")
    lines.append("")
    
    if isinstance(data, list):
        lines.append(f"## Summary")
        lines.append(f"- Total configurations: {len(data)}")
        
        successful = [m for m in data if hasattr(m, 'is_successful') and m.is_successful()]
        lines.append(f"- Successful builds: {len(successful)}")
        lines.append(f"- Success rate: {len(successful)/len(data)*100:.1f}%")
        lines.append("")
        
        if successful:
            # Performance summary
            throughputs = [m.performance.throughput_ops_sec for m in successful 
                          if m.performance.throughput_ops_sec is not None]
            if throughputs:
                lines.append(f"## Performance Summary")
                lines.append(f"- Average throughput: {sum(throughputs)/len(throughputs):.1f} ops/sec")
                lines.append(f"- Best throughput: {max(throughputs):.1f} ops/sec")
                lines.append(f"- Worst throughput: {min(throughputs):.1f} ops/sec")
                lines.append("")
            
            # Resource summary
            lut_utils = [m.resources.lut_utilization_percent for m in successful 
                        if m.resources.lut_utilization_percent is not None]
            if lut_utils:
                lines.append(f"## Resource Summary")
                lines.append(f"- Average LUT utilization: {sum(lut_utils)/len(lut_utils):.1f}%")
                lines.append(f"- Peak LUT utilization: {max(lut_utils):.1f}%")
                lines.append("")
    
    elif isinstance(data, BuildMetrics):
        lines.append(f"## Single Configuration Report")
        lines.append(f"- Model: {data.model_path or 'N/A'}")
        lines.append(f"- Blueprint: {data.blueprint_path or 'N/A'}")
        lines.append(f"- Build success: {data.build.build_success}")
        lines.append("")
        
        if data.build.build_success:
            lines.append(f"### Performance")
            lines.append(f"- Throughput: {data.performance.throughput_ops_sec or 'N/A'} ops/sec")
            lines.append(f"- Latency: {data.performance.latency_ms or 'N/A'} ms")
            lines.append("")
            
            lines.append(f"### Resources")
            lines.append(f"- LUT utilization: {data.resources.lut_utilization_percent or 'N/A'}%")
            lines.append(f"- DSP utilization: {data.resources.dsp_utilization_percent or 'N/A'}%")
            lines.append("")
    
    elif isinstance(data, DataSummary):
        lines.append(f"## Summary Statistics")
        lines.append(f"- Total metrics: {data.metric_count}")
        lines.append(f"- Success rate: {data.success_rate*100:.1f}%")
        lines.append(f"- Average throughput: {data.avg_throughput or 'N/A'}")
        lines.append(f"- Average LUT utilization: {data.avg_lut_utilization or 'N/A'}%")
        lines.append("")
    
    return "\n".join(lines)


def _generate_html_report(data: Any) -> str:
    """Generate HTML format report."""
    # Convert markdown to basic HTML
    markdown_report = _generate_markdown_report(data)
    
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html><head><title>FPGA Data Report</title></head><body>")
    
    # Simple markdown to HTML conversion
    for line in markdown_report.split('\n'):
        if line.startswith('# '):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith('## '):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith('### '):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith('- '):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            html_lines.append(f"<p>{line}</p>")
        else:
            html_lines.append("<br>")
    
    html_lines.append("</body></html>")
    
    return "\n".join(html_lines)


def _generate_text_report(data: Any) -> str:
    """Generate plain text format report."""
    # Convert markdown to plain text
    markdown_report = _generate_markdown_report(data)
    
    text_lines = []
    for line in markdown_report.split('\n'):
        if line.startswith('#'):
            # Remove markdown headers
            text_lines.append(line.lstrip('#').strip())
            text_lines.append("=" * len(line.lstrip('#').strip()))
        else:
            text_lines.append(line)
    
    return "\n".join(text_lines)


def _get_current_timestamp() -> str:
    """Get current timestamp as string."""
    import datetime
    return datetime.datetime.now().isoformat()