"""
BrainSmith Analysis Hooks

Provides hooks for external analysis tools instead of custom implementations.
Users can integrate pandas, scipy, scikit-learn, or any other analysis library.

Key Philosophy:
- Expose structured data for external tools
- Zero maintenance burden for analysis algorithms  
- User choice in analysis libraries
- Better functionality through mature external libraries

Example Usage:
    from brainsmith.analysis import expose_analysis_data
    
    # Get structured data from BrainSmith
    results = brainsmith.forge(model, blueprint)
    data = expose_analysis_data(results['dse_results'])
    
    # Use pandas for analysis
    import pandas as pd
    df = pd.DataFrame(data['solutions'])
    summary = df.describe()
    
    # Use scipy for statistics
    import scipy.stats as stats
    normality = stats.normaltest(df['objective_0'])
    
    # Use scikit-learn for ML
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df[['objective_0', 'objective_1']])
"""

from .hooks import (
    expose_analysis_data,
    register_analyzer,
    get_registered_analyzers,
    get_raw_data,
    export_to_dataframe
)

from .adapters import (
    pandas_adapter,
    scipy_adapter,
    sklearn_adapter
)

from .utils import (
    calculate_basic_statistics,
    extract_metric_arrays,
    format_for_external_tool
)

__version__ = "0.1.0"
__author__ = "BrainSmith Development Team"

# Export all hook functions
__all__ = [
    # Core hooks
    'expose_analysis_data',
    'register_analyzer',
    'get_registered_analyzers',
    'get_raw_data',
    'export_to_dataframe',
    
    # External tool adapters
    'pandas_adapter',
    'scipy_adapter',
    'sklearn_adapter',
    
    # Utilities
    'calculate_basic_statistics',
    'extract_metric_arrays',
    'format_for_external_tool'
]

# Initialize logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Analysis Hooks v{__version__} initialized")
logger.info("Hooks available for: pandas, scipy, scikit-learn, and custom analysis tools")