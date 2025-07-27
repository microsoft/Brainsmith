# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Time parsing utilities for blueprint configuration.
"""

from typing import Union


def parse_time_with_units(value: Union[str, float, int]) -> float:
    """
    Parse time value with unit suffix to nanoseconds.
    
    Supports: ns, us, ms, ps
    Default unit is ns if no suffix.
    
    Examples:
        "5" -> 5.0
        "5ns" -> 5.0
        "5000ps" -> 5.0
        "0.005us" -> 5.0
        "0.000005ms" -> 5.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).strip()
    
    # Unit conversion map to nanoseconds
    units = {
        'ps': 0.001,
        'ns': 1.0,
        'us': 1000.0,
        'ms': 1000000.0
    }
    
    # Check for unit suffix
    for unit, multiplier in units.items():
        if value_str.endswith(unit):
            numeric_part = value_str[:-len(unit)].strip()
            return float(numeric_part) * multiplier
    
    # No unit suffix, assume ns
    return float(value_str)