"""DSE integration tests organized by execution time.

Tiered test execution:
- fast/ - Pure Python, no FINN execution (< 1 min)
- finn/ - Real FINN builds with minimal models (1-30 min)
- rtl/ - RTL simulation tests (30min - hours)
- hardware/ - Bitfile generation and HW validation (hours - days)
"""
