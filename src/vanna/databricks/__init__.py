"""
Databricks integration for Vanna AI

This module provides integration with Databricks SQL warehouses and compute clusters.
"""

try:
  from .databricks_sql import Databricks_SQL, Databricks_Compute
  __all__ = ['Databricks_SQL', 'Databricks_Compute']
except ImportError as e:
  import warnings
  warnings.warn(f"Databricks integration requires additional dependencies. "
                f"Install with: pip install 'vanna[databricks]'. Error: {e}")
  __all__ = []
