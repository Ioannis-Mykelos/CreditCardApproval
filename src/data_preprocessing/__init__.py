"""
Data preprocessing module for credit card approval project.

This module contains functions for loading, cleaning, and preprocessing
credit card application data.
"""

from src.data_preprocessing.dataframe_manipulation import (
    load_data,
    handle_nan_values,
    rename_columns,
)
from src.data_preprocessing.dataframe_preprocessing import (
    encoding_the_columns,
    split_data,
    scale_data,
)

__all__ = [
    "load_data",
    "handle_nan_values",
    "rename_columns",
    "encoding_the_columns",
    "split_data",
    "scale_data",
]

