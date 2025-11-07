"""
Tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
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


def test_load_data():
    """Test that load_data returns a DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_handle_nan_values():
    """Test that handle_nan_values removes NaN values."""
    df = pd.DataFrame({"col1": [1, 2, "?", 4], "col2": [1.0, 2.0, np.nan, 4.0]})
    result = handle_nan_values(df)
    assert result.isna().sum().sum() == 0


def test_rename_columns():
    """Test that rename_columns renames columns correctly."""
    df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})
    result = rename_columns(df)
    assert "col1" in result.columns or "col2" in result.columns


def test_encoding_the_columns():
    """Test that encoding_the_columns encodes categorical variables."""
    df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3]})
    result = encoding_the_columns(df)
    assert result["col1"].dtype != "object"


def test_split_data():
    """Test that split_data returns correct number of splits."""
    # Create a dataframe with 16 columns total
    # After dropping col11 and col13, we'll have 14 columns
    # split_data uses columns 0-12 (13 cols) for X and column 13 for y
    data = {}
    for i in range(1, 17):  # Create col1 through col16
        data[f"col{i}"] = [1, 2, 3, 4, 5] * 10
    df = pd.DataFrame(data)
    
    X, y, X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert X_train.shape[1] == 13  # Should have 13 features after dropping col11 and col13
    assert len(y_train) == len(X_train)  # y should match X length
    assert len(y_test) == len(X_test)  # y_test should match X_test length


def test_scale_data():
    """Test that scale_data scales data correctly."""
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[2, 3], [4, 5]])
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert X_train_scaled.min() >= 0
    assert X_train_scaled.max() <= 1

