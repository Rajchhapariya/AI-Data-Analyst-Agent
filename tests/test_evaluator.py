"""
Unit and regression tests for evaluation correctness comparison logic.
Tests value-level, floating-point tolerance, reordering, and null-handling in compare_query_results.
"""

import pytest
import numpy as np
import pandas as pd

from evaluation.evaluator import compare_query_results


def test_compare_identical_dfs():
    """Verifies that identical DataFrames return True."""
    df1 = pd.DataFrame({"category": ["Furniture", "Technology"], "sales": [1234.56, 7890.12]})
    df2 = pd.DataFrame({"category": ["Furniture", "Technology"], "sales": [1234.56, 7890.12]})
    ok, msg = compare_query_results(df1, df2)
    assert ok is True
    assert "matched ground truth" in msg


def test_compare_different_values():
    """Verifies that DataFrames with different values are rejected."""
    df1 = pd.DataFrame({"category": ["Furniture", "Technology"], "sales": [100.0, 200.0]})
    df2 = pd.DataFrame({"category": ["Furniture", "Technology"], "sales": [999.0, 888.0]})
    ok, msg = compare_query_results(df1, df2)
    assert ok is False
    assert "Value mismatch" in msg


def test_compare_same_row_count_wrong_values():
    """Verifies that identical shape with wrong cell values returns False."""
    df_act = pd.DataFrame({"region": ["East", "West"], "total": [500.0, 600.0]})
    df_exp = pd.DataFrame({"region": ["East", "West"], "total": [500.0, 700.0]})
    ok, msg = compare_query_results(df_act, df_exp)
    assert ok is False
    assert "Value mismatch" in msg


def test_compare_different_row_counts():
    """Verifies that DataFrames with different row counts are rejected."""
    df1 = pd.DataFrame({"category": ["Furniture"], "sales": [100.0]})
    df2 = pd.DataFrame({"category": ["Furniture", "Technology"], "sales": [100.0, 200.0]})
    ok, msg = compare_query_results(df1, df2)
    assert ok is False
    assert "Row count mismatch" in msg


def test_compare_floating_point_rounding_tolerance():
    """Verifies that small floating point discrepancies within tolerance pass."""
    df1 = pd.DataFrame({"sales": [100.003, 250.004]})
    df2 = pd.DataFrame({"sales": [100.000, 250.000]})
    ok, msg = compare_query_results(df1, df2, atol=0.01)
    assert ok is True


def test_compare_reordered_rows_without_mandated_order():
    """Verifies that reordered aggregation rows match when order is not checked."""
    df1 = pd.DataFrame({"region": ["South", "North"], "sales": [300.0, 100.0]})
    df2 = pd.DataFrame({"region": ["North", "South"], "sales": [100.0, 300.0]})
    ok, msg = compare_query_results(df1, df2, check_order=False)
    assert ok is True


def test_compare_reordered_rows_with_mandated_order():
    """Verifies that reordered rows fail when strict ordering is mandated (e.g. ranked lists)."""
    df1 = pd.DataFrame({"region": ["South", "North"], "sales": [300.0, 100.0]})
    df2 = pd.DataFrame({"region": ["North", "South"], "sales": [100.0, 300.0]})
    ok, msg = compare_query_results(df1, df2, check_order=True)
    assert ok is False
    assert "Value mismatch" in msg


def test_compare_null_nan_handling():
    """Verifies that matching NaN/NULL values pass and mismatched NULLs fail."""
    df1 = pd.DataFrame({"col1": ["A", "B"], "col2": [np.nan, 50.0]})
    df2 = pd.DataFrame({"col1": ["A", "B"], "col2": [np.nan, 50.0]})
    ok, msg = compare_query_results(df1, df2)
    assert ok is True

    df3 = pd.DataFrame({"col1": ["A", "B"], "col2": [10.0, 50.0]})
    ok2, msg2 = compare_query_results(df3, df2)
    assert ok2 is False
    assert "Null mismatch" in msg2


def test_compare_ratio_to_percentage_scaling():
    """Verifies that decimal ratios (0.14) and percentages (14.0) match under scale tolerance."""
    df_ratio = pd.DataFrame({"category": ["Tech", "Office", "Furniture"], "margin": [0.14, 0.15, -0.14]})
    df_pct = pd.DataFrame({"category": ["Tech", "Office", "Furniture"], "margin_pct": [14.0, 15.0, -14.0]})
    ok, msg = compare_query_results(df_ratio, df_pct)
    assert ok is True
    assert "matched ground truth" in msg
