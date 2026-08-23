"""
Regression tests for CLI interface (cli.py).
Verifies:
1. display_profile consumes the actual dict contract from DatasetProfiler.profile() without AttributeError.
2. display_trace correctly extracts and displays SQL queries using the canonical 'query' parameter.
3. Cross-platform console encoding and unicode/emoji output safety.
"""

import sys
import io
import pytest
from rich.console import Console

from agent.config import config
from agent.profiler import DatasetProfiler
from agent.schema import (
    ToolType,
    RouterDecision,
    ToolExecutionResult,
    AgentTrace
)
import cli


def test_cli_display_profile_contract():
    """Test A: Verify display_profile works with DatasetProfiler's dict output."""
    profiler = DatasetProfiler(config.dataset_path)
    
    # Capture console output into string buffer
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)
    
    # Temporarily redirect cli.console
    orig_console = cli.console
    cli.console = test_console
    try:
        # Should not raise AttributeError or any exception
        cli.display_profile(profiler)
    finally:
        cli.console = orig_console

    output = buf.getvalue()
    # Verify core profile contents are present in the rendered output
    assert "Dataset Summary" in output
    assert "Total Transactions" in output
    assert "7,500" in output or "7500" in output
    assert "Schema & Missing Values Audit" in output
    assert "sales" in output
    assert "profit" in output
    assert "region" in output
    assert "Temporal Reference Anchors" in output


def test_cli_display_trace_sql_canonical_query_param():
    """Test B: Verify display_trace correctly extracts SQL using canonical 'query' parameter."""
    trace = AgentTrace(
        query="What is total sales?",
        router_decision=RouterDecision(
            reasoning="Metric is sales across full dataset. Use query_data.",
            intent="aggregation",
            tool=ToolType.QUERY_DATA,
            parameters={
                "query": "SELECT SUM(sales) AS total_sales FROM dataset",
                "explanation": "Calculates sum of sales column"
            }
        ),
        tool_result=ToolExecutionResult(
            tool=ToolType.QUERY_DATA,
            success=True,
            data={
                "columns": ["total_sales"],
                "rows": [{"total_sales": 8912345.67}],
                "row_count": 1
            },
            row_count=1,
            execution_time_ms=12.5
        ),
        narrative_response="Total sales reached **$8,912,345.67** across all orders.",
        total_latency_ms=1200.0,
        numerical_validation_passed=True,
        numerical_validation_notes="All numbers verified"
    )

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)
    
    orig_console = cli.console
    cli.console = test_console
    try:
        cli.display_trace(trace)
    finally:
        cli.console = orig_console

    output = buf.getvalue()
    # Verify inspectable trace elements
    assert "Inspectable Execution Trace" in output
    assert "query_data" in output
    # Verify SQL query from canonical 'query' parameter is rendered
    assert "Executed DuckDB SQL" in output
    assert "SELECT SUM(sales) AS total_sales FROM dataset" in output
    # Verify table and narrative response
    assert "Raw Data Output" in output
    assert "Analyst Response" in output
    assert "8,912,345.67" in output
    assert "Numerical Faithfulness Guard: PASSED" in output


def test_cli_display_trace_plot_chart():
    """Test B2: Verify display_trace works for plot_chart without errors."""
    trace = AgentTrace(
        query="Plot sales by category",
        router_decision=RouterDecision(
            reasoning="User requested a visualization. Use plot_chart.",
            intent="visualization",
            tool=ToolType.PLOT_CHART,
            parameters={
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "sales",
                "aggregation": "sum",
                "title": "Sales by Category"
            }
        ),
        tool_result=ToolExecutionResult(
            tool=ToolType.PLOT_CHART,
            success=True,
            data={
                "chart_spec": {
                    "chart_type": "bar",
                    "title": "Sales by Category"
                },
                "summary_table": [{"category": "Technology", "sales": 1000}],
                "row_count": 1
            },
            row_count=1,
            execution_time_ms=15.0
        ),
        narrative_response="Here is the bar chart of total sales by category.",
        total_latency_ms=1500.0,
        numerical_validation_passed=True
    )

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)
    
    orig_console = cli.console
    cli.console = test_console
    try:
        cli.display_trace(trace)
    finally:
        cli.console = orig_console

    output = buf.getvalue()
    assert "Inspectable Execution Trace" in output
    assert "plot_chart" in output
    assert "Rendered Plotly BAR" in output
    assert "Sales by Category" in output


def test_cli_unicode_and_banner_safety():
    """Test C: Verify banner, help, examples and emojis render safely without UnicodeEncodeError."""
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)
    
    orig_console = cli.console
    cli.console = test_console
    try:
        cli.print_banner()
        cli.display_help()
        cli.display_examples()
    finally:
        cli.console = orig_console

    output = buf.getvalue()
    assert "AI DATA ANALYST AGENT" in output
    assert "Available Commands" in output
    assert ":profile" in output
    assert "Sample Prompts to Try" in output
