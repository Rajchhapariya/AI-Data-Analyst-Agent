"""
Unit and Integration Tests for AI Data Analyst Agent.
Tests constrained tools, security guardrails, profiler, synthesizer faithfulness, and pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path

from agent.config import config
from agent.schema import (
    ToolType,
    QueryDataParams,
    PlotChartParams,
    ChartType,
    SummaryStatsParams,
    ClarifyParams,
    ToolExecutionResult
)
from agent.profiler import DatasetProfiler
from agent.tools.query_tool import QueryDataTool
from agent.tools.chart_tool import PlotChartTool
from agent.tools.stats_tool import SummaryStatsTool
from agent.tools.clarify_tool import ClarifyTool
from agent.synthesizer import ResponseSynthesizer
from agent.pipeline import DataAnalystAgent


@pytest.fixture
def sample_dataset_path():
    return config.dataset_path


def test_profiler_schema_and_nulls(sample_dataset_path):
    """Verifies that the DatasetProfiler correctly detects schema, row counts, and nulls."""
    profiler = DatasetProfiler(sample_dataset_path)
    profile = profiler.profile()
    
    assert profile["dataset_info"]["row_count"] == 7500
    assert "sales" in profile["columns"]
    assert "profit" in profile["columns"]
    assert "region" in profile["columns"]
    assert profile["dataset_info"]["memory_usage_mb"] > 0
    # Audit known missing values
    assert profile["columns"]["discount"]["null_count"] > 0
    assert profile["columns"]["shipping_cost"]["null_count"] > 0


def test_query_tool_safe_select(sample_dataset_path):
    """Verifies that safe SELECT queries execute and return structured rows."""
    tool = QueryDataTool(sample_dataset_path)
    params = QueryDataParams(
        query="SELECT region, SUM(sales) AS total_sales FROM dataset GROUP BY region ORDER BY total_sales DESC",
        explanation="Sum sales by region"
    )
    result = tool.execute(params)
    assert result.success is True
    assert len(result.data["rows"]) == 4
    assert "region" in result.data["columns"]
    assert "total_sales" in result.data["columns"]


def test_query_tool_sql_injection_guard(sample_dataset_path):
    """Verifies that malicious or mutating SQL queries are blocked with PermissionError."""
    tool = QueryDataTool(sample_dataset_path)
    
    # Attempt 1: DROP TABLE
    res1 = tool.execute(QueryDataParams(query="DROP TABLE dataset", explanation="test"))
    assert res1.success is False
    assert "security violation" in res1.error.lower()
    
    # Attempt 2: DELETE FROM
    res2 = tool.execute(QueryDataParams(query="DELETE FROM dataset WHERE sales > 100", explanation="test"))
    assert res2.success is False
    assert "security violation" in res2.error.lower()

    # Attempt 3: Multiple queries with semicolon
    res3 = tool.execute(QueryDataParams(query="SELECT 1; DROP TABLE dataset;", explanation="test"))
    assert res3.success is False
    assert "security violation" in res3.error.lower()


def test_chart_tool_bar_chart(sample_dataset_path):
    """Verifies that PlotChartTool creates a valid Plotly bar chart."""
    tool = PlotChartTool(sample_dataset_path)
    params = PlotChartParams(
        chart_type=ChartType.BAR,
        x_column="category",
        y_column="sales",
        aggregation="sum",
        title="Total Sales by Category"
    )
    result = tool.execute(params)
    assert result.success is True
    assert result.data["figure"] is not None
    assert result.data["chart_spec"]["chart_type"] == "bar"
    assert len(result.data["summary_table"]) == 3


def test_chart_tool_scatter_plot(sample_dataset_path):
    """Verifies that PlotChartTool creates a valid Plotly scatter plot."""
    tool = PlotChartTool(sample_dataset_path)
    params = PlotChartParams(
        chart_type=ChartType.SCATTER,
        x_column="sales",
        y_column="profit",
        title="Sales vs Profit Scatter"
    )
    result = tool.execute(params)
    assert result.success is True
    assert result.data["figure"] is not None
    assert result.data["chart_spec"]["chart_type"] == "scatter"


def test_stats_tool_descriptive_metrics(sample_dataset_path):
    """Verifies that SummaryStatsTool computes parametric and non-parametric statistics."""
    tool = SummaryStatsTool(sample_dataset_path)
    params = SummaryStatsParams(
        columns=["sales", "profit"]
    )
    result = tool.execute(params)
    assert result.success is True
    stats = result.data["column_stats"]
    assert "sales" in stats
    assert "profit" in stats
    assert "mean" in stats["sales"]
    assert "std" in stats["sales"]
    assert "q25" in stats["sales"]
    assert "q75" in stats["sales"]
    assert "iqr" in stats["sales"]


def test_clarify_tool():
    """Verifies structured clarification generation for ambiguous queries."""
    tool = ClarifyTool()
    params = ClarifyParams(
        reason="User did not specify whether 'best' refers to revenue or profit.",
        missing_information="Metric definition (sales vs profit)",
        suggested_clarification="Would you like to rank products by total sales or net profit?"
    )
    result = tool.execute(params)
    assert result.success is True
    assert "ambiguity_reason" in result.data
    assert len(result.data["suggested_actions"]) >= 1


def test_numerical_faithfulness_guard():
    """Verifies that ResponseSynthesizer correctly verifies grounded vs ungrounded numerical claims."""
    from agent.llm import default_llm_client
    synthesizer = ResponseSynthesizer(default_llm_client)
    
    # Scenario A: All numbers present in data
    raw_data = {"sales": 15000.50, "orders": 42}
    grounded_text = "The total sales reached $15,000.50 across 42 orders."
    passed_a, notes_a = synthesizer.verify_numerical_faithfulness(grounded_text, raw_data)
    assert passed_a is True

    # Scenario B: Hallucinated number (999999) not in data
    hallucinated_text = "The total sales reached $999,999.00 across 42 orders."
    passed_b, notes_b = synthesizer.verify_numerical_faithfulness(hallucinated_text, raw_data)
    assert passed_b is False
    assert "999999" in notes_b


def test_agent_pipeline_end_to_end(sample_dataset_path):
    """Verifies that DataAnalystAgent processes a query end-to-end and returns an inspectable trace."""
    agent = DataAnalystAgent(sample_dataset_path)
    trace = agent.ask("What is the total sales amount across all orders?")
    
    assert trace.query == "What is the total sales amount across all orders?"
    assert trace.router_decision.tool == ToolType.QUERY_DATA
    assert trace.tool_result.success is True
    assert len(trace.narrative_response) > 0
    assert trace.total_latency_ms > 0
