"""
Unit and Integration Tests for AI Data Analyst Agent.
Tests constrained tools, security guardrails, profiler, synthesizer faithfulness, and pipeline.
"""

import pytest
import pandas as pd
import numpy as np
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

    # Attempt 4: DuckDB read_csv file access
    res4 = tool.execute(QueryDataParams(query="SELECT * FROM read_csv('data/superstore_sales.csv')", explanation="test"))
    assert res4.success is False
    assert "security violation" in res4.error.lower()

    # Attempt 5: DuckDB read_parquet file access
    res5 = tool.execute(QueryDataParams(query="SELECT * FROM read_parquet('/etc/passwd')", explanation="test"))
    assert res5.success is False
    assert "security violation" in res5.error.lower()

    # Attempt 6: DuckDB read_json_auto file access
    res6 = tool.execute(QueryDataParams(query="SELECT * FROM read_json_auto('credentials.json')", explanation="test"))
    assert res6.success is False
    assert "security violation" in res6.error.lower()

    # Attempt 7: DuckDB getenv system function
    res7 = tool.execute(QueryDataParams(query="SELECT getenv('OPENAI_API_KEY')", explanation="test"))
    assert res7.success is False
    assert "security violation" in res7.error.lower()


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
    raw_data = {"sales": 15000.50, "orders": 42, "discount_rate": 0.155, "loss": -250.0}
    grounded_text = "The total sales reached $15,000.50 across 42 orders."
    passed_a, notes_a = synthesizer.verify_numerical_faithfulness(grounded_text, raw_data)
    assert passed_a is True

    # Scenario B: Hallucinated number (999999) not in data
    hallucinated_text = "The total sales reached $999,999.00 across 42 orders."
    passed_b, notes_b = synthesizer.verify_numerical_faithfulness(hallucinated_text, raw_data)
    assert passed_b is False
    assert "999999" in notes_b

    # Scenario C: Percentage representation (0.155 -> 15.5%)
    pct_text = "The average discount rate was 15.5% across 42 orders."
    passed_c, notes_c = synthesizer.verify_numerical_faithfulness(pct_text, raw_data)
    assert passed_c is True

    # Scenario D: Scaled Millions ($9,167,421.88 -> $9.17M)
    raw_millions = {"total_sales": 9167421.88}
    scaled_text = "Total sales reached approximately $9.17M across all regions."
    passed_d, notes_d = synthesizer.verify_numerical_faithfulness(scaled_text, raw_millions)
    assert passed_d is True

    # Scenario E: Negative number (-$250.00)
    neg_text = "The regional net loss was -$250.00."
    passed_e, notes_e = synthesizer.verify_numerical_faithfulness(neg_text, raw_data)
    assert passed_e is True

    # Scenario F: Allowed ordinal ranks and years
    ordinal_text = "In 2024, the top 3 categories drove volume."
    passed_f, notes_f = synthesizer.verify_numerical_faithfulness(ordinal_text, raw_data)
    assert passed_f is True


def test_agent_pipeline_end_to_end(sample_dataset_path):
    """Verifies that DataAnalystAgent processes a query end-to-end and returns an inspectable trace."""
    agent = DataAnalystAgent(sample_dataset_path)
    trace = agent.ask("What is the total sales amount across all orders?")
    
    assert trace.query == "What is the total sales amount across all orders?"
    assert trace.router_decision.tool == ToolType.QUERY_DATA
    assert trace.tool_result.success is True
    assert len(trace.narrative_response) > 0
    assert trace.total_latency_ms > 0


def test_agent_dataframe_initialization():
    """Verifies that DataAnalystAgent can be initialized directly with an in-memory pandas DataFrame."""
    df = pd.DataFrame({
        "employee": ["Alice", "Bob", "Charlie"],
        "department": ["Engineering", "Sales", "Engineering"],
        "salary": [120000, 95000, 110000]
    })
    agent = DataAnalystAgent(dataset_path_or_df=df)
    assert agent.profiler is not None
    assert agent.profiler.file_path == "in_memory_dataframe"
    profile = agent.profiler.profile()
    assert profile["dataset_info"]["row_count"] == 3


def test_custom_csv_upload_ingestion_and_query():
    """Regression test: verifies that custom uploaded CSV DataFrames with boolean, numeric,

    and date fields ingest cleanly without ambiguous truth-value evaluation errors.
    """
    import io
    custom_csv_content = """employee_id,name,department,salary,active,join_date
101,Alice Smith,Engineering,125000.50,True,2021-03-15
102,Bob Jones,Marketing,85000.00,False,2022-07-01
103,Charlie Brown,Engineering,110000.00,True,2020-01-10
104,Diana Prince,Sales,95000.75,True,2023-11-20
"""
    df_uploaded = pd.read_csv(io.StringIO(custom_csv_content))
    
    # 1. Validation checks (as executed in Streamlit sidebar)
    assert not df_uploaded.empty
    assert len(df_uploaded) == 4
    assert len(df_uploaded.columns) == 6
    
    # 2. Agent Initialization
    agent = DataAnalystAgent(dataset_path_or_df=df_uploaded)
    assert agent.profiler is not None
    
    # 3. Profiler Execution
    profile = agent.profiler.profile()
    assert profile["dataset_info"]["row_count"] == 4
    assert profile["dataset_info"]["column_count"] == 6
    assert profile["columns"]["active"]["semantic_type"] != "numeric"
    assert profile["columns"]["salary"]["semantic_type"] == "numeric"
    
    # 4. Schema context generation
    schema_ctx = agent.profiler.get_llm_schema_prompt_context()
    assert "salary" in schema_ctx
    assert "employee_id" in schema_ctx
    
    # 5. Deterministic query execution
    q_result = agent.query_tool.execute(QueryDataParams(
        query="SELECT department, AVG(salary) AS avg_sal FROM dataset GROUP BY department",
        explanation="Average salary by department"
    ))
    assert q_result.success is True
    assert len(q_result.data["rows"]) == 3


def test_custom_csv_empty_and_zero_row_validation():
    """Verifies that empty and header-only custom CSVs are rejected with clear ValueError."""
    import pytest
    import io
    
    # Empty DataFrame (0 rows, 0 cols)
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="no data rows"):
        profiler = DatasetProfiler(df_empty)
        profiler.profile()

    # Header-only DataFrame (0 rows, 3 cols)
    header_only_csv = "col1,col2,col3\n"
    df_header_only = pd.read_csv(io.StringIO(header_only_csv))
    with pytest.raises(ValueError, match="no data rows"):
        profiler = DatasetProfiler(df_header_only)
        profiler.profile()


def test_custom_csv_with_boolean_and_null_columns():
    """Verifies that boolean columns and all-null columns do not cause math or truth-value errors."""
    df_bool_and_nulls = pd.DataFrame({
        "flag": [True, False, True],
        "null_num": [np.nan, np.nan, np.nan],
        "category": ["A", "B", "A"],
        "revenue": [100.0, 200.0, 300.0]
    })
    
    agent = DataAnalystAgent(dataset_path_or_df=df_bool_and_nulls)
    profile = agent.profiler.profile()
    
    # Verify boolean column is not treated as numeric
    assert profile["columns"]["flag"]["semantic_type"] == "categorical"
    # Verify all-null column does not crash stats
    assert profile["columns"]["null_num"]["null_count"] == 3
    
    # Verify Stats tool works on the numeric column
    stats_res = agent.stats_tool.execute(SummaryStatsParams(columns=["revenue"]))
    assert stats_res.success is True
    assert stats_res.data["column_stats"]["revenue"]["mean"] == 200.0
