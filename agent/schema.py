"""
Pydantic schemas for the AI Data Analyst Agent.
Defines strict schemas for router decisions, tool parameters, execution results, and inspectable traces.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """The 4 fixed tools available to the agent router."""
    QUERY_DATA = "query_data"
    PLOT_CHART = "plot_chart"
    SUMMARY_STATS = "summary_stats"
    CLARIFY = "clarify"


class ChartType(str, Enum):
    """Supported chart types in the declarative chart tool."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"


class AggregationType(str, Enum):
    """Supported aggregation functions for charts."""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class DateGranularity(str, Enum):
    """Supported temporal grouping granularities for date columns."""
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"


class QueryDataParams(BaseModel):
    """Parameters for the read-only DuckDB SQL tool."""
    query: str = Field(..., description="A safe, read-only DuckDB SQL SELECT query referencing table `dataset`.")
    explanation: str = Field(..., description="Brief explanation of what the query calculates and how it answers the user question.")


class PlotChartParams(BaseModel):
    """Parameters for the declarative chart generation tool."""
    chart_type: ChartType = Field(..., description="The type of chart to generate: bar, line, scatter, histogram, box.")
    x_column: str = Field(..., description="Exact raw column name for the X-axis from the dataset schema. Do NOT pass SQL expressions or functions.")
    y_column: Optional[str] = Field(None, description="Exact raw column name for the Y-axis from the dataset schema (optional for histogram).")
    group_by: Optional[str] = Field(None, description="Optional raw column name for color/group breakdown.")
    aggregation: Optional[AggregationType] = Field(None, description="Aggregation function if grouping data (e.g. sum, mean).")
    date_granularity: Optional[DateGranularity] = Field(None, description="Optional temporal granularity when x_column is a date column (year, quarter, month, day).")
    title: str = Field(..., description="Descriptive title for the chart.")
    filter_sql: Optional[str] = Field(None, description="Optional SQL WHERE clause filter (e.g. `region = 'West'` or `order_date >= '2023-01-01'`).")


class SummaryStatsParams(BaseModel):
    """Parameters for the descriptive statistics tool."""
    columns: List[str] = Field(..., description="List of numeric or categorical column names to profile.")
    group_by: Optional[str] = Field(None, description="Optional categorical column to group descriptive statistics by.")


class ClarifyParams(BaseModel):
    """Parameters for the disambiguation and out-of-scope handler."""
    reason: Union[str, List[str]] = Field(..., description="Why the user query is ambiguous, missing required definitions, or out-of-scope.")
    missing_information: Union[str, List[str]] = Field(..., description="Specific metric, dimension, or timeframe missing from the request.")
    suggested_clarification: Union[str, List[str]] = Field(..., description="Constructive alternative questions or options the user can choose from.")

    def model_post_init(self, __context: Any) -> None:
        """Normalizes list inputs into human-readable strings if LLM emits array outputs."""
        if isinstance(self.reason, list):
            self.reason = "; ".join(self.reason)
        if isinstance(self.missing_information, list):
            self.missing_information = "; ".join(self.missing_information)
        if isinstance(self.suggested_clarification, list):
            self.suggested_clarification = "\n".join(f"- {s}" for s in self.suggested_clarification)


class RouterDecision(BaseModel):
    """Structured decision output emitted by the Router/Planner."""
    reasoning: str = Field(..., description="Step-by-step rationale for why the chosen tool is the appropriate approach.")
    intent: str = Field(..., description="Summarized intent of the user's question (e.g. 'aggregated metric lookup', 'time-series trend', 'ambiguous ranking').")
    tool: ToolType = Field(..., description="The selected tool to execute.")
    parameters: Dict[str, Any] = Field(..., description="Tool-specific argument payload conforming to the selected tool's schema.")


class ToolExecutionResult(BaseModel):
    """Standardized result returned by any executed tool."""
    tool: ToolType
    success: bool
    data: Optional[Any] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Complete, transparent telemetry record for an agent interaction."""
    query: str
    router_decision: RouterDecision
    tool_result: ToolExecutionResult
    narrative_response: str
    total_latency_ms: float
    numerical_validation_passed: bool
    numerical_validation_notes: Optional[str] = None
