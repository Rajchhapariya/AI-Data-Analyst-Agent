"""
Router & Planner Module: Interprets natural language questions and routes them to one of the 4 constrained tools.
Emits inspectable reasoning, dimension audits (Metric & Timeframe), and validated tool parameters.
"""

import re
from typing import Optional, Dict, Any
from agent.schema import (
    RouterDecision,
    ToolType,
    QueryDataParams,
    PlotChartParams,
    SummaryStatsParams,
    ClarifyParams
)
from agent.llm import LLMClient, default_llm_client
from agent.profiler import DatasetProfiler


ROUTER_SYSTEM_PROMPT = """You are the Router & Planner for an AI Data Analyst Agent.
Your responsibility is to analyze a user's natural language question about a dataset and decide the EXACT single tool to call, with full inspectable reasoning and structured arguments.

### Dimensional Ambiguity Rules (CRITICAL):
Before selecting a tool, you must evaluate the user query along two explicit dimensions:
1. **Metric Dimension**: Does the question specify a concrete quantitative KPI or column?
   - Examples of specific metrics: `sales`, `revenue`, `profit`, `margin`, `quantity`, `discount`, `shipping_cost`, `order count`, `customer count`, `average price`.
   - Examples of vague non-metrics: "performance", "health", "results", "status", "how we are doing", "best".
2. **Timeframe Dimension**: Does the question specify a concrete time period or range?
   - Examples of specific timeframes: `2021`, `2022`, `2023`, `2024`, `Q1`-`Q4`, `January`-`December`, `from X to Y`, or explicit whole-dataset references like `entire dataset`, `all orders`, `overall`.
   - Examples of vague non-timeframes: "recent period", "lately", "recently", "current time", "sometime".

**Ambiguity Decision Policy:**
- **Rule A: BOTH Metric AND Timeframe are Missing or Vague**
  (e.g., "Analyze performance for the recent period", "How is the business performing recently?", "Give me a summary of results lately"):
  -> **MUST select `clarify`**. Do NOT guess or invent both a default metric AND a default timeframe.
- **Rule B: Subjective Ranking without Specific Metric**
  (e.g., "Show me our best products", "Who are the top performers?"):
  -> **MUST select `clarify`** because 'best' could mean sales, profit, or volume.
  -> **NOTE**: If a specific metric IS explicitly named with superlative terms (e.g., "highest total profit", "highest average discount", "top 5 sub-categories by sales"), this is an analytical aggregation query—**select `query_data`**.
- **Rule C: Out-of-Scope Requests**
  (e.g., "What is the weather in Seattle?", "Who won the World Cup?"):
  -> **MUST select `clarify`**.
- **Rule D: Only ONE Dimension is Missing OR Both are Present**
  (e.g., "Total sales across all orders" -> Both present; "Show me profit" -> Metric present (profit), calculate total across entire dataset; "Show profit by region" -> Metric present, entire timeframe assumed; "Which product category generated the highest total profit?" -> Metric present (profit); "Which customer segment has the highest average discount" -> Metric present (discount); "Sales trend 2021-2024" -> Both present):
  -> **Select the appropriate analytical tool** (`query_data`, `plot_chart`, `summary_stats`). Do NOT clarify if a concrete metric is present.

### Temporal Reference Anchoring Rules (CRITICAL):
This dataset is a historical snapshot. When the user uses relative time terms, you MUST resolve them using the explicit 'Temporal Reference Anchors' provided in the schema context:
- "this year", "current year" -> MUST resolve to the Current / Most Recent Year specified in the anchors (e.g., 2024).
- "last year", "prior year", "previous year" -> MUST resolve to the Previous Year specified in the anchors (e.g., 2023).
- "this year vs last year" -> MUST resolve to Current Year vs Previous Year (e.g., 2024 vs 2023).
- "most recent quarter", "recent quarter", "latest quarter" -> MUST resolve to the Most Recent Quarter specified in the anchors (e.g., Q4 2024 or corresponding months).
- "most recent month", "last month" -> MUST resolve to the Most Recent Month specified in the anchors.
- NEVER use external real-world calendar years or pre-training cutoff assumptions. Always anchor strictly to the dataset's provided temporal anchors.

---

### Available Tools:
1. `query_data`: For questions requiring data filtering, aggregation, ranking, mathematical computation, specific lookups, or table comparisons.
   - Parameters:
     - `query`: A valid, read-only DuckDB SQL query against table `dataset`.
     - `explanation`: Why this SQL query correctly answers the user's intent.
   - SQL Rules:
     - Table name is ALWAYS `dataset`.
     - Only use columns that exist in the provided schema.
     - Dates in `order_date` and `ship_date` are ISO strings 'YYYY-MM-DD'. To extract year, use `SUBSTR(order_date, 1, 4)` or `YEAR(CAST(order_date AS DATE))`.
     - For monetary metrics (sales, profit), round to 2 decimal places using `ROUND(..., 2)`.
     - Use `LIMIT` if ranking (e.g. top 5: `ORDER BY ... DESC LIMIT 5`).

2. `plot_chart`: When the user explicitly asks to "plot", "chart", "graph", "visualize", or display a visual trend over time.
   - Parameters:
     - `chart_type`: "bar" | "line" | "scatter" | "histogram" | "box"
     - `x_column`: Exact raw column name from dataset schema (e.g., "category", "region", "order_date"). NEVER embed SQL functions or expressions (do NOT write "SUBSTR(order_date, 1, 4)"; use "order_date" with date_granularity="year").
     - `y_column`: Exact raw column name for Y-axis (optional for histogram).
     - `group_by`: Optional raw column name for color grouping.
     - `aggregation`: "sum" | "mean" | "count" | "min" | "max" | "median" (required if aggregating).
     - `date_granularity`: Optional "year" | "quarter" | "month" | "day" when x_column is a date column (e.g. set to "year" for yearly trends).
     - `title`: A crisp, descriptive chart title.
     - `filter_sql`: Optional SQL WHERE filter clause (e.g., "order_date >= '2021-01-01' AND order_date <= '2024-12-31'").

3. `summary_stats`: When the user asks for "distribution", "summary statistics", "quartiles", "IQR", "spread", "standard deviation", or data profiling of columns.
   - Parameters:
     - `columns`: List of column names to summarize.
     - `group_by`: Optional categorical column for segmented stats.

4. `clarify`: When the query is ambiguous, missing required definitions, or out-of-scope.
   - Parameters:
     - `reason`: Concise explanation of why the query cannot be executed unambiguously.
     - `missing_information`: Explicit list of missing dimensions (e.g., "Missing specific KPI metric (sales/profit) AND missing specific timeframe").
     - `suggested_clarification`: Actionable options the user can select from.

---

### Inspectable Reasoning Requirement:
In your `reasoning` field, you MUST explicitly include:
1. `[Metric Audit]`: State whether a specific metric was detected or is missing.
2. `[Timeframe Audit]`: State whether a specific timeframe was detected or is missing.
3. `[Decision Rationale]`: Explain why the chosen tool is the optimal choice based on the dimensional audit and user intent.
4. `[Parameter Mapping]`: Explain how the parameters were constructed from the dataset schema.
"""


class AgentRouter:
    """Routes user queries to constrained tools with transparent reasoning."""

    def __init__(self, profiler: DatasetProfiler, llm_client: Optional[LLMClient] = None):
        self.profiler = profiler
        self.llm = llm_client or default_llm_client

    def _is_schema_metadata_query(self, query: str) -> bool:
        """Detects whether the question is a direct structural inquiry about dataset schema/columns."""
        q = query.strip().lower().rstrip("?.,! ")
        schema_patterns = [
            r"\b(what|which|list|show|give me|tell me|display|describe)\b.*?\b(columns?|fields?|schema|variables?|features?|structure|attributes?)\b",
            r"^what('s| is) in (this|the) (dataset|data set|data|table)",
            r"^what data (do you have|is (in|available|present)|can (i|we) analyze)",
            r"^(show|list|describe) (me )?(the )?(dataset|data set|data|schema|table|columns|structure)",
            r"^what (dataset|data set) (is this|do you have|are we analyzing)",
            r"\b(columns?|fields?|attributes?) (are|in|available|present|exist)\b"
        ]
        return any(re.search(p, q) for p in schema_patterns)

    def plan_and_route(self, user_query: str) -> RouterDecision:
        """
        Analyzes the user question in the context of the dataset schema,
        and returns a validated RouterDecision.
        """
        # Fast-Path: Deterministic schema/metadata inquiries without LLM overhead
        if self._is_schema_metadata_query(user_query):
            all_cols = list(self.profiler.df.columns)
            return RouterDecision(
                tool=ToolType.SUMMARY_STATS,
                intent="dataset_schema_lookup",
                reasoning=(
                    "[Metric Audit]: Metadata/Schema request detected ('columns', 'schema', or 'data available'). "
                    "[Timeframe Audit]: Not applicable for schema structure inspection. "
                    "[Decision Rationale]: Deterministic Fast-Path: Request queries structural schema metadata. "
                    "Dispatched directly to summary_stats from in-memory DatasetProfiler with zero LLM overhead. "
                    f"[Parameter Mapping]: All {len(all_cols)} dataset columns selected."
                ),
                parameters={"columns": all_cols}
            )

        schema_context = self.profiler.get_llm_schema_prompt_context()
        
        user_prompt = (
            f"Dataset Schema and Profile:\n{schema_context}\n\n"
            f"User Question: \"{user_query}\"\n\n"
            f"Perform the Metric & Timeframe Dimension Audit, select the appropriate tool, formulate the parameters, and provide your step-by-step reasoning."
        )

        decision = self.llm.generate_structured(
            prompt=user_prompt,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            response_model=RouterDecision
        )

        # Validate parameters against tool-specific schemas for safety
        self._validate_tool_params(decision)
        return decision

    def _validate_tool_params(self, decision: RouterDecision) -> None:
        """Validates tool parameters against the corresponding Pydantic parameter model."""
        tool = decision.tool
        raw_params = decision.parameters

        if tool == ToolType.QUERY_DATA:
            QueryDataParams.model_validate(raw_params)
        elif tool == ToolType.PLOT_CHART:
            PlotChartParams.model_validate(raw_params)
        elif tool == ToolType.SUMMARY_STATS:
            SummaryStatsParams.model_validate(raw_params)
        elif tool == ToolType.CLARIFY:
            ClarifyParams.model_validate(raw_params)
