"""
Chart Tool: Declarative visualization builder powered by Plotly.
Constrained to safe chart specifications (bar, line, scatter, histogram, box) with automatic aggregations and modern dark aesthetics.
"""

import time
from typing import Dict, Any, Optional
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go

from agent.config import config
from agent.schema import ToolType, ToolExecutionResult, PlotChartParams, ChartType, AggregationType
from agent.tools.query_tool import QueryDataTool


class PlotChartTool:
    """Generates constrained, interactive Plotly visualizations based on declarative chart specs."""

    def __init__(self, df_or_path: Any):
        if isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path.copy()
        else:
            raise ValueError("df_or_path must be a path string or pandas DataFrame")

    def execute(self, params: PlotChartParams) -> ToolExecutionResult:
        start_time = time.time()
        try:
            df = self.df.copy()

            # Apply SQL filter if specified
            if params.filter_sql:
                clean_filter = params.filter_sql.strip()
                if clean_filter.upper().startswith("WHERE "):
                    clean_filter = clean_filter[6:]
                # Construct the full query first so validate_sql() can inspect it
                query = f"SELECT * FROM temp_ds WHERE {clean_filter}"
                # Run the same SQL safety guardrails used by query_tool.py
                _validator = QueryDataTool.__new__(QueryDataTool)
                _validator.validate_sql(query)
                # Execute safe filter via DuckDB
                con = duckdb.connect(database=":memory:")
                con.register("temp_ds", df)
                df = con.execute(query).fetchdf()

            if df.empty:
                raise ValueError("Chart generation failed: filter resulted in an empty dataset.")

            known_cols = list(df.columns)
            known_cols_lower = {c.lower(): c for c in known_cols}

            chart_type = params.chart_type
            x_col = params.x_column
            y_col = params.y_column
            group_col = params.group_by
            agg = params.aggregation

            # Validate X-axis column against exact known dataset columns
            if x_col not in known_cols:
                if x_col.lower() in known_cols_lower:
                    x_col = known_cols_lower[x_col.lower()]
                else:
                    raise ValueError(
                        f"Invalid x_column '{x_col}'. Must be an exact raw dataset column: {known_cols}. "
                        f"Embedding SQL syntax or expressions into column fields is prohibited."
                    )

            # Validate Y-axis column against exact known dataset columns
            if y_col:
                if y_col not in known_cols:
                    if y_col.lower() in known_cols_lower:
                        y_col = known_cols_lower[y_col.lower()]
                    else:
                        raise ValueError(
                            f"Invalid y_column '{y_col}'. Must be an exact raw dataset column: {known_cols}. "
                            f"Embedding SQL syntax or expressions into column fields is prohibited."
                        )

            # Validate group_by column against exact known dataset columns
            if group_col:
                if group_col not in known_cols:
                    if group_col.lower() in known_cols_lower:
                        group_col = known_cols_lower[group_col.lower()]
                    else:
                        raise ValueError(
                            f"Invalid group_by '{group_col}'. Must be an exact raw dataset column: {known_cols}. "
                            f"Embedding SQL syntax or expressions into column fields is prohibited."
                        )

            # Apply temporal date granularity if specified (e.g. year, quarter, month, day).
            # Detect datetime columns generically via pd.to_datetime inference rather than
            # relying on hardcoded Superstore-specific column names ("order_date", "ship_date").
            # This ensures date granularity works for any uploaded CSV dataset.
            if params.date_granularity:
                _is_date_col = False
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    _is_date_col = True
                else:
                    # Probe the raw string values (same heuristic used by DatasetProfiler)
                    _sample = df[x_col].dropna()
                    if len(_sample) > 0:
                        _sv = str(_sample.iloc[0])
                        if ("-" in _sv or "/" in _sv) and len(_sv) in (10, 19):
                            try:
                                pd.to_datetime(_sample.head(50), errors="raise")
                                _is_date_col = True
                            except Exception:
                                _is_date_col = False

                if _is_date_col:
                    dt_series = pd.to_datetime(df[x_col], errors="coerce")
                    gran = params.date_granularity.value if hasattr(params.date_granularity, "value") else str(params.date_granularity).lower()
                    if gran == "year":
                        df[x_col] = dt_series.dt.year.astype(str)
                    elif gran == "quarter":
                        df[x_col] = dt_series.dt.to_period("Q").astype(str)
                    elif gran == "month":
                        df[x_col] = dt_series.dt.to_period("M").astype(str)
                    elif gran == "day":
                        df[x_col] = dt_series.dt.strftime("%Y-%m-%d")

            # Aggregate if necessary for bar/line charts
            processed_df = df
            if agg and y_col and chart_type in [ChartType.BAR, ChartType.LINE]:
                group_keys = [x_col]
                if group_col and group_col in df.columns and group_col != x_col:
                    group_keys.append(group_col)

                agg_func = {
                    AggregationType.SUM: "sum",
                    AggregationType.MEAN: "mean",
                    AggregationType.COUNT: "count",
                    AggregationType.MIN: "min",
                    AggregationType.MAX: "max",
                    AggregationType.MEDIAN: "median"
                }.get(agg, "sum")

                processed_df = df.groupby(group_keys, as_index=False)[y_col].agg(agg_func)
                # Sort for clean visualization
                if chart_type == ChartType.LINE:
                    processed_df = processed_df.sort_values(by=x_col)
                elif chart_type == ChartType.BAR and not group_col:
                    processed_df = processed_df.sort_values(by=y_col, ascending=False).head(20)

            # Build Plotly Figure
            fig: go.Figure
            color_sequence = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899"]

            if chart_type == ChartType.BAR:
                fig = px.bar(
                    processed_df,
                    x=x_col,
                    y=y_col,
                    color=group_col if group_col in processed_df.columns else None,
                    title=params.title,
                    color_discrete_sequence=color_sequence,
                    barmode="group"
                )
            elif chart_type == ChartType.LINE:
                fig = px.line(
                    processed_df,
                    x=x_col,
                    y=y_col,
                    color=group_col if group_col in processed_df.columns else None,
                    title=params.title,
                    markers=True,
                    color_discrete_sequence=color_sequence
                )
            elif chart_type == ChartType.SCATTER:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=group_col if group_col in df.columns else None,
                    title=params.title,
                    color_discrete_sequence=color_sequence,
                    opacity=0.7
                )
            elif chart_type == ChartType.HISTOGRAM:
                fig = px.histogram(
                    df,
                    x=x_col,
                    color=group_col if group_col in df.columns else None,
                    title=params.title,
                    color_discrete_sequence=color_sequence,
                    marginal="box"
                )
            elif chart_type == ChartType.BOX:
                fig = px.box(
                    df,
                    x=x_col,
                    y=y_col,
                    color=group_col if group_col in df.columns else None,
                    title=params.title,
                    color_discrete_sequence=color_sequence
                )
            else:
                raise ValueError(f"Unsupported chart type '{chart_type}'")

            # Apply sleek modern dark theme styling
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(30, 41, 59, 0.5)",
                font=dict(family="Inter, sans-serif", size=13, color="#E2E8F0"),
                title=dict(font=dict(size=16, color="#F8FAFC", weight="bold")),
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(15, 23, 42, 0.5)"
                )
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)")

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            
            # Summary of aggregated data
            preview_rows = processed_df.head(15).to_dict(orient="records")
            for r in preview_rows:
                for k, v in r.items():
                    if pd.isna(v):
                        r[k] = None

            return ToolExecutionResult(
                tool=ToolType.PLOT_CHART,
                success=True,
                data={
                    "chart_spec": params.model_dump(),
                    "summary_table": preview_rows,
                    "row_count": len(processed_df),
                    "figure": fig,
                    "figure_dict": fig.to_dict()
                },
                row_count=len(processed_df),
                execution_time_ms=elapsed_ms,
                metadata={"title": params.title, "chart_type": params.chart_type.value}
            )

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return ToolExecutionResult(
                tool=ToolType.PLOT_CHART,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
                metadata={"title": params.title}
            )
