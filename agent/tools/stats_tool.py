"""
Summary Stats Tool: Computes parametric and non-parametric descriptive statistics.
Supports multi-column profiling and categorical group-by breakdowns.
"""

import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from agent.schema import ToolType, ToolExecutionResult, SummaryStatsParams


class SummaryStatsTool:
    """Calculates comprehensive descriptive statistics for specified columns."""

    def __init__(self, df_or_path: Any):
        if isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path.copy()
        else:
            raise ValueError("df_or_path must be a path string or pandas DataFrame")

    def execute(self, params: SummaryStatsParams) -> ToolExecutionResult:
        start_time = time.time()
        try:
            df = self.df
            stats_results: Dict[str, Any] = {}
            target_cols = params.columns

            # Verify columns exist
            valid_cols = [c for c in target_cols if c in df.columns]
            if not valid_cols:
                raise ValueError(f"None of the requested columns {target_cols} were found in the dataset.")

            if params.group_by and params.group_by in df.columns:
                group_col = params.group_by
                grouped_summary = {}
                for col in valid_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        grp = df.groupby(group_col)[col].agg([
                            ("count", "count"),
                            ("mean", lambda x: round(x.mean(), 2)),
                            ("median", lambda x: round(x.median(), 2)),
                            ("std", lambda x: round(x.std(), 2)),
                            ("min", lambda x: round(x.min(), 2)),
                            ("max", lambda x: round(x.max(), 2)),
                            ("sum", lambda x: round(x.sum(), 2))
                        ]).reset_index()
                        grouped_summary[col] = grp.to_dict(orient="records")
                stats_results["grouped_stats"] = {
                    "group_by_column": group_col,
                    "metrics": grouped_summary
                }
            else:
                col_stats = {}
                for col in valid_cols:
                    series = df[col]
                    n_total = len(series)
                    n_null = int(series.isna().sum())
                    
                    if pd.api.types.is_numeric_dtype(series):
                        clean = series.dropna()
                        q25 = float(np.percentile(clean, 25)) if not clean.empty else 0.0
                        q75 = float(np.percentile(clean, 75)) if not clean.empty else 0.0
                        iqr = round(q75 - q25, 2)
                        skew = round(float(clean.skew()), 3) if len(clean) > 2 else 0.0

                        col_stats[col] = {
                            "type": "numeric",
                            "total_count": n_total,
                            "null_count": n_null,
                            "mean": round(float(clean.mean()), 2) if not clean.empty else 0.0,
                            "std": round(float(clean.std()), 2) if not clean.empty else 0.0,
                            "median": round(float(clean.median()), 2) if not clean.empty else 0.0,
                            "min": round(float(clean.min()), 2) if not clean.empty else 0.0,
                            "max": round(float(clean.max()), 2) if not clean.empty else 0.0,
                            "q25": round(q25, 2),
                            "q75": round(q75, 2),
                            "iqr": iqr,
                            "skewness": skew
                        }
                    else:
                        top_vals = series.value_counts(dropna=True).head(5).to_dict()
                        col_stats[col] = {
                            "type": "categorical",
                            "total_count": n_total,
                            "null_count": n_null,
                            "unique_count": int(series.nunique()),
                            "top_frequencies": {str(k): int(v) for k, v in top_vals.items()}
                        }

                stats_results["column_stats"] = col_stats

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return ToolExecutionResult(
                tool=ToolType.SUMMARY_STATS,
                success=True,
                data=stats_results,
                row_count=len(valid_cols),
                execution_time_ms=elapsed_ms,
                metadata={"columns_analyzed": valid_cols, "group_by": params.group_by}
            )

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return ToolExecutionResult(
                tool=ToolType.SUMMARY_STATS,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
                metadata={"columns": params.columns}
            )
