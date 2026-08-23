"""
Dataset Profiler: Generates comprehensive schema, statistical distribution, data quality audits,
and dynamic temporal reference anchors for historical datasets.
Provides compact schema summaries for LLM router context.
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np


class DatasetProfiler:
    """Profiles a pandas DataFrame or CSV file for schema, quality, and statistical summaries."""

    def __init__(self, df_or_path: Any):
        """Initializes the statistical dataset profiler.
        
        Args:
            df_or_path: CSV filepath string or active in-memory pandas DataFrame.
        """
        if isinstance(df_or_path, str):
            self.file_path = df_or_path
            self.df = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            self.file_path = "in_memory_dataframe"
            self.df = df_or_path.copy()
        else:
            raise ValueError("df_or_path must be a file path (str) or a pandas DataFrame")

        self._profile_cache: Optional[Dict[str, Any]] = None

    def profile(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Runs full profiling on the dataset and returns a structured dictionary."""
        if self._profile_cache is not None and not force_refresh:
            return self._profile_cache

        df = self.df
        num_rows, num_cols = df.shape
        if num_rows == 0:
            raise ValueError("Dataset contains no data rows (empty or header-only file).")
        columns_profile = {}
        data_quality_issues = []

        # Datetime column candidate detection
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = round((null_count / num_rows) * 100, 2)
            unique_count = int(series.nunique(dropna=True))

            # Detect semantic type
            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            is_datetime = False
            datetime_range = None

            # Attempt datetime inference if string
            if not is_numeric and series.dropna().shape[0] > 0:
                sample_val = str(series.dropna().iloc[0])
                if ("-" in sample_val or "/" in sample_val) and len(sample_val) in (10, 19):
                    try:
                        parsed = pd.to_datetime(series.dropna().head(100))
                        is_datetime = True
                        dt_series = pd.to_datetime(series.dropna())
                        datetime_range = {
                            "min": str(dt_series.min().strftime("%Y-%m-%d")),
                            "max": str(dt_series.max().strftime("%Y-%m-%d"))
                        }
                    except Exception:
                        is_datetime = False

            col_info: Dict[str, Any] = {
                "name": col,
                "dtype": dtype,
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": unique_count,
                "semantic_type": "datetime" if is_datetime else ("numeric" if is_numeric else ("categorical" if unique_count < 100 else "text_identifier"))
            }

            if null_count > 0:
                data_quality_issues.append({
                    "column": col,
                    "issue_type": "missing_values",
                    "severity": "medium" if null_pct > 5.0 else "low",
                    "description": f"Column '{col}' has {null_count} missing values ({null_pct}%)."
                })

            if is_numeric:
                clean_num = series.dropna()
                if not clean_num.empty:
                    min_v = float(np.round(clean_num.min(), 2))
                    max_v = float(np.round(clean_num.max(), 2))
                    mean_v = float(np.round(clean_num.mean(), 2))
                    std_val = clean_num.std()
                    std_v = float(np.round(std_val, 2)) if len(clean_num) > 1 and not pd.isna(std_val) else 0.0
                    med_v = float(np.round(clean_num.median(), 2))
                    q25_v = float(np.round(clean_num.quantile(0.25), 2))
                    q75_v = float(np.round(clean_num.quantile(0.75), 2))
                else:
                    min_v = max_v = mean_v = std_v = med_v = q25_v = q75_v = 0.0

                col_info["stats"] = {
                    "min": min_v,
                    "max": max_v,
                    "mean": mean_v,
                    "std": std_v,
                    "median": med_v,
                    "q25": q25_v,
                    "q75": q75_v,
                }
                # Check for unexpected negatives
                if col in ["sales", "quantity", "shipping_cost", "discount"] and (clean_num < 0).any():
                    data_quality_issues.append({
                        "column": col,
                        "issue_type": "negative_value_violation",
                        "severity": "high",
                        "description": f"Column '{col}' contains negative values which violates domain logic."
                    })
            elif is_datetime:
                col_info["datetime_range"] = datetime_range
            else:
                # Categorical or text: top distinct values
                top_vals = series.value_counts(dropna=True).head(8).to_dict()
                col_info["sample_values"] = {str(k): int(v) for k, v in top_vals.items()}

            columns_profile[col] = col_info

        # Compute dynamic Temporal Reference Anchors from primary date column (e.g. order_date)
        temporal_anchors = {}
        date_candidates = [c for c, ci in columns_profile.items() if ci["semantic_type"] == "datetime"]
        # Prefer 'order_date' if present
        primary_date_col = "order_date" if "order_date" in date_candidates else (date_candidates[0] if date_candidates else None)
        
        if primary_date_col and primary_date_col in df.columns:
            try:
                dt_series = pd.to_datetime(df[primary_date_col].dropna())
                if not dt_series.empty:
                    max_dt = dt_series.max()
                    min_dt = dt_series.min()
                    max_year = int(max_dt.year)
                    last_year = max_year - 1
                    quarter = (max_dt.month - 1) // 3 + 1
                    q_start_m = (quarter - 1) * 3 + 1
                    temporal_anchors = {
                        "date_column": primary_date_col,
                        "current_year": max_year,
                        "previous_year": last_year,
                        "most_recent_quarter": f"Q{quarter} {max_year}",
                        "most_recent_quarter_months": [
                            f"{max_year}-{q_start_m:02d}",
                            f"{max_year}-{q_start_m+1:02d}",
                            f"{max_year}-{q_start_m+2:02d}"
                        ],
                        "most_recent_month": max_dt.strftime("%Y-%m"),
                        "min_date": str(min_dt.strftime("%Y-%m-%d")),
                        "max_date": str(max_dt.strftime("%Y-%m-%d")),
                    }
            except Exception:
                pass

        profile_data = {
            "dataset_info": {
                "file_path": self.file_path,
                "row_count": num_rows,
                "column_count": num_cols,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "columns": list(df.columns)
            },
            "columns": columns_profile,
            "temporal_anchors": temporal_anchors,
            "data_quality_audit": {
                "total_issues_found": len(data_quality_issues),
                "issues": data_quality_issues
            }
        }

        self._profile_cache = profile_data
        return profile_data

    def get_llm_schema_prompt_context(self) -> str:
        """
        Formats a clean, high-density schema summary text block for injection into the LLM Router prompt.
        Ensures the agent knows exact column names, data types, distinct categories, and temporal anchors.
        """
        p = self.profile()
        info = p["dataset_info"]
        lines = [
            f"Table Name: `dataset` (Rows: {info['row_count']:,}, Columns: {info['column_count']})",
            "Columns & Semantic Types:"
        ]

        for col_name, cinfo in p["columns"].items():
            stype = cinfo["semantic_type"]
            dtype = cinfo["dtype"]
            null_str = f", {cinfo['null_count']} nulls" if cinfo['null_count'] > 0 else ""

            if stype == "numeric":
                stats = cinfo.get("stats", {})
                lines.append(
                    f"  - `{col_name}` ({dtype}, numeric{null_str}): "
                    f"min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, median={stats.get('median')}"
                )
            elif stype == "datetime":
                dr = cinfo.get("datetime_range", {})
                lines.append(
                    f"  - `{col_name}` ({dtype}, datetime{null_str}): "
                    f"range [{dr.get('min')} to {dr.get('max')}]"
                )
            elif stype == "categorical":
                sample_keys = list(cinfo.get("sample_values", {}).keys())[:6]
                samples_str = ", ".join([f"'{k}'" for k in sample_keys])
                lines.append(
                    f"  - `{col_name}` ({dtype}, categorical{null_str}, {cinfo['unique_count']} unique): "
                    f"e.g. [{samples_str}]"
                )
            else:
                lines.append(f"  - `{col_name}` ({dtype}, identifier{null_str}, {cinfo['unique_count']} unique)")

        # Inject dynamic temporal reference anchors
        if p.get("temporal_anchors"):
            t = p["temporal_anchors"]
            lines.append("\nTemporal Reference Anchors (Derived from Dataset):")
            lines.append(f"  - Current / Most Recent Year ('this year', 'current year'): {t['current_year']}")
            lines.append(f"  - Previous Year ('last year', 'prior year'): {t['previous_year']}")
            lines.append(f"  - Most Recent Quarter ('recent quarter', 'latest quarter'): {t['most_recent_quarter']} (months: {', '.join(t['most_recent_quarter_months'])})")
            lines.append(f"  - Most Recent Month ('last month', 'recent month'): {t['most_recent_month']}")
            lines.append(f"  - Dataset Full Date Span: {t['min_date']} to {t['max_date']}")
        else:
            lines.append("\nTemporal Reference Anchors: None (This dataset contains zero date/time columns; temporal filtering is not available).")

        return "\n".join(lines)


if __name__ == "__main__":
    profiler = DatasetProfiler("data/superstore_sales.csv")
    profile_res = profiler.profile()
    
    # Save full JSON profile
    with open("data/data_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_res, f, indent=2)
        
    print("=== DATASET PROFILE SUMMARY ===")
    print(f"Total Rows: {profile_res['dataset_info']['row_count']:,}")
    print(f"Total Columns: {profile_res['dataset_info']['column_count']}")
    print(f"Memory: {profile_res['dataset_info']['memory_usage_mb']} MB")
    print(f"\nTemporal Anchors: {profile_res.get('temporal_anchors')}")
    print("\n=== LLM SCHEMA PROMPT CONTEXT ===")
    print(profiler.get_llm_schema_prompt_context())
