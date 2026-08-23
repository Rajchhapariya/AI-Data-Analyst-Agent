"""
Query Data Tool: Executes validated, read-only DuckDB SQL queries over the dataset.
Includes strict AST/keyword guardrails, row-count limits, and execution timing.
"""

import time
import re
from typing import Dict, Any, Optional, List
import pandas as pd
import duckdb

from agent.config import config
from agent.schema import ToolType, ToolExecutionResult, QueryDataParams


class QueryDataTool:
    """Read-only SQL tool powered by DuckDB in-memory engine with safety guardrails."""

    def __init__(self, df_or_path: Any):
        if isinstance(df_or_path, str):
            self.df = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path.copy()
        else:
            raise ValueError("df_or_path must be a path string or pandas DataFrame")

        self.con = duckdb.connect(database=":memory:")
        # Register the dataframe as the target table
        self.con.register(config.table_name, self.df)

    def validate_sql(self, sql_query: str) -> None:
        """
        Validates that the SQL query is strictly read-only and safe.
        Raises ValueError on any violation.
        """
        clean_sql = sql_query.strip()
        
        # Remove comments (-- or /* */) before validation
        clean_sql = re.sub(r"--.*?(\n|$)", " ", clean_sql)
        clean_sql = re.sub(r"/\*.*?\*/", " ", clean_sql, flags=re.DOTALL).strip()
        
        # Must start with SELECT or WITH (for CTEs)
        upper_sql = clean_sql.upper()
        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
            raise ValueError("Security violation: Only read-only SELECT and WITH (CTE) queries are permitted.")

        # Disallow statement chaining with semicolons
        semicolon_count = clean_sql.count(";")
        if semicolon_count > 1 or (semicolon_count == 1 and not clean_sql.endswith(";")):
            raise ValueError("Security violation: Multiple chained SQL statements are not allowed.")

        # Check for disallowed mutating / DDL / administrative keywords
        for forbidden in config.disallowed_sql_keywords:
            pattern = rf"\b{forbidden}\b"
            if re.search(pattern, upper_sql):
                raise ValueError(f"Security violation: SQL keyword '{forbidden}' is prohibited in read-only mode.")

    def execute(self, params: QueryDataParams) -> ToolExecutionResult:
        """Executes the query and returns a standardized ToolExecutionResult."""
        start_time = time.time()
        raw_sql = params.query.strip().rstrip(";")
        
        try:
            self.validate_sql(raw_sql)
            
            # Execute query on DuckDB
            result_df: pd.DataFrame = self.con.execute(raw_sql).fetchdf()
            
            # Enforce max row cap
            if len(result_df) > config.max_query_rows:
                result_df = result_df.head(config.max_query_rows)
                row_cap_applied = True
            else:
                row_cap_applied = False
                
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            
            # Format serializable records
            # Handle timestamps and NaN for clean JSON
            clean_records = result_df.to_dict(orient="records")
            # Replace NaNs with None
            for row in clean_records:
                for k, v in row.items():
                    if pd.isna(v):
                        row[k] = None

            return ToolExecutionResult(
                tool=ToolType.QUERY_DATA,
                success=True,
                data={
                    "columns": list(result_df.columns),
                    "rows": clean_records,
                    "row_count": len(result_df),
                    "dataframe": result_df # Retained for programmatic consumers
                },
                row_count=len(result_df),
                execution_time_ms=elapsed_ms,
                metadata={
                    "sql_query": raw_sql,
                    "explanation": params.explanation,
                    "row_cap_applied": row_cap_applied
                }
            )

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return ToolExecutionResult(
                tool=ToolType.QUERY_DATA,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
                metadata={"sql_query": raw_sql, "explanation": params.explanation}
            )
