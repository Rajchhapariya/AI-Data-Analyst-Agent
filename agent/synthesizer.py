"""
Response Synthesizer: Converts raw tool execution outputs into executive-level natural language answers.
Includes an automated Numerical Faithfulness Guard to catch any hallucinated numbers.
"""

import re
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

from agent.schema import ToolType, RouterDecision, ToolExecutionResult
from agent.llm import LLMClient, default_llm_client
from agent.config import config


SYNTHESIZER_SYSTEM_PROMPT = """You are the Senior Data Storyteller for the AI Data Analyst Agent.
Your job is to take the structured data and execution result from a data tool and deliver a clear, concise, and executive-ready natural language answer.

Guidelines:
1. Direct Answer First: State the bottom-line answer in the very first sentence.
2. Grounded in Data: Use ONLY the exact numbers, percentages, and metrics provided in the Tool Result. DO NOT invent or estimate numbers.
3. Clean Formatting: Use bold text for key figures, bullet points for multi-item results, and markdown tables if showing top 3-5 rows.
4. Business Context: Highlight interesting patterns (e.g. margin trends, anomalous categories) when evident from the data.
5. If the tool is `clarify`, politely explain why the question needs refinement and guide the user on what to provide.
"""


class ResponseSynthesizer:
    """Synthesizes narrative data answers and enforces numerical faithfulness against raw tool output."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initializes the response synthesizer with an LLM client.
        
        Args:
            llm_client: Optional custom LLMClient instance for natural language synthesis.
        """
        self.llm = llm_client or default_llm_client

    def synthesize(
        self,
        user_query: str,
        decision: RouterDecision,
        tool_result: ToolExecutionResult
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Synthesizes narrative response and runs the numerical faithfulness verification.
        Returns: (narrative_response, validation_passed, validation_notes)
        """
        # If tool execution failed, format a clear error explanation
        if not tool_result.success:
            err_msg = (
                f"I encountered an issue executing the requested {decision.tool.value} operation.\n\n"
                f"**Error Details**: `{tool_result.error}`\n\n"
                f"**Planned Action**: {decision.reasoning}"
            )
            return err_msg, True, "Tool execution failed; error reported directly."

        # If clarify tool, format the structured clarification directly
        if decision.tool == ToolType.CLARIFY:
            cdata = tool_result.data or {}
            msg = cdata.get("clarification_message") or str(tool_result.data)
            return msg, True, "Clarification handled without numerical synthesis."

        # Fast-Path: Deterministic schema metadata response
        if decision.intent == "dataset_schema_lookup" and tool_result.data:
            col_stats = tool_result.data.get("column_stats", {})
            total_records = tool_result.row_count or max([info.get("total_count", 0) for info in col_stats.values()], default=0)
            table_md = "| Column | Semantic Type | Unique Values | Status |\n| :--- | :--- | :--- | :--- |\n"
            for col, info in col_stats.items():
                ctype = info.get("type", "string")
                uniques = info.get("unique_count", info.get("total_count", 0))
                nulls = info.get("null_count", 0)
                status = f"⚠️ {nulls} nulls" if nulls > 0 else "✅ Clean"
                table_md += f"| `{col}` | `{ctype}` | {uniques:,} | {status} |\n"
            
            resp = (
                f"The dataset contains **{len(col_stats)} columns** across **{total_records:,} records**:\n\n"
                f"{table_md}\n\n"
                f"You can ask me to calculate aggregations, plot visual trends (bar, line, scatter, box, histogram), or analyze column distributions across any of these dimensions."
            )
            return resp, True, "Schema metadata returned directly from in-memory profiler."

        # Format tool data for synthesizer prompt
        data_summary_str = self._format_data_for_prompt(tool_result)

        user_prompt = (
            f"User Question: \"{user_query}\"\n"
            f"Tool Executed: {decision.tool.value}\n"
            f"Router Intent & Reasoning: {decision.reasoning}\n\n"
            f"=== RAW TOOL EXECUTION DATA ===\n"
            f"{data_summary_str}\n"
            f"===============================\n\n"
            f"Please write a concise, professional answer directly answering the question based strictly on this data."
        )

        narrative = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )

        # Run numerical faithfulness guard
        if config.enable_faithfulness_check:
            passed, notes = self.verify_numerical_faithfulness(narrative, tool_result)
        else:
            passed, notes = True, "Faithfulness check disabled in config."

        return narrative, passed, notes

    def _format_data_for_prompt(self, tool_result: ToolExecutionResult) -> str:
        """Serializes tool execution result data into a compact text block for the prompt."""
        data = tool_result.data
        if not data:
            return "No records returned (empty result)."

        if tool_result.tool == ToolType.QUERY_DATA:
            rows = data.get("rows", [])
            cols = data.get("columns", [])
            row_count = data.get("row_count", len(rows))
            
            if not rows:
                return "Query returned 0 rows."
            
            preview = rows[:15]
            preview_df = pd.DataFrame(preview)
            return f"Query returned {row_count} total rows. Preview of top rows:\n{preview_df.to_string(index=False)}"

        elif tool_result.tool == ToolType.PLOT_CHART:
            spec = data.get("chart_spec", {})
            summary = data.get("summary_table", [])
            summary_df = pd.DataFrame(summary) if summary else "No tabular summary"
            return (
                f"Generated {spec.get('chart_type')} chart titled '{spec.get('title')}'.\n"
                f"Underlying aggregated data preview:\n{summary_df.to_string(index=False) if isinstance(summary_df, pd.DataFrame) else summary_df}"
            )

        elif tool_result.tool == ToolType.SUMMARY_STATS:
            import json
            return json.dumps(data, indent=2)

        return str(data)

    def verify_numerical_faithfulness(
        self,
        narrative: str,
        tool_result: ToolExecutionResult
    ) -> Tuple[bool, str]:
        """
        Extracts numbers from narrative and checks if they appear in or are derived from the raw data.
        Supports currency formatting ($), percentages (%), negative values, and scaled metrics (K/M).
        
        Returns: (passed: bool, explanation: str)
        """
        # Extract candidate numbers from narrative: matches numbers like 1,234.56, 45.2%, $9,167,421.88, 7500, 3059.41, -132.51
        number_pattern = r"[-+]?\$?\b(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?\b"
        tokens = re.findall(number_pattern, narrative)
        
        # Clean extracted numbers into standard float values
        narrative_numbers = []
        for t in tokens:
            cleaned = t.replace("$", "").replace("%", "").replace(",", "").strip()
            try:
                val = float(cleaned)
                narrative_numbers.append((t, val))
            except ValueError:
                continue

        if not narrative_numbers:
            return True, "No numerical claims found in narrative to verify."

        # Collect raw tool numbers
        raw_numbers = self._extract_numbers_from_tool_result(tool_result)

        # Context allowlist: ordinal ranks, small bullet counts, and calendar years
        context_allowlist = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 2020, 2021, 2022, 2023, 2024, 2025, 2026}

        unmatched = []
        for token_str, n in narrative_numbers:
            if n in context_allowlist:
                continue

            matched = False
            for r in raw_numbers:
                # 1. Direct match with relative or absolute tolerance
                if abs(n - r) < 0.05 or (r != 0 and abs(n - r) / abs(r) < 0.02):
                    matched = True
                    break
                # 2. Percentage representation (e.g. raw 0.155 -> narrative 15.5%)
                if abs(n - (r * 100.0)) < 0.1 or (r != 0 and abs(n - (r * 100.0)) / abs(r * 100.0) < 0.02):
                    matched = True
                    break
                # 3. Scaled representations (e.g. raw 2,179,369.32 -> narrative 2.18M)
                if r > 1000:
                    if abs(n - (r / 1e6)) < 0.05 or abs(n - round(r / 1e6, 2)) < 0.01:
                        matched = True
                        break
                    if abs(n - (r / 1e3)) < 0.05 or abs(n - round(r / 1e3, 2)) < 0.01:
                        matched = True
                        break

            if not matched:
                unmatched.append(f"'{token_str}' ({n})")

        if unmatched:
            notes = f"Warning: Found ungrounded numerical claims in narrative: {', '.join(unmatched[:4])}"
            return False, notes

        return True, "Numerical faithfulness validated: all narrative figures match tool output."

    def _extract_numbers_from_tool_result(self, tool_result: Any) -> List[float]:
        """Recursively pulls all numerical values from tool result data."""
        nums = []
        if tool_result is None:
            return nums
        if isinstance(tool_result, ToolExecutionResult):
            data = tool_result.data
        elif isinstance(tool_result, dict) and "data" in tool_result:
            data = tool_result["data"]
        else:
            data = tool_result
        if not data:
            return nums

        def _traverse(obj: Any) -> None:
            """Recursively traverses nested dictionaries/lists to extract raw floats."""
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                nums.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    _traverse(v)
            elif isinstance(obj, list):
                for item in obj:
                    _traverse(item)

        _traverse(data)
        return nums
