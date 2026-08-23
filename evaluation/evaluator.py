"""
Automated Evaluation Harness for the AI Data Analyst Agent.
Runs the 20-question benchmark, calculates tool-selection metrics (Precision, Recall, F1),
evaluates answer correctness, and produces structured evaluation reports.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import duckdb

from agent.pipeline import DataAnalystAgent
from agent.schema import ToolType, AgentTrace
from agent.config import config


def compare_query_results(
    actual_df: Optional[pd.DataFrame],
    expected_df: Optional[pd.DataFrame],
    rtol: float = 1e-2,
    atol: float = 1e-2,
    check_order: bool = False
) -> Tuple[bool, str]:
    """Compares two analytical query result DataFrames for value-level numerical correctness.
    
    Args:
        actual_df: DataFrame returned by the agent's executed SQL query.
        expected_df: DataFrame returned by the ground-truth reference SQL query.
        rtol: Relative tolerance for floating-point numerical comparisons.
        atol: Absolute tolerance for floating-point / monetary comparisons ($0.01 default).
        check_order: If True, requires exact row ordering; otherwise sorts before comparison.
        
    Returns:
        Tuple of (is_correct: bool, explanation: str).
    """
    if actual_df is None or expected_df is None:
        return False, "One of the result DataFrames is None."

    if len(actual_df) != len(expected_df):
        return False, f"Row count mismatch: expected {len(expected_df)} rows, got {len(actual_df)} rows."

    if len(actual_df) == 0 and len(expected_df) == 0:
        return True, "Both queries returned 0 rows (matching empty result)."

    if len(actual_df.columns) != len(expected_df.columns):
        return False, f"Column count mismatch: expected {len(expected_df.columns)} columns, got {len(actual_df.columns)}."

    act = actual_df.copy()
    exp = expected_df.copy()

    # Normalize column names by position to handle alias variations (e.g. sum_sales vs total_sales)
    act.columns = [f"col_{i}" for i in range(len(act.columns))]
    exp.columns = [f"col_{i}" for i in range(len(exp.columns))]

    # Handle row ordering: if order is not strictly mandated, sort both DataFrames consistently
    if not check_order and len(act) > 1:
        sort_cols = list(act.columns)
        try:
            act = act.sort_values(by=sort_cols).reset_index(drop=True)
            exp = exp.sort_values(by=sort_cols).reset_index(drop=True)
        except Exception:
            act = act.reset_index(drop=True)
            exp = exp.reset_index(drop=True)
    else:
        act = act.reset_index(drop=True)
        exp = exp.reset_index(drop=True)

    # Cell-by-cell numerical and categorical comparison
    for col in act.columns:
        s_act = act[col]
        s_exp = exp[col]

        for i in range(len(act)):
            v_act = s_act.iloc[i]
            v_exp = s_exp.iloc[i]

            # Null handling
            if pd.isna(v_act) and pd.isna(v_exp):
                continue
            if pd.isna(v_act) != pd.isna(v_exp):
                return False, f"Null mismatch at row {i}, column '{col}': actual={v_act}, expected={v_exp}."

            # Numeric comparison with relative, absolute, and ratio-to-percentage scale tolerance
            if isinstance(v_act, (int, float, np.number)) and isinstance(v_exp, (int, float, np.number)):
                f_act = float(v_act)
                f_exp = float(v_exp)

                direct_match = np.isclose(f_act, f_exp, rtol=rtol, atol=atol)
                # Ratio vs Percentage tolerance (e.g. 0.15 vs 15.0 or 0.14 vs 14.0)
                pct_match_1 = np.isclose(f_act * 100.0, f_exp, rtol=rtol, atol=atol * 100.0)
                pct_match_2 = np.isclose(f_act, f_exp * 100.0, rtol=rtol, atol=atol * 100.0)

                if not (direct_match or pct_match_1 or pct_match_2):
                    return False, f"Value mismatch at row {i}, column '{col}': actual={v_act}, expected={v_exp} (tolerance atol={atol}, rtol={rtol})."
            else:
                str_act = str(v_act).strip().lower()
                str_exp = str(v_exp).strip().lower()
                if str_act != str_exp:
                    return False, f"Value mismatch at row {i}, column '{col}': actual='{v_act}', expected='{v_exp}'."

    return True, f"All {len(act)} rows and {len(act.columns)} columns matched ground truth within numerical tolerance."


class AgentEvaluator:
    """Evaluates DataAnalystAgent against curated ground-truth benchmark questions."""

    def __init__(
        self,
        benchmark_file: str = "evaluation/benchmark_dataset.json",
        dataset_path: Optional[str] = None
    ):
        """Initializes the benchmark harness with ground-truth test cases and DuckDB verifier.
        
        Args:
            benchmark_file: Path to JSON ground-truth benchmark suite.
            dataset_path: Path to dataset CSV file.
        """
        self.benchmark_file = benchmark_file
        self.dataset_path = dataset_path or config.dataset_path
        
        with open(benchmark_file, "r", encoding="utf-8") as f:
            self.benchmark_cases = json.load(f)
            
        self.agent = DataAnalystAgent(dataset_path_or_df=self.dataset_path)
        self.db_con = duckdb.connect(database=":memory:")
        self.db_con.register(config.table_name, pd.read_csv(self.dataset_path))

    def run_benchmark(self, verbose: bool = True) -> Dict[str, Any]:
        """Runs all benchmark test cases and computes performance metrics."""
        results = []
        tool_confusion: Dict[str, Dict[str, int]] = {
            t.value: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            for t in ToolType
        }
        
        total_cases = len(self.benchmark_cases)
        correct_tool_selections = 0
        successful_executions = 0
        correct_answers = 0
        faithful_narratives = 0
        total_latency_ms = 0.0

        if verbose:
            print(f"Starting Evaluation Benchmark ({total_cases} test cases)...")
            print("=" * 70)

        for idx, tc in enumerate(self.benchmark_cases, 1):
            tc_id = tc["id"]
            question = tc["question"]
            expected_tool = tc["expected_tool"]
            category = tc.get("category", "general")
            gt_sql = tc.get("ground_truth_sql")

            if verbose:
                print(f"[{idx}/{total_cases}] ({tc_id}) '{question}'")

            # Brief pause between calls
            if idx > 1:
                time.sleep(0.2)

            start_t = time.time()
            try:
                trace: AgentTrace = self.agent.ask(question)
                actual_tool = trace.router_decision.tool.value
                exec_success = trace.tool_result.success
                faith_passed = trace.numerical_validation_passed
                total_latency_ms += trace.total_latency_ms
            except Exception as err:
                actual_tool = "error"
                exec_success = False
                faith_passed = False
                trace = None
                if verbose:
                    print(f"   -> EXCEPTION: {err}")

            # Tool Selection Accuracy
            tool_match = (actual_tool == expected_tool)
            if tool_match:
                correct_tool_selections += 1
            if exec_success:
                successful_executions += 1
            if faith_passed:
                faithful_narratives += 1

            # Update Confusion Matrix per tool
            for t in ToolType:
                t_val = t.value
                if expected_tool == t_val and actual_tool == t_val:
                    tool_confusion[t_val]["TP"] += 1
                elif expected_tool != t_val and actual_tool == t_val:
                    tool_confusion[t_val]["FP"] += 1
                elif expected_tool == t_val and actual_tool != t_val:
                    tool_confusion[t_val]["FN"] += 1
                else:
                    tool_confusion[t_val]["TN"] += 1

            # Answer Correctness Evaluation
            answer_correct = False
            validation_details = ""

            if tool_match and exec_success and trace:
                if expected_tool == "query_data" and gt_sql:
                    # Validate SQL result against ground truth DuckDB query via value-level cell comparison
                    try:
                        gt_df = self.db_con.execute(gt_sql).fetchdf()
                        actual_rows = trace.tool_result.data.get("rows", [])
                        actual_df = pd.DataFrame(actual_rows) if actual_rows else pd.DataFrame()
                        
                        # Determine if query requires semantic ranking/ordering (e.g. top, highest, lowest, limit)
                        q_lower = question.lower()
                        is_ranking_query = any(k in q_lower for k in ["top", "highest", "lowest", "rank", "bottom", "first"]) or ("LIMIT" in gt_sql.upper())
                        
                        is_correct, details = compare_query_results(
                            actual_df=actual_df,
                            expected_df=gt_df,
                            check_order=is_ranking_query
                        )
                        answer_correct = is_correct
                        validation_details = details
                    except Exception as sql_err:
                        answer_correct = False
                        validation_details = f"Ground truth validation error: {sql_err}"
                
                elif expected_tool == "plot_chart":
                    chart_spec = trace.tool_result.data.get("chart_spec", {})
                    if chart_spec.get("chart_type") and trace.tool_result.data.get("figure"):
                        answer_correct = True
                        validation_details = f"Valid Plotly {chart_spec.get('chart_type')} chart rendered successfully."
                
                elif expected_tool == "summary_stats":
                    stats_data = trace.tool_result.data or {}
                    if "column_stats" in stats_data or "grouped_stats" in stats_data:
                        answer_correct = True
                        validation_details = "Descriptive statistics computed successfully."
                
                elif expected_tool == "clarify":
                    c_data = trace.tool_result.data or {}
                    if c_data.get("ambiguity_reason") and c_data.get("suggested_actions"):
                        answer_correct = True
                        validation_details = "Clarification prompt formulated with actionable alternatives."

            if answer_correct:
                correct_answers += 1

            case_result = {
                "id": tc_id,
                "category": category,
                "question": question,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "tool_match": tool_match,
                "execution_success": exec_success,
                "answer_correct": answer_correct,
                "numerical_faithfulness": faith_passed,
                "router_reasoning": trace.router_decision.reasoning if trace else "",
                "narrative_response": trace.narrative_response if trace else "",
                "validation_details": validation_details,
                "latency_ms": trace.total_latency_ms if trace else 0.0
            }
            results.append(case_result)

            if verbose:
                status_tool = "PASS" if tool_match else f"FAIL (got {actual_tool})"
                status_ans = "PASS" if answer_correct else "FAIL"
                print(f"   -> Tool: [{status_tool}] | Answer: [{status_ans}] | Latency: {case_result['latency_ms']}ms\n")

        # Compute Classification Metrics per Tool
        tool_metrics = {}
        for t_val, counts in tool_confusion.items():
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
            tool_metrics[t_val] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn
            }

        summary_metrics = {
            "total_questions": total_cases,
            "tool_selection_accuracy_pct": round((correct_tool_selections / total_cases) * 100, 2),
            "execution_success_rate_pct": round((successful_executions / total_cases) * 100, 2),
            "answer_correctness_rate_pct": round((correct_answers / total_cases) * 100, 2),
            "numerical_faithfulness_pct": round((faithful_narratives / total_cases) * 100, 2),
            "average_latency_ms": round(total_latency_ms / total_cases, 2),
            "tool_metrics": tool_metrics,
            "results": results
        }

        # Save results to evaluation/benchmark_results.json
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        return summary_metrics

    def generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generates a clean markdown report of the benchmark evaluation."""
        lines = [
            "# AI Data Analyst Agent — Benchmark Evaluation Report",
            f"\n**Total Test Cases**: {summary['total_questions']}",
            f"- **Tool Selection Accuracy**: **{summary['tool_selection_accuracy_pct']}%**",
            f"- **Tool Execution Success Rate**: **{summary['execution_success_rate_pct']}%**",
            f"- **Answer Correctness Rate**: **{summary['answer_correctness_rate_pct']}%**",
            f"- **Numerical Faithfulness Guard**: **{summary['numerical_faithfulness_pct']}%**",
            f"- **Average Query Latency**: **{summary['average_latency_ms']} ms**\n",
            "## Tool Classification Metrics (Precision / Recall / F1)",
            "| Tool | Precision | Recall | F1-Score | Support |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for tool_name, m in summary["tool_metrics"].items():
            lines.append(
                f"| `{tool_name}` | {m['precision']:.2f} | {m['recall']:.2f} | **{m['f1_score']:.2f}** | {m['support']} |"
            )

        lines.append("\n## Detailed Test Case Results")
        lines.append("| ID | Category | Question | Expected Tool | Actual Tool | Tool Match | Answer Correct |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for r in summary["results"]:
            tm_icon = "✅" if r["tool_match"] else "❌"
            ac_icon = "✅" if r["answer_correct"] else "❌"
            q_short = r["question"][:45] + ("..." if len(r["question"]) > 45 else "")
            lines.append(
                f"| {r['id']} | `{r['category']}` | {q_short} | `{r['expected_tool']}` | `{r['actual_tool']}` | {tm_icon} | {ac_icon} |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    evaluator = AgentEvaluator()
    summary = evaluator.run_benchmark(verbose=True)
    md_report = evaluator.generate_markdown_report(summary)
    with open("evaluation/benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    print("\n" + "=" * 70)
    try:
        print(md_report)
    except Exception:
        print("Benchmark Summary saved to evaluation/benchmark_summary.md")
