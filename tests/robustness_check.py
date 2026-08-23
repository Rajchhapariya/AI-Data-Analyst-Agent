import os
import sys
import time
from typing import List, Dict, Any

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure workspace root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from agent.pipeline import DataAnalystAgent
from agent.schema import ToolType


# ==============================================================================
# 1. Adversarial & Paraphrased Single-Turn Test Cases
# ==============================================================================
TEST_CASES = [
    # Schema / Metadata Questions
    {
        "category": "Schema / Metadata",
        "question": "What columns are in this data set",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },
    {
        "category": "Schema / Metadata",
        "question": "what columns are in this dataset",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },
    {
        "category": "Schema / Metadata",
        "question": "list the columns",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },
    {
        "category": "Schema / Metadata",
        "question": "what fields exist in the data",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },
    {
        "category": "Schema / Metadata",
        "question": "show me the dataset structure",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },
    {
        "category": "Schema / Metadata",
        "question": "what data do you have access to",
        "expected_tool": "summary_stats",
        "expected_path": "fast-path",
        "expected_behavior": "Returns instant in-memory schema table with zero LLM calls"
    },

    # Dimensional Ambiguity (Both metric and timeframe missing)
    {
        "category": "Dimensional Ambiguity",
        "question": "how are we doing lately",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts execution with clarify due to dual missing dimensions"
    },
    {
        "category": "Dimensional Ambiguity",
        "question": "what's trending",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts execution with clarify due to dual missing dimensions"
    },
    {
        "category": "Dimensional Ambiguity",
        "question": "give me an update",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts execution with clarify due to dual missing dimensions"
    },

    # Single-Dimension Ambiguity / Semi-Specified
    {
        "category": "Single Dimension",
        "question": "show me profit",
        "expected_tool": "query_data",
        "expected_path": "router-llm",
        "expected_behavior": "Executes aggregation (metric present, full dataset timeframe assumed)"
    },
    {
        "category": "Single Dimension",
        "question": "total sales in 2023",
        "expected_tool": "query_data",
        "expected_path": "router-llm",
        "expected_behavior": "Executes SQL filtering for sales in 2023"
    },
    {
        "category": "Single Dimension",
        "question": "compare this year to last year",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts with clarify because timeframe is specified but metric is missing"
    },

    # Out-of-Scope Topics
    {
        "category": "Out-of-Scope",
        "question": "what's the weather like",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts with clarify identifying out-of-scope domain"
    },
    {
        "category": "Out-of-Scope",
        "question": "what's the stock price of Apple",
        "expected_tool": "clarify",
        "expected_path": "router-llm",
        "expected_behavior": "Halts with clarify identifying out-of-scope domain"
    },

    # Specific Deterministic Queries
    {
        "category": "Specific Query",
        "question": "total profit for Q1 2023",
        "expected_tool": "query_data",
        "expected_path": "router-llm",
        "expected_behavior": "Executes SQL aggregation for 2023-01-01 to 2023-03-31 profit"
    },
    {
        "category": "Specific Query",
        "question": "average discount by region",
        "expected_tool": "query_data",
        "expected_path": "router-llm",
        "expected_behavior": "Executes SQL grouping by region and averaging discount"
    },
    {
        "category": "Specific Query",
        "question": "plot sales by category as a bar chart",
        "expected_tool": "plot_chart",
        "expected_path": "router-llm",
        "expected_behavior": "Executes PlotChartTool generating bar chart of sales by category"
    }
]


# ==============================================================================
# 2. Multi-Turn Stateless Test Sequences
# ==============================================================================
MULTI_TURN_SEQUENCES = [
    {
        "name": "Sequential Metric Ellipsis",
        "turn_1": {
            "query": "What was total profit in 2024?",
            "expected_tool": "query_data",
            "stateless_note": "Turn 1: Baseline analytical lookup"
        },
        "turn_2": {
            "query": "What about 2023?",
            "expected_tool": "clarify",
            "stateless_note": "Turn 2: Ellipsis lacks metric; correctly halts with clarify without state leakage"
        }
    },
    {
        "name": "Pronoun Reference ('it')",
        "turn_1": {
            "query": "Show me total sales by region",
            "expected_tool": "query_data",
            "stateless_note": "Turn 1: Baseline regional aggregation"
        },
        "turn_2": {
            "query": "Now show it as a chart instead",
            "expected_tool": "clarify",
            "stateless_note": "Turn 2: 'it' has no antecedent in stateless architecture; correctly halts with clarify"
        }
    },
    {
        "name": "Deterministic Idempotency",
        "turn_1": {
            "query": "total profit for Q1 2023",
            "expected_tool": "query_data",
            "stateless_note": "Turn 1: First SQL execution"
        },
        "turn_2": {
            "query": "total profit for Q1 2023",
            "expected_tool": "query_data",
            "stateless_note": "Turn 2: Identical repeated query; yields deterministic identical SQL"
        }
    }
]


# ==============================================================================
# 3. Multi-Hop / Compound Reasoning Queries
# ==============================================================================
COMPOUND_QUERIES = [
    {
        "id": "MH-01",
        "question": "Which region had the highest profit margin, and how does that compare to their sales volume rank?",
        "expected_parts": "Part 1: Region with max(profit/sales) | Part 2: Regional sales volume rank comparison"
    },
    {
        "id": "MH-02",
        "question": "What's the difference between our top and bottom performing categories by profit?",
        "expected_parts": "Part 1: Max profit category | Part 2: Min profit category | Part 3: Numerical difference"
    },
    {
        "id": "MH-03",
        "question": "Show me the region with the highest sales and tell me its average discount rate",
        "expected_parts": "Part 1: Region with max sales | Part 2: Average discount rate for that region"
    }
]


def run_robustness_test(agent: DataAnalystAgent, console: Console) -> bool:
    console.print("\n[bold cyan]🔬 Part 1: Adversarial & Paraphrased Input Robustness Matrix[/bold cyan]\n")

    total_passed = 0
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=18)
    table.add_column("User Question", style="white", width=34)
    table.add_column("Code Path", style="yellow", width=12)
    table.add_column("Tool Selected", style="blue", width=15)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Behavior Note", style="green", width=32)

    for tc in TEST_CASES:
        q = tc["question"]
        expected_tool = tc["expected_tool"]
        expected_path = tc["expected_path"]

        t0 = time.time()
        trace = agent.ask(q)
        latency_ms = round((time.time() - t0) * 1000, 1)

        actual_tool = trace.router_decision.tool.value
        actual_path = "fast-path" if trace.router_decision.intent == "dataset_schema_lookup" else "router-llm"

        tool_ok = (actual_tool == expected_tool)
        path_ok = (actual_path == expected_path)
        execution_ok = trace.tool_result.success

        is_pass = tool_ok and path_ok and execution_ok
        if is_pass:
            total_passed += 1
            status_str = "[bold green]PASS[/bold green]"
        else:
            status_str = "[bold red]FAIL[/bold red]"

        if actual_tool == "clarify":
            reason_snip = trace.router_decision.parameters.get("reason", "Clarification")
            if isinstance(reason_snip, list):
                reason_snip = "; ".join(reason_snip)
            note = f"Clarify: {reason_snip[:30]}..."
        elif actual_path == "fast-path":
            note = "Direct in-memory schema table"
        else:
            note = f"Executed {actual_tool} cleanly"

        table.add_row(
            tc["category"],
            f'"{q}"',
            actual_path,
            actual_tool,
            f"{latency_ms} ms",
            status_str,
            note
        )

    console.print(table)
    total_cases = len(TEST_CASES)
    pct = round((total_passed / total_cases) * 100, 1)
    console.print(f"\n[bold]Part 1 Summary[/bold]: [bold green]{total_passed}/{total_cases}[/bold green] test cases passed ([bold cyan]{pct}%[/bold cyan])\n")
    return total_passed == total_cases


def run_multiturn_stateless_check(agent: DataAnalystAgent, console: Console) -> bool:
    console.print("\n[bold cyan]🔄 Part 2: Multi-Turn Stateless Boundary & Zero-State-Leakage Suite[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sequence Name", style="cyan", width=22)
    table.add_column("Turn", justify="center", style="white", width=6)
    table.add_column("User Query", style="white", width=30)
    table.add_column("Tool", style="blue", width=14)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Stateless Status", justify="center", width=16)
    table.add_column("Architectural Verification Note", style="green", width=36)

    all_passed = True

    for seq in MULTI_TURN_SEQUENCES:
        name = seq["name"]
        
        # Turn 1
        t1 = seq["turn_1"]
        t0 = time.time()
        trace_1 = agent.ask(t1["query"])
        lat_1 = round((time.time() - t0) * 1000, 1)
        tool_1 = trace_1.router_decision.tool.value
        t1_ok = (tool_1 == t1["expected_tool"]) and trace_1.tool_result.success

        table.add_row(
            name,
            "1",
            f'"{t1["query"]}"',
            tool_1,
            f"{lat_1} ms",
            "[bold green]BASELINE OK[/bold green]" if t1_ok else "[bold red]FAIL[/bold red]",
            t1["stateless_note"]
        )

        # Turn 2 on the EXACT SAME agent instance
        t2 = seq["turn_2"]
        t0 = time.time()
        trace_2 = agent.ask(t2["query"])
        lat_2 = round((time.time() - t0) * 1000, 1)
        tool_2 = trace_2.router_decision.tool.value
        t2_ok = (tool_2 == t2["expected_tool"]) and trace_2.tool_result.success

        if not (t1_ok and t2_ok):
            all_passed = False

        status_str = "[bold green]STATELESS PASS[/bold green]" if t2_ok else "[bold red]STATE LEAK FAIL[/bold red]"

        table.add_row(
            "",
            "2",
            f'"{t2["query"]}"',
            tool_2,
            f"{lat_2} ms",
            status_str,
            t2["stateless_note"]
        )

    console.print(table)
    console.print(f"\n[bold]Part 2 Summary[/bold]: [bold green]{'All 3 multi-turn stateless boundary checks PASSED' if all_passed else 'Stateless check FAILED'}[/bold green]\n")
    return all_passed


def run_compound_reasoning_check(agent: DataAnalystAgent, console: Console) -> None:
    console.print("\n[bold cyan]🧩 Part 3: Multi-Hop / Compound Reasoning Architectural Audit[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Compound Question", style="white", width=30)
    table.add_column("Tool", style="blue", width=12)
    table.add_column("SQL / Reason Payload", style="yellow", width=28)
    table.add_column("Resolution Scope", style="magenta", width=18)
    table.add_column("Safety Mode Classification", style="green", width=30)

    for item in COMPOUND_QUERIES:
        q = item["question"]
        trace = agent.ask(q)

        tool = trace.router_decision.tool.value
        params = trace.router_decision.parameters

        if tool == "query_data":
            payload = params.get("query", "")
            if trace.tool_result.success:
                resolution_scope = "Both Hops in 1 SQL"
                safety_mode = "[bold cyan](d) Full Single-Pass Resolution[/bold cyan]"
            else:
                resolution_scope = "SQL Syntax Error"
                safety_mode = "[bold green](b) Reports Error Honestly[/bold green]"
        elif tool == "clarify":
            reason = params.get("reason", "")
            if isinstance(reason, list):
                reason = "; ".join(reason)
            payload = f"Clarify: {reason[:30]}..."
            resolution_scope = "Dispatched Clarify"
            safety_mode = "[bold green](a) Declines/Clarifies Gracefully[/bold green]"
        else:
            payload = str(params)[:30]
            resolution_scope = "Partial"
            safety_mode = "[bold red](c) Hallucination Risk[/bold red]"

        table.add_row(
            item["id"],
            f'"{q}"',
            tool,
            payload[:26] + "..." if len(payload) > 26 else payload,
            resolution_scope,
            safety_mode
        )

    console.print(table)


if __name__ == "__main__":
    console = Console()
    agent = DataAnalystAgent()

    p1_ok = run_robustness_test(agent, console)
    p2_ok = run_multiturn_stateless_check(agent, console)
    run_compound_reasoning_check(agent, console)

    sys.exit(0 if (p1_ok and p2_ok) else 1)
