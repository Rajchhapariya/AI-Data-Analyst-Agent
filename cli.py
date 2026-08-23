"""
AI Data Analyst Agent — Rich Terminal CLI
An interactive, high-performance CLI for exploring data, querying the agent,
inspecting step-by-step reasoning traces, and running benchmark evaluations.
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent.config import config
from agent.pipeline import DataAnalystAgent
from agent.profiler import DatasetProfiler
from agent.schema import AgentTrace, ToolType
from evaluation.evaluator import AgentEvaluator

console = Console()


def print_banner():
    """Renders a sleek ASCII hero banner."""
    banner_text = """
 [bold cyan]╔═══════════════════════════════════════════════════════════════════╗[/bold cyan]
 [bold cyan]║[/bold cyan]            [bold white]AI DATA ANALYST AGENT — INTERACTIVE CLI[/bold white]               [bold cyan]║[/bold cyan]
 [bold cyan]║[/bold cyan]   [dim]Autonomous BI • Constrained Safe Tools • Inspectable Reasoning[/dim]  [bold cyan]║[/bold cyan]
 [bold cyan]╚═══════════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner_text, style="bold cyan")
    console.print(f" [dim]Model: [bold green]{config.default_model}[/bold green] | Dataset: [bold yellow]{Path(config.dataset_path).name}[/bold yellow] | Engine: [bold blue]DuckDB SQL[/bold blue][/dim]\n")


def display_help():
    """Displays command cheat-sheet."""
    table = Table(title="Available Commands", border_style="cyan", show_header=True)
    table.add_column("Command", style="bold yellow", width=18)
    table.add_column("Description", style="white")
    
    table.add_row("<Any Question>", "Ask a natural-language question about the dataset")
    table.add_row(":profile", "Display dataset schema, row counts, memory, and null audit")
    table.add_row(":eval", "Run the automated 20-question Ground Truth benchmark suite")
    table.add_row(":examples", "List suggested questions covering queries, charts & stats")
    table.add_row(":clear", "Clear terminal screen")
    table.add_row(":help", "Show this help table")
    table.add_row(":exit / :quit", "Exit the application")
    console.print(table)
    console.print()


def display_examples():
    """Displays curated sample prompts."""
    examples = [
        ("Query Lookup", "What is the total sales amount across all orders?"),
        ("Aggregation", "What are the top 5 sub-categories by total sales volume?"),
        ("Comparison", "Compare total sales and profit across all 4 regions."),
        ("Plot Chart", "Can you plot a bar chart of total sales by product category?"),
        ("Scatter Plot", "Create a scatter plot comparing sales versus profit."),
        ("Summary Stats", "What are the summary statistics and quartiles for sales and profit?"),
        ("Ambiguity Guard", "Show me our best products."),
        ("Out-of-Scope Guard", "What will the weather be in Seattle next week?")
    ]
    table = Table(title="Sample Prompts to Try", border_style="magenta", show_header=True)
    table.add_column("Category", style="bold cyan", width=20)
    table.add_column("Prompt", style="white")
    for cat, p in examples:
        table.add_row(cat, p)
    console.print(table)
    console.print()


def display_profile(profiler: DatasetProfiler):
    """Renders formatted dataset quality profile."""
    prof = profiler.profile()
    
    # Overview table
    overview = Table(title="Dataset Summary", border_style="green")
    overview.add_column("Metric", style="bold cyan")
    overview.add_column("Value", style="bold white")
    overview.add_row("Total Transactions", f"{prof.total_rows:,}")
    overview.add_row("Total Columns", str(len(prof.columns)))
    overview.add_row("In-Memory Footprint", f"{prof.memory_usage_mb:.2f} MB")
    overview.add_row("Table Name in DuckDB", config.table_name)
    console.print(overview)
    console.print()

    # Columns & Nulls Table
    schema_table = Table(title="Schema & Missing Values Audit", border_style="blue")
    schema_table.add_column("Column", style="bold yellow")
    schema_table.add_column("Data Type", style="cyan")
    schema_table.add_column("Unique Count", style="white", justify="right")
    schema_table.add_column("Null Count", style="white", justify="right")
    schema_table.add_column("Quality Status", style="green")

    for col, dtype in prof.columns.items():
        nulls = prof.null_counts.get(col, 0)
        uniques = prof.unique_counts.get(col, 0)
        status = "[red]⚠️ Nulls Found[/red]" if nulls > 0 else "[green]✅ Clean[/green]"
        schema_table.add_row(col, dtype, f"{uniques:,}", str(nulls), status)

    console.print(schema_table)
    console.print()


def display_trace(trace: AgentTrace):
    """Renders the step-by-step reasoning trace and synthesized result."""
    tool_val = trace.router_decision.tool.value
    tool_colors = {
        "query_data": "blue",
        "plot_chart": "magenta",
        "summary_stats": "green",
        "clarify": "yellow"
    }
    color = tool_colors.get(tool_val, "cyan")

    # 1. Step-by-Step Reasoner Panel
    reasoning_content = (
        f"[bold white]Step 1: Router Planning & Reasoning[/bold white]\n"
        f"[dim]{trace.router_decision.reasoning}[/dim]\n\n"
        f"[bold white]Step 2: Selected Constrained Tool[/bold white]: [{color} bold]{tool_val}[/{color} bold]\n"
        f"[bold white]Tool Parameters[/bold white]: [dim]{trace.router_decision.parameters}[/dim]"
    )
    console.print(Panel(reasoning_content, title="🔍 Inspectable Execution Trace", border_style=color))

    # 2. If SQL query tool was used, format raw SQL and table
    if tool_val == "query_data":
        sql_query = trace.router_decision.parameters.get("sql_query", "")
        if sql_query:
            console.print("\n[bold cyan]⚡ Executed DuckDB SQL:[/bold cyan]")
            console.print(Syntax(sql_query, "sql", theme="monokai", line_numbers=False))

        rows = trace.tool_result.data.get("rows", []) if trace.tool_result.data else []
        if rows and len(rows) > 0:
            console.print(f"\n[bold cyan]📋 Raw Data Output ({len(rows)} rows):[/bold cyan]")
            cols = list(rows[0].keys())
            res_table = Table(border_style="dim", show_header=True)
            for c in cols:
                res_table.add_column(c, style="white")
            for r in rows[:10]:
                res_table.add_row(*[str(r.get(c, "")) for c in cols])
            if len(rows) > 10:
                res_table.add_row(*[f"... +{len(rows)-10} more rows" if i == 0 else "..." for i in range(len(cols))])
            console.print(res_table)

    elif tool_val == "plot_chart":
        spec = trace.tool_result.data.get("chart_spec", {}) if trace.tool_result.data else {}
        console.print(f"\n[bold magenta]📈 Rendered Plotly {spec.get('chart_type', 'chart').upper()}[/bold magenta]: [white]'{spec.get('title', '')}'[/white]")
        console.print("[dim](Interactive chart is viewable in the Streamlit Web App: `streamlit run app/streamlit_app.py`)[/dim]")

    # 3. Main Synthesized Answer
    console.print("\n" + "=" * 60)
    console.print(Panel(Markdown(trace.narrative_response), title="💡 Analyst Response", border_style="bold green"))

    # 4. Numerical Faithfulness Guard Badge
    if trace.numerical_validation_passed:
        console.print(" [bold green]🛡️ Numerical Faithfulness Guard: PASSED[/bold green] [dim](All cited numbers verified against tool output)[/dim]")
    else:
        console.print(f" [bold red]⚠️ Numerical Faithfulness Guard: FLAG[/bold red] [dim]({trace.numerical_validation_notes})[/dim]")

    console.print(f" [dim]⏱️ Total Latency: {trace.total_latency_ms} ms[/dim]\n")


def run_cli():
    """Main interactive REPL loop."""
    print_banner()
    
    with console.status("[bold green]Initializing Data Analyst Agent pipeline & profiler..."):
        agent = DataAnalystAgent()
        profiler = DatasetProfiler(config.dataset_path)
        
    console.print("[bold green]✓ Agent Ready![/bold green] Type your question below, or type [bold cyan]:help[/bold cyan] for options.\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]AgentQuery[/bold cyan]").strip()
            
            if not user_input:
                continue

            # Command routing
            cmd_lower = user_input.lower()
            if cmd_lower in (":exit", ":quit", "exit", "quit", "q"):
                console.print("\n[bold yellow]👋 Exiting AI Data Analyst Agent. Goodbye![/bold yellow]\n")
                break
                
            elif cmd_lower in (":help", "help", "h"):
                display_help()
                continue
                
            elif cmd_lower in (":examples", "examples"):
                display_examples()
                continue
                
            elif cmd_lower in (":profile", "profile", ":info"):
                display_profile(profiler)
                continue
                
            elif cmd_lower in (":clear", "clear", "cls"):
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue
                
            elif cmd_lower in (":eval", ":benchmark", "eval"):
                console.print("\n[bold cyan]Starting Automated Benchmark Suite (20 Questions)...[/bold cyan]\n")
                evaluator = AgentEvaluator()
                summary = evaluator.run_benchmark(verbose=True)
                md_rep = evaluator.generate_markdown_report(summary)
                with open("evaluation/benchmark_summary.md", "w", encoding="utf-8") as f:
                    f.write(md_rep)
                console.print("\n[bold green]✓ Benchmark Completed! Summary saved to evaluation/benchmark_summary.md[/bold green]\n")
                continue

            # Natural Language Agent Query
            with console.status(f"[bold green]Analyzing: '{user_input}'..."):
                trace = agent.ask(user_input)

            display_trace(trace)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Interrupted. Exiting...[/bold yellow]\n")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Error: {e}[/bold red]\n")


if __name__ == "__main__":
    run_cli()
