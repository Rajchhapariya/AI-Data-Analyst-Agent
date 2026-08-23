"""
AI Data Analyst Agent — Streamlit Web Application
An inspectable, reasoning-first data analytics interface built for graduate-level demonstration.
Displays the linear execution sequence: Question -> Router Reasoning -> Deterministic Tool -> Faithful Synthesis.
"""

import sys
import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add workspace root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.config import config
from agent.pipeline import DataAnalystAgent
from agent.profiler import DatasetProfiler
from agent.schema import ToolType, AgentTrace
from evaluation.evaluator import AgentEvaluator


# ==============================================================================
# Page Configuration & Minimalist Design System
# ==============================================================================
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Top Header */
    .app-header {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .arch-caption {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #334155;
    }

    /* Execution Step Cards */
    .step-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #131d2e;
        border: 1px solid #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    .reasoning-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    
    .reasoning-highlight {
        color: #38bdf8;
        font-weight: 600;
    }
    
    /* Disambiguation / Clarify State Card */
    .clarify-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .clarify-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fbbf24;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .clarify-body {
        font-size: 0.93rem;
        color: #fde68a;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-query { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #0284c7; }
    .badge-chart { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid #9333ea; }
    .badge-stats { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
    .badge-clarify { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }
    
    /* Numerical Faithfulness Banner */
    .guard-banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 14px;
    }
    .guard-pass {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        color: #6ee7b7;
    }
    .guard-fail {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }

    /* Hide ONLY Deploy button and unwanted footer, keeping sidebar toggle visible */
    .stDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        display: none !important;
    }
    
    /* Ensure Sidebar Collapsed Control (Arrow) is always visible and clickable */
    [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        color: #38bdf8 !important;
        z-index: 1000 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Helper: Export AgentTrace as Formatted Markdown Report
# ==============================================================================
def format_trace_as_markdown(trace: AgentTrace, dataset_name: str = "superstore_sales.csv") -> str:
    """Formats an AgentTrace into a clean, human-readable executive Markdown report."""
    tool_name = trace.router_decision.tool.value
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_lines = [
        f"# AI Data Analyst Agent — Analysis Report",
        f"",
        f"- **Generated At**: `{timestamp_str}`",
        f"- **Active Dataset**: `{dataset_name}`",
        f"- **Total Latency**: `{trace.total_latency_ms:,.1f} ms`",
        f"- **Tool Selected**: `{tool_name}`",
        f"- **Execution Status**: `{'SUCCESS' if trace.tool_result.success else 'FAILED'}`",
        f"- **Numerical Faithfulness Guard**: `{'PASSED' if trace.numerical_validation_passed else 'FLAGGED'}`",
        f"",
        f"---",
        f"",
        f"## 1. User Question",
        f"> {trace.query}",
        f"",
        f"## 2. Agent Router Decision & Reasoning",
        f"```text",
        trace.router_decision.reasoning,
        f"```",
        f"",
        f"**Intent**: `{trace.router_decision.intent}`  ",
        f"**Tool Parameters**:",
        f"```json",
        json.dumps(trace.router_decision.parameters, indent=2),
        f"```",
        f"",
        f"---",
        f"",
        f"## 3. Tool Execution Output"
    ]

    if tool_name == "clarify":
        md_lines.extend([
            f"### ⚠️ Clarification Required (Safety Guardrail Triggered)",
            f"- **Reason**: {trace.router_decision.parameters.get('reason', 'N/A')}",
            f"- **Missing Dimensions**: `{trace.router_decision.parameters.get('missing_information', 'N/A')}`",
            f"- **Suggested Alternatives**:",
            f"{trace.router_decision.parameters.get('suggested_clarification', 'N/A')}"
        ])
    elif tool_name == "query_data":
        query_sql = trace.router_decision.parameters.get("query", "")
        md_lines.extend([
            f"### Executed SQL Query",
            f"```sql",
            query_sql,
            f"```",
            f""
        ])
        if trace.tool_result.data and "rows" in trace.tool_result.data:
            rows = trace.tool_result.data["rows"]
            if rows:
                df_rows = pd.DataFrame(rows)
                md_lines.append(f"### Query Results ({len(rows)} rows)")
                md_lines.append(df_rows.to_markdown(index=False))
            else:
                md_lines.append("_Query executed successfully but returned 0 rows._")
        elif not trace.tool_result.success:
            md_lines.append(f"**Execution Error**: `{trace.tool_result.error}`")
    elif tool_name == "plot_chart":
        md_lines.extend([
            f"### Declarative Chart Specification",
            f"- **Chart Type**: `{trace.router_decision.parameters.get('chart_type')}`",
            f"- **X Axis**: `{trace.router_decision.parameters.get('x_column')}`",
            f"- **Y Axis**: `{trace.router_decision.parameters.get('y_column')}`",
            f"- **Aggregation**: `{trace.router_decision.parameters.get('aggregation')}`",
            f"- **Title**: {trace.router_decision.parameters.get('title')}",
            f""
        ])
        if trace.tool_result.data and "summary_table" in trace.tool_result.data:
            summary = trace.tool_result.data["summary_table"]
            if summary:
                md_lines.append("### Underlying Aggregated Data")
                md_lines.append(pd.DataFrame(summary).to_markdown(index=False))
    elif tool_name == "summary_stats":
        if trace.tool_result.data and "column_stats" in trace.tool_result.data:
            stats = trace.tool_result.data["column_stats"]
            md_lines.append("### Descriptive Statistics")
            md_lines.append(pd.DataFrame(stats).T.to_markdown())

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Grounded Executive Summary",
        trace.narrative_response,
        f"",
        f"---",
        f"",
        f"## 5. Verification & Telemetry Audit",
        f"- **Numerical Faithfulness Guard**: `{'PASSED' if trace.numerical_validation_passed else 'FLAGGED'}`",
        f"- **Notes**: {trace.numerical_validation_notes or 'Grounded in deterministic tool output'}"
    ])

    return "\n".join(md_lines)


# ==============================================================================
# Cached Pipeline & Dataset Profiler
# ==============================================================================
@st.cache_resource
def get_default_agent() -> DataAnalystAgent:
    return DataAnalystAgent(config.dataset_path)

@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    return pd.read_csv(config.dataset_path)

@st.cache_data
def get_default_profile():
    profiler = DatasetProfiler(config.dataset_path)
    return profiler.profile()


# ==============================================================================
# Sidebar: Dataset Source & Uploader
# ==============================================================================
with st.sidebar:
    st.subheader("📁 Dataset Source")
    
    uploaded_file = st.file_uploader(
        "Upload Custom CSV",
        type=["csv"],
        key="csv_uploader",
        help="Upload your own CSV dataset to profile and query dynamically with the agent."
    )

    if uploaded_file is not None:
        # Check if file changed or newly uploaded
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            try:
                df_custom = pd.read_csv(uploaded_file)
                if df_custom.empty or len(df_custom) == 0:
                    st.error("❌ **Upload Rejected**: Uploaded file contains no data rows (empty or header-only).")
                    for k in ["custom_df", "uploaded_filename", "custom_agent", "custom_profile"]:
                        st.session_state.pop(k, None)
                elif len(df_custom.columns) == 0:
                    st.error("❌ **Upload Rejected**: Uploaded file contains no columns.")
                    for k in ["custom_df", "uploaded_filename", "custom_agent", "custom_profile"]:
                        st.session_state.pop(k, None)
                else:
                    st.session_state["custom_df"] = df_custom
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["custom_agent"] = DataAnalystAgent(dataset_path_or_df=df_custom)
                    st.session_state["custom_profile"] = st.session_state["custom_agent"].profiler.profile()
                    st.toast(f"✅ Loaded: {uploaded_file.name} ({len(df_custom):,} rows)", icon="📊")
            except Exception as ex:
                st.error(f"❌ **Error parsing uploaded CSV**: {ex}")
                for k in ["custom_df", "uploaded_filename", "custom_agent", "custom_profile"]:
                    st.session_state.pop(k, None)

    # Determine Active State (Custom vs Default Superstore)
    if "custom_df" in st.session_state and st.session_state["custom_df"] is not None:
        agent = st.session_state["custom_agent"]
        df_raw = st.session_state["custom_df"]
        profile_data = st.session_state["custom_profile"]
        active_filename = st.session_state["uploaded_filename"]
        is_custom_active = True
        
        if st.button("🔄 Reset to Default Superstore", width="stretch"):
            for k in ["custom_df", "uploaded_filename", "custom_agent", "custom_profile"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        agent = get_default_agent()
        df_raw = get_default_dataset()
        profile_data = get_default_profile()
        active_filename = Path(config.dataset_path).name
        is_custom_active = False

    st.divider()
    st.subheader("⚙️ System State")
    dataset_tag = " `[Custom Upload]`" if is_custom_active else ""
    st.caption(f"**Model**: `{config.default_model}`\n\n**Dataset**: `{active_filename}`{dataset_tag}\n\n**Shape**: {len(df_raw):,} rows × {len(df_raw.columns)} cols")
    st.divider()
    
    st.subheader("💡 Demo Queries")
    st.caption("Select a query to trigger 1-click execution:")

    examples = [
        ("SQL: Top 5 Sub-Categories", "What are the top 5 sub-categories by total sales volume?"),
        ("SQL: 2023 Regional Profit", "What was the total profit in the East region during 2023?"),
        ("SQL: 4-Region Comparison", "Compare total sales and profit across all 4 regions."),
        ("Chart: Sales by Category", "Can you plot a bar chart of total sales by product category?"),
        ("Chart: Sales vs Profit Scatter", "Create a scatter plot comparing sales versus profit."),
        ("Stats: Sales & Profit Metrics", "What are the summary statistics, mean, and quartiles for sales and profit?"),
        ("Ambiguity: Best Products", "Show me our best products."),
        ("Ambiguity: Recent Period", "Analyze performance for the recent period."),
        ("Guardrail: Weather Out-of-Scope", "What will the weather be in Seattle next week?")
    ]

    def select_demo_query(prompt_text: str):
        st.session_state["user_query_input"] = prompt_text
        st.session_state["query_text"] = prompt_text
        st.session_state["auto_submit"] = True

    for label, prompt_str in examples:
        st.button(
            label,
            key=f"btn_{label}",
            width="stretch",
            on_click=select_demo_query,
            args=(prompt_str,)
        )


# ==============================================================================
# Header & Architecture Context Bar
# ==============================================================================
active_badge_html = f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 8px;">📁 {active_filename} ({len(df_raw):,} rows)</span>'
st.markdown(f"""
<div class="app-header">
    <div class="app-title">AI Data Analyst Agent {active_badge_html}</div>
    <div class="app-subtitle">
        Autonomous, inspectable analytics with constrained tool routing, DuckDB SQL execution, and anti-hallucination verification.
    </div>
    <div class="arch-caption">
        🛡️ <b>Architecture</b>: Constrained Pydantic Router • DuckDB Read-Only SQL • Declarative Plotly • Zero Arbitrary <code>exec()</code>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Top Navigation Tabs
# ==============================================================================
tab_chat, tab_profiler, tab_eval = st.tabs([
    "💬 Agent Reasoning & Execution (Main Demo)",
    "📊 Dataset Profile & Quality Audit",
    "🧪 Benchmark Evaluation Studio"
])


# ==============================================================================
# TAB 1: MAIN DEMO — REASONING & EXECUTION PIPELINE
# ==============================================================================
with tab_chat:
    # Form wrapper enables natural Enter-key submission
    with st.form(key="query_form", clear_on_submit=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                "Natural Language Question:",
                placeholder="Ask a business, visual, or statistical question about the dataset...",
                label_visibility="collapsed",
                key="user_query_input"
            )
        with col_btn:
            run_query = st.form_submit_button("🚀 Analyze", type="primary", width="stretch")

    auto_submit = st.session_state.pop("auto_submit", False)
    should_execute = (run_query or auto_submit) and bool(user_query and user_query.strip())

    if should_execute:
        query_to_run = user_query.strip()
        st.session_state["query_text"] = query_to_run
        
        with st.spinner("Analyzing question, evaluating dimensions, and executing tool..."):
            trace: AgentTrace = agent.ask(query_to_run)
            st.session_state["last_trace"] = trace
            st.session_state["last_trace_dataset"] = active_filename
    elif st.session_state.get("last_trace") is not None and st.session_state.get("last_trace_dataset") == active_filename:
        trace = st.session_state["last_trace"]
    else:
        trace = None

    if trace is not None:
        tool_name = trace.router_decision.tool.value
        badge_class = {
            "query_data": "badge-query",
            "plot_chart": "badge-chart",
            "summary_stats": "badge-stats",
            "clarify": "badge-clarify"
        }.get(tool_name, "badge-query")

        # ----------------------------------------------------------------------
        # STEP 1: ROUTER DECISION & REASONING (Prominent Inspectable Card)
        # ----------------------------------------------------------------------
        st.markdown(f"""
        <div class="step-header">
            <span>Step 1: Router Decision & Dimensional Audit</span>
            <span>Tool: <span class="badge {badge_class}">{tool_name}</span> &nbsp;|&nbsp; ⏱️ {trace.total_latency_ms} ms</span>
        </div>
        <div class="reasoning-card">
            <b>Router Plan & Dimensional Rationale:</b><br>
            {trace.router_decision.reasoning}
        </div>
        """, unsafe_allow_html=True)

        tool_data = trace.tool_result.data or {}

        # ----------------------------------------------------------------------
        # STEP 2 & 3: TOOL EXECUTION & SYNTHESIS (HANDLES CLARIFY VS ANALYTICAL)
        # ----------------------------------------------------------------------
        if tool_name == "clarify":
            # Intentional Disambiguation Card
            reason_txt = trace.router_decision.parameters.get("reason", "Query is underspecified.")
            missing_txt = trace.router_decision.parameters.get("missing_information", "Specific metric and/or timeframe.")
            sugg_txt = trace.router_decision.parameters.get("suggested_clarification", "")
            reasoning_txt = trace.router_decision.reasoning or ""

            # Determine dynamic context-aware guardrail badge
            lower_context = (str(reason_txt) + " " + str(reasoning_txt) + " " + str(missing_txt)).lower()
            if "weather" in lower_context or "out of scope" in lower_context or "external" in lower_context:
                guard_type = "Out-of-Scope Topic"
                guard_desc = "Execution halted because the request asks for external data outside this retail sales dataset."
            elif "subjective" in lower_context or "best" in lower_context or "top" in lower_context:
                guard_type = "Subjective Ambiguity"
                guard_desc = "Execution halted because ranking requires an explicit business metric (e.g., sales, profit, or volume)."
            elif "dimension" in lower_context or "timeframe" in lower_context or "metric" in lower_context:
                guard_type = "Dimensional Ambiguity"
                guard_desc = f"Execution halted because required analytical dimensions are missing ({missing_txt})."
            else:
                guard_type = "Disambiguation Required"
                guard_desc = f"Execution halted: {reason_txt}"

            st.markdown(f"""
            <div class="clarify-card">
                <div class="clarify-title">⚠️ Clarification Required ({guard_type})</div>
                <div class="clarify-body">
                    <b>Why Execution Was Halted:</b> {reason_txt}<br><br>
                    <b>Missing Dimensions:</b> <code>{missing_txt}</code><br><br>
                    <b>Suggested Disambiguation:</b> {sugg_txt}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"🛡️ **Guardrail Triggered ({guard_type})**: {guard_desc}")

        else:
            # Deterministic Analytical Tool Output
            status_text = "✅ SUCCESS" if trace.tool_result.success else "❌ ERROR"
            st.markdown(f"""
            <div class="step-header">
                <span>Step 2: Deterministic Tool Execution Output</span>
                <span>Status: {status_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Robust Error Handling (Discrepancy A Fix)
            if not trace.tool_result.success:
                st.error(f"❌ **Tool Execution Failed**: {trace.tool_result.error}")
                if tool_name == "query_data":
                    executed_sql = trace.router_decision.parameters.get("query", trace.router_decision.parameters.get("sql_query", ""))
                    if executed_sql:
                        st.code(executed_sql, language="sql")
            else:
                if tool_name == "query_data":
                    executed_sql = trace.router_decision.parameters.get("query", trace.router_decision.parameters.get("sql_query", ""))
                    if executed_sql:
                        st.code(executed_sql, language="sql")
                    
                    rows = tool_data.get("rows", [])
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                        st.caption(f"Returned {len(rows):,} rows from DuckDB in-memory engine.")
                    else:
                        st.warning("Query executed successfully but returned 0 rows.")

                elif tool_name == "plot_chart":
                    if "figure" in tool_data:
                        fig = tool_data["figure"]
                        fig.update_layout(template="plotly_dark", height=460)
                        st.plotly_chart(fig, width="stretch")
                    
                    summary_table = tool_data.get("summary_table") or tool_data.get("raw_aggregated_data")
                    if summary_table:
                        with st.expander("View Underlying Aggregated Data"):
                            st.dataframe(pd.DataFrame(summary_table), width="stretch")

                elif tool_name == "summary_stats":
                    if "column_stats" in tool_data:
                        st.dataframe(pd.DataFrame(tool_data["column_stats"]).T, width="stretch")
                    elif "grouped_stats" in tool_data:
                        st.json(tool_data["grouped_stats"])

            # Step 3: Faithful Synthesis
            st.markdown(f"""
            <div class="step-header" style="margin-top: 20px;">
                <span>Step 3: Grounded Analyst Synthesis</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(trace.narrative_response)

            # Numerical Faithfulness Guard Badge
            if trace.numerical_validation_passed:
                st.markdown("""
                <div class="guard-banner guard-pass">
                    🛡️ <b>Numerical Faithfulness Guard: PASSED</b> — All cited numbers verified against deterministic tool output matrix.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="guard-banner guard-fail">
                    ⚠️ <b>Numerical Faithfulness Guard: FLAG</b> — {trace.numerical_validation_notes}
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # EXPORT / DOWNLOAD RESULT AS MARKDOWN
        # ----------------------------------------------------------------------
        st.divider()
        col_export_msg, col_export_btn = st.columns([4, 2])
        with col_export_msg:
            st.caption(f"📄 **Export Trace**: Download complete inspectable telemetry, tool outputs, and synthesis for `{active_filename}`.")
        with col_export_btn:
            md_content = format_trace_as_markdown(trace, dataset_name=active_filename)
            safe_q_slug = re.sub(r'[^a-zA-Z0-9]', '_', trace.query[:30]).strip('_').lower()
            st.download_button(
                label="📥 Download Result (.md)",
                data=md_content,
                file_name=f"trace_{safe_q_slug}_{int(time.time())}.md",
                mime="text/markdown",
                type="secondary",
                width="stretch"
            )


# ==============================================================================
# TAB 2: DATASET PROFILER & QUALITY AUDIT
# ==============================================================================
with tab_profiler:
    st.subheader(f"📁 Dataset Profile & In-Memory Quality Audit: `{active_filename}`")
    st.caption("Pre-computed metadata and quality diagnostics fed directly into the Router schema context.")
    
    k1, k2, k3, k4 = st.columns(4)
    tot_rows = profile_data["dataset_info"]["row_count"]
    k1.metric("Total Records", f"{tot_rows:,}")
    if "sales" in df_raw.columns:
        k2.metric("Total Sales", f"${df_raw['sales'].sum():,.0f}")
    else:
        k2.metric("Columns", f"{len(df_raw.columns):,}")
        
    if "profit" in df_raw.columns:
        k3.metric("Total Profit", f"${df_raw['profit'].sum():,.0f}")
    elif "salary" in df_raw.columns:
        k3.metric("Total Salary", f"${df_raw['salary'].sum():,.0f}")
    else:
        num_nulls = int(df_raw.isna().sum().sum())
        k3.metric("Total Missing Values", f"{num_nulls:,}")

    k4.metric("In-Memory Footprint", f"{profile_data['dataset_info']['memory_usage_mb']:.2f} MB")

    st.divider()
    st.subheader("📋 Schema & Missing Values Audit")
    
    schema_rows = []
    for col, cinfo in profile_data["columns"].items():
        nulls = cinfo.get("null_count", 0)
        pct_null = (nulls / tot_rows) * 100 if tot_rows > 0 else 0
        uniques = cinfo.get("unique_count", 0)
        stype = cinfo.get("semantic_type", "")
        raw_dtype = cinfo.get("dtype", "")
        
        if stype == "datetime":
            display_dtype = "datetime (ISO 8601)"
        elif stype == "categorical":
            display_dtype = "categorical"
        elif stype == "numeric":
            display_dtype = raw_dtype
        else:
            display_dtype = "string"
            
        schema_rows.append({
            "Column": col,
            "Data Type": display_dtype,
            "Unique Values": uniques,
            "Null Count": nulls,
            "Null %": f"{pct_null:.2f}%",
            "Status": "⚠️ Intentional Nulls" if nulls > 0 else "✅ Clean"
        })
    st.dataframe(pd.DataFrame(schema_rows), width="stretch")

    st.divider()
    st.subheader("🔍 Column Distribution Explorer")
    selected_col = st.selectbox(
        "Inspect Column Distribution:",
        options=list(df_raw.columns),
        index=list(df_raw.columns).index("sales") if "sales" in df_raw.columns else 0
    )
    
    # Enhanced distribution handling (Quirk C Fix)
    if selected_col in ["order_date", "ship_date"]:
        ts_df = df_raw.copy()
        ts_df["month_year"] = pd.to_datetime(ts_df[selected_col], errors="coerce").dt.to_period("M").astype(str)
        monthly_counts = ts_df["month_year"].value_counts().sort_index().reset_index()
        monthly_counts.columns = ["Month", "Order Count"]
        fig_ts = px.bar(
            monthly_counts,
            x="Month",
            y="Order Count",
            template="plotly_dark",
            title=f"Monthly Transaction Volume Trend ('{selected_col}')",
            color_discrete_sequence=["#38bdf8"]
        )
        st.plotly_chart(fig_ts, width="stretch")
    elif pd.api.types.is_numeric_dtype(df_raw[selected_col]):
        fig_dist = px.histogram(
            df_raw,
            x=selected_col,
            nbins=35,
            template="plotly_dark",
            title=f"Distribution of '{selected_col}'",
            color_discrete_sequence=["#38bdf8"]
        )
        st.plotly_chart(fig_dist, width="stretch")
    else:
        top_cats = df_raw[selected_col].value_counts().head(15).reset_index()
        top_cats.columns = [selected_col, "Count"]
        fig_cat = px.bar(
            top_cats,
            x=selected_col,
            y="Count",
            template="plotly_dark",
            title=f"Top Frequencies in '{selected_col}'",
            color_discrete_sequence=["#c084fc"]
        )
        st.plotly_chart(fig_cat, width="stretch")
        if df_raw[selected_col].nunique() > 15:
            st.caption(f"Showing top 15 values out of {df_raw[selected_col].nunique():,} unique values.")


# ==============================================================================
# TAB 3: BENCHMARK & EVALUATION STUDIO
# ==============================================================================
with tab_eval:
    st.subheader("🧪 Ground-Truth Benchmark Evaluation")
    st.caption("Quantitative evaluation harness across 20 multi-domain test cases.")

    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        run_benchmark = st.button("▶️ Run Full Benchmark (20 Qs)", type="primary", width="stretch")
    with col_txt:
        st.caption("Evaluates tool classification accuracy, execution success, answer correctness, and numerical grounding.")

    results_path = Path("evaluation/benchmark_results.json")

    # In-place benchmark execution without jarring tab-resetting reruns (Discrepancy B Fix)
    if run_benchmark:
        with st.spinner("Running automated 20-question evaluation suite..."):
            evaluator = AgentEvaluator()
            summary = evaluator.run_benchmark(verbose=False)
            md_rep = evaluator.generate_markdown_report(summary)
            with open("evaluation/benchmark_summary.md", "w", encoding="utf-8") as f:
                f.write(md_rep)
            st.session_state["benchmark_summary_data"] = summary
            st.success("✅ Benchmark completed successfully! Metrics refreshed below.")

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as rf:
            bdata = json.load(rf)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tool Selection Accuracy", f"{bdata['tool_selection_accuracy_pct']}%")
        m2.metric("Execution Success Rate", f"{bdata['execution_success_rate_pct']}%")
        m3.metric("Answer Correctness", f"{bdata['answer_correctness_rate_pct']}%")
        m4.metric("Avg Latency", f"{bdata['average_latency_ms']:.0f} ms")

        st.divider()
        st.subheader("🎯 Classification Performance by Tool")
        tm_rows = []
        for t_name, metrics in bdata.get("tool_metrics", {}).items():
            tm_rows.append({
                "Tool": t_name,
                "Precision": f"{metrics['precision']:.2f}",
                "Recall": f"{metrics['recall']:.2f}",
                "F1-Score": f"{metrics['f1_score']:.2f}",
                "Support (# Qs)": metrics["support"]
            })
        st.dataframe(pd.DataFrame(tm_rows), width="stretch")

        st.subheader("📋 Detailed Test Case Results")
        cases = bdata.get("results", [])
        if cases:
            cases_df = pd.DataFrame(cases)[[
                "id", "category", "question", "expected_tool", "actual_tool", "tool_match", "answer_correct", "latency_ms"
            ]].copy()
            cases_df["tool_match"] = cases_df["tool_match"].map({True: "✅ Match", False: "❌ Mismatch"})
            cases_df["answer_correct"] = cases_df["answer_correct"].map({True: "✅ Correct", False: "❌ Incorrect"})
            st.dataframe(cases_df, width="stretch")

        err_file = Path("evaluation/error_analysis.md")
        if err_file.exists():
            with st.expander("📖 View Empirical Error Analysis & Failure Case Studies"):
                with open(err_file, "r", encoding="utf-8") as ef:
                    st.markdown(ef.read())
    else:
        st.info("No benchmark results found. Click 'Run Full Benchmark' above to execute evaluation.")
