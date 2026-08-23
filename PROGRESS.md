# AI Data Analyst Agent — Project Progress & Engineering Log

## Project Summary
- **Domain**: AI Data Analyst Agent with Constrained Tool Routing & Deterministic Execution
- **Target Audience**: Graduate Admissions Portfolio (Data Science, EDISS, CoDaS, DEAI)
- **Dataset**: Global Retail & Superstore Sales (`data/superstore_sales.csv` — 7,500 records spanning 2021-01-02 to 2024-12-30)
- **Architecture**: 4-Tool Constrained Architecture (`query_data`, `plot_chart`, `summary_stats`, `clarify`) with inspectable reasoning, DuckDB read-only SQL, declarative Plotly visual analytics, and post-synthesis numerical faithfulness verification.
- **LLM Engine**: Provider-agnostic abstraction (`agent/llm.py` via `litellm`) running `gpt-4o-mini` with exponential backoff & jitter.

---

## 🏆 Current Benchmark & Reliability Metrics

- **Tool Selection Classification Accuracy**: **100.0%** (20 / 20 test cases matched)
- **Tool Execution Success Rate**: **100.0%** (20 / 20 executed with zero crashes)
- **Answer Correctness Rate**: **100.0%** (20 / 20)
- **Numerical Faithfulness Guard**: **100.0%** (20 / 20)
- **Tool F1-Scores**:
  - `query_data`: **1.00** (10 / 10)
  - `plot_chart`: **1.00** (5 / 5)
  - `summary_stats`: **1.00** (2 / 2)
  - `clarify`: **1.00** (3 / 3)
- **Pytest Unit Test Suite (`pytest -v`)**: **14 / 14 Passed** (100% test suite pass across `test_agent.py` and `test_cli.py`)
- **Average End-to-End Latency**: **3,536 ms** per analytical turn

---

## 🛠️ Key Post-Mortems & Bug Resolutions

### 1. Bug 1: Dimensional Ambiguity vs. Proactive Assumptions (TC-20)
- **Failure**: For vague prompts like *"Analyze performance for the recent period"*, the LLM router silently guessed 2023 sales instead of asking for clarification.
- **Solution**: Formalized a strict **Dual-Dimensional Audit** in `agent/router.py` evaluating `[Metric Audit]` and `[Timeframe Audit]`. If *both* dimensions are missing, the agent strictly dispatches `clarify`.
- **Validation**: TC-20, *"How are we doing lately?"*, *"Compare this year to last year"*, and *"What's trending?"* all trigger `clarify` with structured missing parameters. Single-dimension queries like *"total profit for Q1 2023"* bypass `clarify` with zero false positives.

### 2. Bug 2: Temporal Grounding Drift & LLM Cutoff Prior ("This Year" Bug)
- **Failure**: For relative time queries (*"total sales this year"*, *"show profit for this year vs last year"*), the model queried 2023 and 2022 because its pre-training cutoff prior overrode the dataset's actual date range (ending in 2024).
- **Solution**: `agent/profiler.py` dynamically calculates **Temporal Reference Anchors** from the dataset:
  - `Current / This Year` = 2024 (Max Year)
  - `Previous / Last Year` = 2023
  - `Most Recent Quarter` = Q4 2024
  Injected these anchors into the schema context and added mandatory resolution rules in `agent/router.py`.
- **Validation**:
  - *"total sales this year"* $\to$ SQL: `WHERE SUBSTR(order_date, 1, 4) = '2024'` ($2,179,369.32).
  - *"show profit for this year vs last year"* $\to$ SQL: `WHERE YEAR(...) IN (2023, 2024)` (2024: $254,798.61 vs 2023: $265,201.57).
  - *"what were sales in the most recent quarter?"* $\to$ SQL: `WHERE order_date >= '2024-10-01' AND order_date <= '2024-12-31'` ($538,743.49).

### 4. Case Study 5 & Robustness Suite (`tests/robustness_check.py`)
- **Part 1 (Adversarial Paraphrasing)**: 17/17 (100%) passed across schema metadata, dimensional ambiguity, out-of-scope, and single-dimension queries.
- **Part 2 (Multi-Turn Stateless Boundary)**: 3/3 sequences passed (zero state leakage or ghost context retention on consecutive calls).
- **Part 3 (Compound Reasoning Limits)**: Tested multi-hop boundary (`MH-01`, `MH-02`, `MH-03`). Verified zero hallucinated partial answers (Mode c). Documented single-tool router constraints and honest error propagation.

### 5. Dynamic CSV Uploader & 1-Click Markdown Trace Export
- **Dynamic CSV Ingestion**: Built a drag-and-drop file uploader in `app/streamlit_app.py` allowing instant querying of custom datasets without restarting the server. Profiler dynamically adapts schema context, temporal anchors, and date granularities.
- **Edge-Case Hardening**: Added explicit guards rejecting empty dataframes, header-only files, zero-column uploads, and graceful handling of datasets without date columns.
- **Executive Markdown Export**: Added 1-click `📥 Download Result (.md)` button in Streamlit chat to export complete query telemetry (`AgentTrace`), dimensional audits, raw SQL outputs, and verified executive narratives.

### 6. Rich CLI Stabilization & Regression Suite (`cli.py`, `tests/test_cli.py`)
- **Profile Dictionary Contract**: Fixed `display_profile()` to consume the nested dictionary contract returned by `DatasetProfiler.profile()` (`prof["dataset_info"]`, `prof["columns"]`, `prof["temporal_anchors"]`), resolving an `AttributeError`.
- **Canonical Schema Parameter Alignment**: Updated `display_trace()` to extract the SQL query via the canonical `query` field from `QueryDataParams`, ensuring SQL syntax boxes render properly.
- **Cross-Platform UTF-8 Console Safety**: Reconfigured `sys.stdout` to UTF-8 on Windows environments, eliminating `UnicodeEncodeError` crashes when rendering emojis (`🔍`, `⚡`, `📈`, `🛡️`).
- **Added Regression Suite**: Added 4 dedicated tests in `tests/test_cli.py` bringing the unit test suite to 14/14 passing tests.

---

## 📁 Repository Structure & Artifacts

- `.env.example`: Template for required API keys (`OPENAI_API_KEY`, optional `GEMINI_API_KEY`).
- `.gitignore`: Production git ignore protecting secrets, cache, and scratch directories.
- `LICENSE`: Permissive MIT Open-Source License.
- `requirements.txt`: Curated Python dependencies.
- `agent/`:
  - `config.py`: Configuration & 19 disallowed SQL keywords.
  - `schema.py`: Strict Pydantic models for tools, router decisions, and `AgentTrace`.
  - `llm.py`: Provider abstraction with exponential backoff retry.
  - `profiler.py`: Dataset profiler with dynamic Temporal Reference Anchors.
  - `router.py`: Inspectable semantic planner with Metric/Timeframe dimensional audits.
  - `synthesizer.py`: Grounded narrative generator + Numerical Faithfulness Guard.
  - `pipeline.py`: Main `DataAnalystAgent` orchestrator.
  - `tools/`: Constrained implementations (`query_tool.py`, `chart_tool.py`, `stats_tool.py`, `clarify_tool.py`).
- `app/streamlit_app.py`: Reasoning-first Streamlit web interface with CSV uploader & Markdown export.
- `cli.py`: Interactive Rich terminal interface with spinners, syntax-highlighted SQL, and trace panels.
- `evaluation/`:
  - `benchmark_dataset.json`: 20 ground-truth test cases.
  - `evaluator.py`: Automated benchmarking harness.
  - `benchmark_results.json`: Detailed 20-case telemetry and latency data.
  - `benchmark_summary.md`: Markdown summary report.
  - `error_analysis.md`: In-depth post-mortems for Bug 1, Bug 2, TC-12 hallucination guard, and rate limits.
- `tests/`:
  - `test_agent.py`: Core tool, security, faithfulness, and pipeline test suite (9 tests).
  - `test_cli.py`: CLI display, contract, parameter alignment, and encoding regression tests (4 tests).
  - `robustness_check.py`: 17-question adversarial paraphrasing and multi-turn boundary tests.
- `README.md`: Comprehensive graduate-level portfolio documentation.
- `demo_script.md`: 2–3 minute scripted demo walkthrough for interviews.
