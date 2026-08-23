# AI Data Analyst Agent — Project Progress & Engineering Log

## Project Summary
- **Domain**: AI Data Analyst Agent with Constrained Tool Routing & Deterministic Execution
- **Dataset**: Global Retail & Superstore Sales (`data/superstore_sales.csv` — 7,500 records spanning 2021-01-02 to 2024-12-30)
- **Architecture**: 4-Tool Constrained Architecture (`query_data`, `plot_chart`, `summary_stats`, `clarify`) with inspectable reasoning, DuckDB read-only SQL, declarative Plotly visual analytics, and post-synthesis numerical faithfulness verification.
- **LLM Engine**: Provider-agnostic abstraction (`agent/llm.py` via `litellm`) running `gpt-4o-mini` with exponential backoff & jitter.

---

## 🏆 Current Benchmark & Reliability Metrics

- **Tool Selection Classification Accuracy**: **100.0%** (20 / 20 test cases matched)
- **Tool Execution Success Rate**: **100.0%** (20 / 20 executed with zero crashes)
- **Answer Correctness Rate (Value-Level Matrix)**: **85.0%** (17 / 20)
- **Numerical Faithfulness Guard**: **80.0%** (16 / 20)
- **Tool F1-Scores**:
  - `query_data`: **1.00** (10 / 10)
  - `plot_chart`: **1.00** (5 / 5)
  - `summary_stats`: **1.00** (2 / 2)
  - `clarify`: **1.00** (3 / 3)
- **Pytest Unit Test Suite (`pytest -v`)**: **26 / 26 Passed** (100% test suite pass across `test_agent.py`, `test_cli.py`, and `test_evaluator.py`)
- **Average End-to-End Latency**: **4,142 ms** per analytical turn

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

### 3. Bug 3: Strict Numerical Faithfulness Guard
- **Failure**: Synthesizer occasionally cited intermediate sample rows not present in final tool outputs.
- **Solution**: Built a post-synthesis validator in `agent/synthesizer.py` verifying numerical presence against raw tool data matrices within tolerance.
- **Validation**: Caught ungrounded preview figures in `TC-12` and flagged them cleanly in telemetry.

### 4. Bug 4: Provider Quota Resilience & Model Failover
- **Failure**: Rapid consecutive calls caused API rate limit errors.
- **Solution**: Added exponential backoff retry loops with jitter and unified multi-provider routing via LiteLLM.

### 5. Bug 5: Multi-Turn Stateless Boundary & Zero Context Leakage
- **Failure**: Follow-up ellipsis queries (e.g. *"What about 2023?"*) could cause hallucinated carry-overs in stateless deployments.
- **Solution**: Verified via `tests/robustness_check.py` that core agent reasoning maintains zero state buffers across queries, cleanly dispatching `clarify` when an isolated query lacks context.

### 6. Bug 6: Custom CSV Upload Ingestion & Type Safety (Milestone 11)
- **Failure**: Uploading custom CSVs containing boolean columns (`True`/`False`) or non-standard numeric layouts caused casting/truth-value evaluation exceptions during profiling.
- **Solution**: Disambiguated `is_numeric` in `DatasetProfiler` and `SummaryStatsTool` to explicitly exclude boolean dtypes (`is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)`), guarded statistical summary aggregations, and added 3 regression tests in `tests/test_agent.py`.

---

## 🚀 Development Milestones

### 1. Repository Setup & Clean Git State
- Initialized local git tracking, established directory hierarchy, created `.gitignore`, and set up environment configs.

### 2. Core Architecture Implementation
- Implemented `DatasetProfiler`, `AgentRouter`, `QueryDataTool`, `PlotChartTool`, `SummaryStatsTool`, `ClarifyTool`, `ResponseSynthesizer`, and `DataAnalystAgent`.

### 3. Streamlit Interactive Application
- Built a multi-tab dark-mode analytical UI with inspectable execution traces, SQL viewer, interactive Plotly charts, dynamic column distribution explorer, and custom CSV uploader.

### 4. Terminal Command-Line Interface (`cli.py`)
- Created a Rich-powered interactive REPL with formatted trace panels and `:profile` commands.

### 5. Automated Evaluation Benchmark & Telemetry Studio
- Built a 20-question ground-truth evaluation suite with precision, recall, and F1 scoring.

### 6. Failure Recovery & Error Analysis Documentation
- Created `evaluation/error_analysis.md` documenting failure studies and edge-case mitigations.

### 7. Dual-Dimensional Routing & Temporal Reference Anchoring
- Enforced two-dimensional metric/timeframe audits and dynamic temporal anchors.

### 8. Multi-Turn Stateless Boundaries & Compound Query Robustness
- Created `tests/robustness_check.py` auditing 17 adversarial paraphrases, 3 multi-turn state tests, and 3 compound queries.

### 9. 1–2 Minute Interactive Demo Recording & Checklist
- Created `demo_recording_checklist.md` with explicit timecoded segments, spoken highlights, and fallback queries.

### 10. Audit Remediation & Value-Level Correctness Harness
- Implemented cell-by-cell matrix comparator `compare_query_results()` in `evaluation/evaluator.py`, expanded DuckDB denial tokens to 33, and verified 23/23 unit tests.

### 11. Custom CSV Upload Ingestion & Type Safety Hardening (Milestone 11)
- Resolved boolean series misclassification in `DatasetProfiler`, hardened `SummaryStatsTool`, ensured type-safe Streamlit metric cards, and expanded test suite to **26/26 passing tests**.

---

## 📁 Repository Structure & Artifacts

- `.env.example`: Template for required API keys (`OPENAI_API_KEY`, optional `GEMINI_API_KEY`).
- `.gitignore`: Production git ignore protecting secrets, cache, local video recordings, and scratch directories.
- `LICENSE`: Permissive MIT Open-Source License.
- `requirements.txt`: Curated Python dependencies including DuckDB, Plotly, LiteLLM, and Tabulate.
- `demo_recording_checklist.md`: Step-by-step 1–2 minute recording script and pre-flight checklist.
- `demo_script.md`: Spoken script and presenter talking points for screen demo.
- `PROGRESS.md`: Chronological milestone tracker and engineering log.
- `README.md`: Architecture blueprint, systems rationale, quantitative benchmark table, and setup guide.
- `agent/`: Core agent package (profiler, router, tools, synthesizer, coordinator pipeline).
- `app/`: Multi-tab Streamlit dashboard application (`streamlit_app.py`).
- `cli.py`: Rich terminal REPL interface.
- `data/`: Bundled Superstore Sales dataset (`superstore_sales.csv`) and cached schema profile.
- `evaluation/`: Ground-truth benchmark suite (`benchmark_dataset.json`), evaluator harness (`evaluator.py`), and error analysis (`error_analysis.md`).
- `tests/`: Official unit & regression test suite (26 passing tests across `test_agent.py`, `test_cli.py`, `test_evaluator.py`, and `robustness_check.py`).
