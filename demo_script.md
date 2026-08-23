# AI Data Analyst Agent — Scripted Demo Walkthrough (2–3 Minutes)

> **Audience**: Technical Interviewers, AI Systems Architects, and Engineering Assessors  
> **Target Duration**: ~2.5 to 3.0 Minutes  
> **Key Message**: *"This project demonstrates deliberate AI systems engineering: constrained safe tools, inspectable reasoning traces, DuckDB SQL execution, interactive visualizations, and deterministic anti-hallucination verification."*

---

## 🎬 Demo Preparation Checklist
1. Ensure `.env` is configured with `OPENAI_API_KEY` (or `GEMINI_API_KEY`).
2. Launch the Streamlit application in a clean browser window:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. Have a terminal window open with the interactive CLI ready:
   ```bash
   python cli.py
   ```

---

## ⏱️ Section-by-Section Walkthrough

### 1. Introduction & Architectural Justification (0:00 – 0:35)
- **What to say**:
  > *"Hi everyone. Today I'm demonstrating the **AI Data Analyst Agent**—an autonomous, conversational data analytics system. When building LLM data assistants, the industry default is often an unconstrained wrapper where the model writes arbitrary Python code executed via `exec()`.
  > 
  > In enterprise and research environments, that creates severe remote code execution vulnerabilities, unpredictable runtime crashes, and hallucinated metrics.
  > 
  > In this project, I engineered a **constrained tool architecture**. The LLM acts strictly as a semantic router and parameter extractor, dispatching requests across four deterministic tools: an in-memory **DuckDB SQL engine**, a declarative **Plotly chart builder**, a **descriptive statistics profiler**, and a **safety/disambiguation handler**."*

---

### 2. Dataset Profiling & Complex SQL Querying (0:35 – 1:15)
- **Action**: In the Streamlit UI, click **Tab 2: Dataset Profile & Quality Audit**.
- **What to say**:
  > *"Before processing user questions, our profiler audits the dataset in memory—analyzing 7,500 retail transactions across 4 years. It computes dynamic Temporal Reference Anchors directly from the data (anchoring 'this year' to 2024 and 'last year' to 2023) and audits data quality—such as the 38 intentional nulls in discount."*

- **Action**: Switch to **Tab 1: Agent Reasoning & Execution** and click the sidebar button:  
  `SQL: Top 5 Sub-Categories` *(1-click auto-executes)*.
- **What to say**:
  > *"When I trigger a business query, notice three key engineering layers:
  > 1. **Step 1 Inspectable Reasoning**: The agent displays an open dimensional audit card before tool dispatch.
  > 2. **Step 2 DuckDB AST Guard**: It generates and executes a read-only SQL aggregation on DuckDB with strict AST blacklists blocking any mutations.
  > 3. **Step 3 Numerical Faithfulness Guard**: This green verification badge confirms that our post-synthesis validator parsed every cited number and cross-checked it against the raw data matrix, ensuring zero hallucinated figures."*

- **Action**: In Tab 1, scroll to the bottom of the execution trace and point to the **`📥 Download Result (.md)`** button.
- **What to say**:
  > *"Users and BI auditors can also export any analytical result into an executive Markdown report with one click—capturing the full telemetry trace, prompt context, and verified data tables for persistence."*

---

### 3. Visual Analytics & Descriptive Statistics (1:15 – 1:55)
- **Action**: In Tab 1, click sidebar button:  
  `Chart: Sales by Category`
- **What to say**:
  > *"For visual analytics, the agent identifies chart parameters and dispatches `plot_chart`. It extracts declarative properties—chart type, axes, aggregation function—and renders a responsive, dark-mode Plotly visualization without running arbitrary plotting scripts."*

- **Action**: Click sidebar button:  
  `Stats: Sales & Profit Metrics`
- **What to say**:
  > *"For statistical inquiries, it calls `summary_stats`, computing non-parametric quartiles, interquartile ranges, and skewness to explain data distributions clearly."*

---

### 4. Ambiguity Disambiguation & Guardrails (1:55 – 2:30)
- **Action**: Click sidebar button:  
  `Ambiguity: Recent Period`
- **What to say**:
  > *"A critical challenge in LLM agents is dimensional ambiguity. In our dual-audit policy, if both the metric and timeframe are missing—like 'analyze performance for the recent period'—the agent halts execution and renders a structured Clarification Card rather than silently hallucinating assumptions."*

- **Action**: Click sidebar button:  
  `Guardrail: Weather Out-of-Scope`
- **What to say**:
  > *"Similarly, out-of-domain requests are gracefully caught, explaining dataset boundaries and suggesting relevant sales inquiries."*

---

### 5. Quantitative Evaluation & Interview Wrap-Up (2:30 – 3:00)
- **Action**: Switch to **Tab 3: Benchmark Evaluation Studio**.
- **What to say**:
  > *"Finally, to prove system reliability beyond anecdotal tests, I built an automated 20-question Ground Truth benchmark harness and a standalone 17-question adversarial robustness suite.
  > 
  > The system achieves **100% tool selection accuracy (20/20)**, **1.00 F1-score across all 4 tools**, **100% answer correctness rate**, and **100% numerical faithfulness rate**, with an average end-to-end latency of **3,536 ms** per query.
  > 
  > We've also documented full empirical post-mortems for edge cases—including dimensional ambiguity, temporal anchoring drift, multi-turn stateless boundaries, and compound reasoning limits—in `evaluation/error_analysis.md`.
  > 
  > This architecture proves that constrained, inspectable tool calling provides the safety, speed, and determinism necessary for production-grade AI data science."*

---

## 🎯 Quick Reference for Common Interview Questions

| Question | Recommended Defense Response |
| :--- | :--- |
| **Why not allow Python code generation via Code Interpreter?** | Code generation introduces arbitrary code execution (RCE) vulnerabilities, library version fragility, non-deterministic latency, and high hallucination risk. Constrained tools provide guaranteed determinism and safety. |
| **Why DuckDB instead of SQLite or Pandas?** | DuckDB is columnar, highly vectorized, optimized for OLAP analytics, runs in-memory with zero server setup, and executes complex SQL aggregations faster than Pandas in Python runtime. |
| **How do you prevent the LLM from hallucinating numbers in text?** | We built a post-synthesis **Numerical Faithfulness Guard** in `agent/synthesizer.py` that extracts numeric tokens with regex and cross-validates them against raw SQL/tool outputs before returning the response. |
| **How does the system resolve relative dates like 'this year'?** | `DatasetProfiler` computes **Temporal Reference Anchors** directly from dataset timestamps (`max_year = 2024`, `last_year = 2023`, `most_recent_quarter = Q4 2024`) and injects them into the schema context, overriding LLM pre-training cutoff priors. |
| **How does the system handle rate limits and multi-provider switching?** | We abstracted LLM calls via `litellm` in `agent/llm.py` with exponential backoff and jitter, allowing seamless switching between OpenAI, Gemini, and Claude via `.env` configuration. |
