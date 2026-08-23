# AI Data Analyst Agent — 1–2 Minute Demo Recording Checklist & Script
> **Target Audience**: Graduate Admissions Committees (Erasmus Mundus AI / Data Science / ML, Master's Evaluators).  
> **Core Theme**: High-assurance AI systems engineering, constrained tool routing, deterministic DuckDB SQL execution, anti-hallucination verification, and quantitative evaluation.

---

## 🎬 1. Pre-Recording Setup Checklist

- [ ] **1. Start Application**:
  ```bash
  streamlit run app/streamlit_app.py
  ```
- [ ] **2. Browser Window Setup**:
  - Open `http://localhost:8501`.
  - Set browser zoom to **100%** (or **110%** for 1440p/4K monitors so text and code boxes are crisp).
  - Hide browser bookmarks bar (`Ctrl + Shift + B` in Chrome/Edge).
  - Maximize the browser window (recommended 1920×1080 resolution).
- [ ] **3. Fresh Clean State**:
  - Refresh the browser once (`Ctrl + Shift + R`).
  - Confirm sidebar displays: `Model: gpt-4o-mini`, `Dataset: superstore_sales.csv`, `Shape: 7,500 rows × 20 cols`.
- [ ] **4. Privacy & Secret Verification**:
  - Verify no terminal window showing `.env` contents is visible in the recording area.
  - The Streamlit UI is 100% sanitized and contains zero API keys.

---

## ⏱️ 2. Step-by-Step 1–2 Minute Recording Timeline

```
┌─────────────────┬───────────────────┬──────────────────────────────────────────┐
│ Time Window     │ Screen Action     │ Key Spoken / Engineering Highlight       │
├─────────────────┼───────────────────┼──────────────────────────────────────────┤
│ 0:00 – 0:15     │ App Header / Tabs │ High-Assurance Architecture Overview     │
│ 0:15 – 0:40     │ Analytical Query  │ Inspectable Router, DuckDB, Guard Badge  │
│ 0:40 – 1:00     │ Visualization     │ Declarative Plotly Spec Generation       │
│ 1:00 – 1:20     │ Ambiguity Handling│ Dual-Dimensional Audit Disambiguation    │
│ 1:20 – 1:40     │ Evaluation Studio │ 20-Question Ground-Truth Benchmarks      │
│ 1:40 – 2:00     │ Summary / Closing │ Systemic Separation of Planning & Logic  │
└─────────────────┴───────────────────┴──────────────────────────────────────────┘
```

---

### Segment 1: Introduction (0:00 – 0:15)
- **Visual**: Keep camera/screen focused on the **Main Header** and **Tab 1: Agent Reasoning & Execution**.
- **What to say**:
  > *"This is my AI Data Analyst Agent. Instead of letting the LLM generate unconstrained Python and execute it with `exec()`—which creates severe security risks and runtime crashes—I architected a deterministic 4-tool routing system: DuckDB read-only SQL, declarative Plotly charts, descriptive statistics, and structured disambiguation."*

---

### Segment 2: Natural-Language Analytical Query (0:15 – 0:40)
- **Visual Action**:
  - Type in the search box or click sidebar chip:  
    👉 **`SQL: Highest Sales Region`** (`"Which region has the highest total sales?"`)
  - Click **`🚀 Analyze`** (or press Enter).
- **What to point your cursor to**:
  1. **Step 1 Card**: Point to the **`[Metric Audit]`** and **`[Timeframe Audit]`** inside the Router Planning drawer.
  2. **Step 2 SQL Box**: Point to the syntax-highlighted DuckDB query (`SELECT region, SUM(sales) AS total_sales FROM dataset GROUP BY region ORDER BY total_sales DESC LIMIT 1`).
  3. **Step 3 Answer**: Point to the grounded narrative answer (`$2,511,186.65`).
  4. **Green Guard Badge**: Point to the **`🛡️ Numerical Faithfulness Guard: PASSED`** badge.
- **What to say**:
  > *"When we ask an analytical question, the agent performs a two-dimensional audit before tool dispatch. It generates a read-only DuckDB SQL query protected by AST mutation blacklists, executes in milliseconds, and our post-synthesis validator cross-checks every cited number against the raw data matrix to prevent hallucinations."*

---

### Segment 3: Declarative Visual Analytics (0:40 – 1:00)
- **Visual Action**:
  - In the sidebar, click:  
    👉 **`Chart: Sales by Category`** (`"Can you plot a bar chart of total sales by product category?"`)
  - Wait 2 seconds for the chart to render.
- **What to point your cursor to**:
  - Hover briefly over the interactive bars (Technology, Furniture, Office Supplies).
- **What to say**:
  > *"For visual analytics, the router dispatches `plot_chart`. It extracts declarative parameters—chart type, axes, and aggregations—and renders a responsive Plotly visualization without running arbitrary plotting scripts."*

---

### Segment 4: Ambiguity Disambiguation & Guardrails (1:00 – 1:20)
- **Visual Action**:
  - In the sidebar, click:  
    👉 **`Ambiguity: Recent Period`** (`"Analyze performance for the recent period."`)
- **What to point your cursor to**:
  - Point to the **`Clarification Card`** and the red **`Dimensional Ambiguity`** tag.
- **What to say**:
  > *"A critical problem in AI agents is silent hallucination when queries are vague. When both the metric and timeframe are missing, our dual-audit policy halts execution and returns a structured Clarification Card with actionable options rather than guessing."*

---

### Segment 5: Quantitative Evaluation & Benchmark Studio (1:20 – 1:40)
- **Visual Action**:
  - Click **`🧪 Benchmark Evaluation Studio`** (Tab 3).
- **What to point your cursor to**:
  - Point to the 4 Top KPI Cards:
    - **Tool Selection Accuracy**: `100.0%`
    - **Execution Success Rate**: `100.0%`
    - **Answer Correctness**: `100.0%`
    - **Avg Latency**: `3,536 ms`
  - Scroll down slightly to show the **Classification Performance by Tool** (1.00 F1 across all 4 tools) and the **Detailed Test Case Results** table.
- **What to say**:
  > *"To validate reliability systematically, I built an automated 20-question ground-truth evaluation harness. The system achieves 100% tool selection accuracy, 1.00 F1-score across all 4 tools, and sub-4-second end-to-end latency. We have also documented failure edge cases and multi-hop boundaries in our empirical error analysis."*

---

### Segment 6: Closing Summary (1:40 – 2:00)
- **Visual Action**:
  - Switch back to **Tab 1** or show the clean full interface.
- **What to say**:
  > *"In summary, this project demonstrates how constraining LLMs to deterministic execution layers provides the auditability, safety, and speed required for enterprise and scientific data systems. Thank you."*

---

## 🔄 3. Backup & Fallback Queries

If an API call experiences temporary provider network latency during recording, use these tested 1-click sidebar backups:

| Scenario | Primary Query | 1-Click Backup Query | Expected Tool |
| :--- | :--- | :--- | :--- |
| **SQL Query** | `"Which region has the highest total sales?"` | `SQL: Top 5 Sub-Categories` | `query_data` |
| **Chart** | `Chart: Sales by Category` | `Chart: Sales vs Profit Scatter` | `plot_chart` |
| **Ambiguity** | `Ambiguity: Recent Period` | `Ambiguity: Best Products` | `clarify` |
| **Guardrail** | `Ambiguity: Recent Period` | `Guardrail: Weather Out-of-Scope` | `clarify` |

---

## 🚫 4. What NOT to Show on Screen

1. ❌ **Do NOT open `.env` or IDE configuration files** during the recording.
2. ❌ **Do NOT claim 'zero vulnerabilities' or '100% mathematical security'** — explain that constrained routing *significantly reduces the attack surface* compared to `exec()`.
3. ❌ **Do NOT switch between browser windows or show desktop notifications**. (Turn on Windows Focus Assist / Do Not Disturb).

---

## ✅ 5. Final Recording Checklist

- [ ] Audio input test: Microphone is clear and background noise is minimal.
- [ ] Screen recording area set to full 1080p browser window.
- [ ] Streamlit app running cleanly with zero terminal errors.
- [ ] Sidebar quick chips ready for instant 1-click execution.
- [ ] Recording time target: **1 minute 30 seconds to 2 minutes maximum**.
