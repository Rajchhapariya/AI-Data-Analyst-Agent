# AI Data Analyst Agent — Benchmark Evaluation Report

**Total Test Cases**: 20
- **Tool Selection Accuracy**: **100.0%**
- **Tool Execution Success Rate**: **100.0%**
- **Answer Correctness Rate**: **100.0%**
- **Numerical Faithfulness Guard**: **100.0%**
- **Average Query Latency**: **3535.99 ms**

## Tool Classification Metrics (Precision / Recall / F1)
| Tool | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `query_data` | 1.00 | 1.00 | **1.00** | 10 |
| `plot_chart` | 1.00 | 1.00 | **1.00** | 5 |
| `summary_stats` | 1.00 | 1.00 | **1.00** | 2 |
| `clarify` | 1.00 | 1.00 | **1.00** | 3 |

## Detailed Test Case Results
| ID | Category | Question | Expected Tool | Actual Tool | Tool Match | Answer Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TC-01 | `lookup` | What is the total sales amount across the ent... | `query_data` | `query_data` | ✅ | ✅ |
| TC-02 | `lookup` | How many total orders and unique customers ar... | `query_data` | `query_data` | ✅ | ✅ |
| TC-03 | `lookup` | What was the total profit in the East region ... | `query_data` | `query_data` | ✅ | ✅ |
| TC-04 | `aggregation` | Which product category generated the highest ... | `query_data` | `query_data` | ✅ | ✅ |
| TC-05 | `aggregation` | What are the top 5 sub-categories by total sa... | `query_data` | `query_data` | ✅ | ✅ |
| TC-06 | `aggregation` | Which customer segment has the highest averag... | `query_data` | `query_data` | ✅ | ✅ |
| TC-07 | `comparison` | Compare total sales and total profit across a... | `query_data` | `query_data` | ✅ | ✅ |
| TC-08 | `comparison` | How do sales and profit compare between First... | `query_data` | `query_data` | ✅ | ✅ |
| TC-09 | `comparison` | What was the average shipping cost for Critic... | `query_data` | `query_data` | ✅ | ✅ |
| TC-10 | `aggregation` | What is the average profit margin (profit div... | `query_data` | `query_data` | ✅ | ✅ |
| TC-11 | `chart` | Can you plot a bar chart of total sales by pr... | `plot_chart` | `plot_chart` | ✅ | ✅ |
| TC-12 | `chart` | Visualize the yearly sales trend from 2021 to... | `plot_chart` | `plot_chart` | ✅ | ✅ |
| TC-13 | `chart` | Create a scatter plot comparing sales versus ... | `plot_chart` | `plot_chart` | ✅ | ✅ |
| TC-14 | `chart` | Show a box plot of shipping costs grouped by ... | `plot_chart` | `plot_chart` | ✅ | ✅ |
| TC-15 | `chart` | Plot a histogram of order quantities to show ... | `plot_chart` | `plot_chart` | ✅ | ✅ |
| TC-16 | `stats` | What are the summary statistics, mean, standa... | `summary_stats` | `summary_stats` | ✅ | ✅ |
| TC-17 | `stats` | Give me the descriptive statistics and distri... | `summary_stats` | `summary_stats` | ✅ | ✅ |
| TC-18 | `ambiguity_out_of_scope` | Show me our best products. | `clarify` | `clarify` | ✅ | ✅ |
| TC-19 | `ambiguity_out_of_scope` | What will the weather and temperature be in S... | `clarify` | `clarify` | ✅ | ✅ |
| TC-20 | `ambiguity_out_of_scope` | Analyze performance for the recent period. | `clarify` | `clarify` | ✅ | ✅ |