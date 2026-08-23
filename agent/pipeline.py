"""
Agent Pipeline Coordinator: Orchestrates end-to-end data analytics workflow.
Manages profiling, router decisions, constrained tool dispatch, synthesis, and telemetry trace generation.
"""

import time
from typing import Optional, Any
import pandas as pd

from agent.config import config
from agent.schema import (
    ToolType,
    RouterDecision,
    ToolExecutionResult,
    AgentTrace,
    QueryDataParams,
    PlotChartParams,
    SummaryStatsParams,
    ClarifyParams
)
from agent.profiler import DatasetProfiler
from agent.router import AgentRouter
from agent.synthesizer import ResponseSynthesizer
from agent.tools.query_tool import QueryDataTool
from agent.tools.chart_tool import PlotChartTool
from agent.tools.stats_tool import SummaryStatsTool
from agent.tools.clarify_tool import ClarifyTool
from agent.llm import LLMClient, default_llm_client


class DataAnalystAgent:
    """The central Data Analyst Agent coordinator."""

    def __init__(
        self,
        dataset_path_or_df: Optional[Any] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initializes the agent coordinator, dataset profiler, router, and the 4 execution tools.
        
        Args:
            dataset_path_or_df: CSV filepath string or active in-memory pandas DataFrame.
            llm_client: Optional custom LLMClient instance for model calls.
        """
        if dataset_path_or_df is not None:
            self.dataset_source = dataset_path_or_df
        else:
            self.dataset_source = config.dataset_path
        self.llm = llm_client or default_llm_client

        # Initialize dataset profiler
        self.profiler = DatasetProfiler(self.dataset_source)
        # Pre-compute profile
        self.profiler.profile()

        # Initialize the 4 constrained tools
        self.query_tool = QueryDataTool(self.dataset_source)
        self.chart_tool = PlotChartTool(self.dataset_source)
        self.stats_tool = SummaryStatsTool(self.dataset_source)
        self.clarify_tool = ClarifyTool()

        # Initialize router and synthesizer
        self.router = AgentRouter(self.profiler, self.llm)
        self.synthesizer = ResponseSynthesizer(self.llm)

    def ask(self, question: str) -> AgentTrace:
        """
        Processes a natural language user question end-to-end,
        recording an inspectable AgentTrace at every step.
        """
        start_time = time.time()
        
        # Step 1: Routing & Planning (with inspectable reasoning)
        decision: RouterDecision = self.router.plan_and_route(question)

        # Step 2: Tool Dispatching
        tool_result: ToolExecutionResult
        try:
            if decision.tool == ToolType.QUERY_DATA:
                params = QueryDataParams.model_validate(decision.parameters)
                tool_result = self.query_tool.execute(params)

            elif decision.tool == ToolType.PLOT_CHART:
                params = PlotChartParams.model_validate(decision.parameters)
                tool_result = self.chart_tool.execute(params)

            elif decision.tool == ToolType.SUMMARY_STATS:
                params = SummaryStatsParams.model_validate(decision.parameters)
                tool_result = self.stats_tool.execute(params)

            elif decision.tool == ToolType.CLARIFY:
                params = ClarifyParams.model_validate(decision.parameters)
                tool_result = self.clarify_tool.execute(params)

            else:
                raise ValueError(f"Unknown tool type '{decision.tool}'")

        except Exception as dispatch_err:
            tool_result = ToolExecutionResult(
                tool=decision.tool,
                success=False,
                error=f"Tool parameter validation/dispatch failed: {dispatch_err}",
                execution_time_ms=0.0
            )

        # Step 3: Response Synthesis & Numerical Faithfulness Guard
        narrative, faith_passed, faith_notes = self.synthesizer.synthesize(
            user_query=question,
            decision=decision,
            tool_result=tool_result
        )

        total_latency = round((time.time() - start_time) * 1000, 2)

        # Step 4: Construct Full Telemetry Trace
        trace = AgentTrace(
            query=question,
            router_decision=decision,
            tool_result=tool_result,
            narrative_response=narrative,
            total_latency_ms=total_latency,
            numerical_validation_passed=faith_passed,
            numerical_validation_notes=faith_notes
        )

        return trace

    def run(self, question: str) -> AgentTrace:
        """Alias for ask()."""
        return self.ask(question)
