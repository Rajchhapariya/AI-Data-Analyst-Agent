"""
Clarify Tool: Structured disambiguation and out-of-scope query handler.
Provides transparent explanations of missing parameters and guided user suggestions.
"""

import time
from typing import Dict, Any
from agent.schema import ToolType, ToolExecutionResult, ClarifyParams


class ClarifyTool:
    """Handles ambiguous questions, undefined business metrics, and out-of-domain requests."""

    def execute(self, params: ClarifyParams) -> ToolExecutionResult:
        """Constructs structured disambiguation metadata and guided user suggestions.
        
        Args:
            params: Validated ClarifyParams specifying missing dimensions and suggested follow-ups.
            
        Returns:
            ToolExecutionResult with formatted clarification message and missing parameter details.
        """
        start_time = time.time()
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        
        clarification_data = {
            "ambiguity_reason": params.reason,
            "missing_parameters": params.missing_information,
            "suggested_actions": params.suggested_clarification,
            "clarification_message": (
                f"**Clarification Needed**: {params.reason}\n\n"
                f"**Missing Information**: {params.missing_information}\n\n"
                f"**Suggestions**: {params.suggested_clarification}"
            )
        }

        return ToolExecutionResult(
            tool=ToolType.CLARIFY,
            success=True,
            data=clarification_data,
            row_count=0,
            execution_time_ms=elapsed_ms,
            metadata={
                "reason": params.reason,
                "missing_info": params.missing_information
            }
        )
