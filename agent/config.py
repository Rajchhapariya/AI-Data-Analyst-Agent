"""
Configuration module for the AI Data Analyst Agent.
Handles environment variables, default model selection, safety parameters, and dataset paths.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_PATH = str(DATA_DIR / "superstore_sales.csv")


@dataclass
class AgentConfig:
    """Central configuration for agent runtime and LLM providers."""
    # LLM Settings
    default_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o")
    temperature: float = 0.0
    max_tokens: int = 1500
    request_timeout_seconds: int = 30
    
    # Dataset Settings
    dataset_path: str = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)
    table_name: str = "dataset"
    
    # Safety & Tool Guardrails
    max_query_rows: int = 500
    disallowed_sql_keywords: tuple = (
        "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
        "REPLACE", "TRUNCATE", "GRANT", "COPY", "ATTACH", "DETACH",
        "PRAGMA", "INSTALL", "LOAD", "EXPORT", "CALL", "EXEC", "EXECUTE"
    )
    
    # Observability
    log_traces: bool = True
    enable_faithfulness_check: bool = True


# Global default configuration instance
config = AgentConfig()
