"""
LLM Client Abstraction Layer: Provides unified, model-agnostic interface for Gemini, OpenAI, Claude, and local models.
Enforces structured JSON extraction, robust retries, and clean error handling.
"""

import os
import json
import re
import time
import warnings
from typing import Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel
import litellm

from agent.config import config

# Suppress noisy deprecation warnings
warnings.filterwarnings("ignore")
T = TypeVar("T", bound=BaseModel)

# Silence LiteLLM telemetry and verbose logging
litellm.telemetry = False
litellm.suppress_debug_info = True


class LLMClient:
    """Base class and unified interface for LLM calls with structured JSON output parsing."""

    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        self.model_name = model_name or config.default_model
        self.temperature = temperature if temperature is not None else config.temperature
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> Optional[str]:
        """Resolves the appropriate API key based on the model provider."""
        if "gemini" in self.model_name.lower():
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif "openai" in self.model_name.lower() or "gpt" in self.model_name.lower():
            return os.getenv("OPENAI_API_KEY")
        elif "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower():
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates plain text response from the LLM with exponential backoff for rate limits."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 4
        backoff_seconds = 4.0

        for attempt in range(1, max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=config.max_tokens,
                    api_key=self.api_key,
                    timeout=config.request_timeout_seconds
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str or "exhausted" in err_str
                
                if is_rate_limit and attempt < max_retries:
                    # Check if retryDelay is specified in error message
                    delay_match = re.search(r"retry in ([\d\.]+)s", str(e), re.IGNORECASE)
                    wait_time = float(delay_match.group(1)) + 1.0 if delay_match else (backoff_seconds * (2 ** (attempt - 1)))
                    # Cap wait time to 35s
                    wait_time = min(wait_time, 35.0)
                    time.sleep(wait_time)
                    continue
                    
                # Try fallback model if primary fails on non-rate-limit error
                if self.model_name != config.fallback_model and not is_rate_limit:
                    try:
                        response = litellm.completion(
                            model=config.fallback_model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=config.max_tokens,
                            api_key=self.api_key,
                            timeout=config.request_timeout_seconds
                        )
                        return response.choices[0].message.content or ""
                    except Exception:
                        pass
                
                if attempt == max_retries:
                    raise RuntimeError(f"LLM text generation failed with model '{self.model_name}' after {max_retries} attempts: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T]
    ) -> T:
        """
        Prompts the LLM and validates/parses the output into the specified Pydantic response model.
        Extracts JSON from markdown code fences if wrapped by the LLM.
        """
        # Append schema instructions to system prompt
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system_prompt = (
            f"{system_prompt}\n\n"
            "CRITICAL: You MUST respond ONLY with a valid JSON object strictly matching this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Do not include any conversational preamble or postscript outside of the JSON."
        )

        raw_response = self.generate_text(prompt=prompt, system_prompt=augmented_system_prompt)
        parsed_json = self._extract_json(raw_response)
        
        try:
            return response_model.model_validate(parsed_json)
        except Exception as validation_err:
            # One retry with explicit error feedback
            retry_prompt = (
                f"The previous output failed validation against the schema.\n"
                f"Validation Error: {validation_err}\n"
                f"Original Output was:\n{raw_response}\n\n"
                f"Please fix the formatting and return ONLY the valid JSON conforming to the schema."
            )
            retry_raw = self.generate_text(prompt=retry_prompt, system_prompt=augmented_system_prompt)
            retry_json = self._extract_json(retry_raw)
            return response_model.model_validate(retry_json)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extracts JSON dictionary from raw LLM output text, stripping code fences if present."""
        text = text.strip()
        # Look for ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Look for outermost { and }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse JSON from LLM response: {err}\nResponse text: {text}") from err


# Default client instance
default_llm_client = LLMClient()
