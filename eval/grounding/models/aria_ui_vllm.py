# copy from Aria-UI

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any

try:
    # vllm is optional; this module expects vllm when used
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)

class AriaUIVLLM:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024):
        if LLM is None:
            raise RuntimeError("vllm is not installed; aria_ui_vllm requires vllm")
        self.model = model
        self.max_tokens = max_tokens
        self.llm = LLM(model=model)

    def generate(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=self.max_tokens)
        out = self.llm.generate(prompt, sampling_params=params)
        return out.text

# Add any helpers needed for integration with ariaui.py
def create_vllm_agent(model_name: str = "gpt-4o-mini") -> AriaUIVLLM:
    return AriaUIVLLM(model=model_name)
