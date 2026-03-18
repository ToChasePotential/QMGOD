# chugao/llm_deepseek.py
from __future__ import annotations

import os
import time
from typing import Callable, Optional, List, Dict, Any

from openai import OpenAI


def build_deepseek_llm_generate(
    *,
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com/v1",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout_s: float = 60.0,
    max_retries: int = 2,
) -> Callable[[str], str]:
    """
    Return a callable llm_generate(prompt:str)->str for UnionGraph-RAG.
    Compatible with your step1/2/3 injection style:
      - Step1: JSON-only output for Double-keys
      - Step2/3: may output JSON or text, your parsers handle it
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def _call(prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"DeepSeek call failed after retries: {last_err}")

    return _call
