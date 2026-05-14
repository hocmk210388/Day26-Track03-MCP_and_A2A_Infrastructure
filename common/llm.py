"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    # Cap completion length: OpenRouter bills by max_tokens budget; the OpenAI client
    # default can be very high (~64k) and triggers 402 on low-credit accounts.
    # Default 1024: low OpenRouter balances often fail at 2048 (402 "can only afford …").
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,  # Bài Tập 1.2: giảm temperature để output ổn định hơn
        max_tokens=max_tokens,
    )