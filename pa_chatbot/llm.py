"""Thin helpers around the Anthropic SDK."""

import json
from functools import lru_cache

import anthropic

from .config import supports_effort


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment (.env is loaded in config.py).
    return anthropic.Anthropic()


def structured_call(
    *,
    model: str,
    effort: str,
    system: str,
    prompt: str,
    schema: dict,
    max_tokens: int = 8000,
) -> dict | None:
    """One request constrained to a JSON schema. Returns None on a refusal or
    a truncated response so callers can fall back to something sensible."""
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if supports_effort(model):
        output_config["effort"] = effort
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config=output_config,
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
