"""
Groq LLM wrapper — optional. All callers degrade gracefully if no API key.
Uses the same model already configured in cre_agents.py.
"""

import os
from pathlib import Path

_MODEL = "llama-3.3-70b-versatile"
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    # Load .env from repo root if present
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        return _client
    except ImportError:
        return None


def available() -> bool:
    return _get_client() is not None


def ask(prompt: str, system: str = "You are a concise executive assistant.", max_tokens: int = 1024) -> str | None:
    """Call Groq and return the text response, or None if unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"
