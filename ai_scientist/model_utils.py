"""Centralized model capability detection.

To add support for a new model family, append its pattern to the
appropriate set below.  Every call site imports helpers from here,
so a single edit propagates everywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model-pattern registries
# ---------------------------------------------------------------------------

# Models that require ``max_completion_tokens`` instead of ``max_tokens``.
_COMPLETION_TOKEN_PATTERNS: set[str] = {
    "o1",
    "o3",
    "o4",
    "gpt-5",
}

# Models that do NOT support the ``system`` role in messages.
_NO_SYSTEM_ROLE_PATTERNS: set[str] = {
    "o1",
}

# Models that ignore the ``temperature`` parameter.
_NO_TEMPERATURE_PATTERNS: set[str] = {
    "o1",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _matches(model: str, patterns: set[str]) -> bool:
    return any(p in model for p in patterns)


def uses_completion_tokens(model: str) -> bool:
    """Return *True* if *model* requires ``max_completion_tokens``."""
    return _matches(model, _COMPLETION_TOKEN_PATTERNS)


def supports_system_role(model: str) -> bool:
    """Return *True* if *model* accepts ``system`` role messages."""
    return not _matches(model, _NO_SYSTEM_ROLE_PATTERNS)


def supports_temperature(model: str) -> bool:
    """Return *True* if *model* accepts the ``temperature`` parameter."""
    return not _matches(model, _NO_TEMPERATURE_PATTERNS)


def token_param(model: str, value: int | None = None) -> dict[str, int]:
    """Return the correct max-token kwarg dict for *model*.

    >>> token_param("gpt-5.2", 4096)
    {'max_completion_tokens': 4096}
    >>> token_param("gpt-4o", 4096)
    {'max_tokens': 4096}
    """
    v = value or 4096
    if uses_completion_tokens(model):
        return {"max_completion_tokens": v}
    return {"max_tokens": v}
