"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy


_REGISTRY: dict[str, type[Strategy]] = {}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = ["Strategy", "GenericStrategy", "get_strategy"]
