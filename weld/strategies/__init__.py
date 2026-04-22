"""Per-category weld seam fitting strategies."""

from __future__ import annotations

from weld.strategies.base import Strategy, GenericStrategy
from weld.strategies.bellmouth import BellmouthStrategy
from weld.strategies.square_tube import SquareTubeStrategy


_REGISTRY: dict[str, type[Strategy]] = {
    "bellmouth": BellmouthStrategy,
    "square_tube": SquareTubeStrategy,
}


def get_strategy(name: str | None) -> Strategy:
    """Return a strategy instance by name.

    Falls back to GenericStrategy when name is None or not in the registry.
    """
    if name is None:
        return GenericStrategy()
    cls = _REGISTRY.get(name, GenericStrategy)
    return cls()


__all__ = [
    "Strategy", "GenericStrategy",
    "BellmouthStrategy", "SquareTubeStrategy",
    "get_strategy",
]
