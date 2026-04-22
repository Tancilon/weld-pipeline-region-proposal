"""CoverPlateStrategy: racetrack shape, works with generic + force_close."""

from __future__ import annotations

from weld.strategies.base import GenericStrategy


class CoverPlateStrategy(GenericStrategy):
    force_close: bool = True
