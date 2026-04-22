"""HBeamStrategy: open ⊃-shaped weld seam, same topology as channel_steel."""

from __future__ import annotations

from weld.strategies.channel_steel import ChannelSteelStrategy


class HBeamStrategy(ChannelSteelStrategy):
    """Two open paths, each 3 lines + 2 arcs — identical algorithm to channel_steel."""
    pass
