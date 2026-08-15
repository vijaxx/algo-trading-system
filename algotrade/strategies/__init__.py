"""Strategy registry."""

from __future__ import annotations

from .base import LONG, SHORT, Signal, Strategy
from .bollinger import BollingerBreakout
from .ema_rsi import EmaRsiCrossover
from .orb import OpeningRangeBreakout
from .supertrend_strategy import SupertrendStrategy
from .vwap_reversion import VwapReversion

STRATEGY_REGISTRY = {
    OpeningRangeBreakout.name: OpeningRangeBreakout,
    EmaRsiCrossover.name: EmaRsiCrossover,
    VwapReversion.name: VwapReversion,
    SupertrendStrategy.name: SupertrendStrategy,
    BollingerBreakout.name: BollingerBreakout,
}


def build_strategy(name: str, **params) -> Strategy:
    """Instantiate a strategy by its registry name."""
    try:
        cls = STRATEGY_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError(f"unknown strategy {name!r}; available: {known}") from None
    return cls(**params)


__all__ = [
    "LONG",
    "SHORT",
    "Signal",
    "Strategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
    "OpeningRangeBreakout",
    "EmaRsiCrossover",
    "VwapReversion",
    "SupertrendStrategy",
    "BollingerBreakout",
]
