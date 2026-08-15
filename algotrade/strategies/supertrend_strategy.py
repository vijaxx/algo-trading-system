"""Supertrend trend-following."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..indicators import supertrend
from .base import LONG, SHORT, Signal, Strategy


class SupertrendStrategy(Strategy):
    """Enter on a Supertrend direction flip, exit on the opposite flip.

    Supertrend is an ATR-banded trailing stop: the band ratchets in the
    direction of the trend and never loosens, so the flip is a clean discrete
    event. We enter on the bar where direction changes and use the Supertrend
    line itself as the initial stop, which keeps the stop at exactly the level
    the indicator would flip at.

    `should_exit` closes the position when direction flips back, so the
    strategy is always either flat or aligned with the current trend.
    """

    name = "supertrend"

    def __init__(
        self, period: int = 10, multiplier: float = 2.0, rr: float = 2.0
    ) -> None:
        super().__init__(period=period, multiplier=multiplier, rr=rr)
        self.period = period
        self.multiplier = multiplier
        self.rr = rr

    def _st(self, history: pd.DataFrame) -> pd.DataFrame:
        return supertrend(
            history["high"],
            history["low"],
            history["close"],
            period=self.period,
            multiplier=self.multiplier,
        )

    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        if len(history) < self.period + 3:
            return None

        st = self._st(history)
        dir_now = st["direction"].iloc[-1]
        dir_prev = st["direction"].iloc[-2]
        line = float(st["supertrend"].iloc[-1])
        if pd.isna(dir_now) or pd.isna(dir_prev) or pd.isna(line):
            return None

        price = float(history["close"].iloc[-1])

        if dir_prev < 0 and dir_now > 0:
            risk = price - line
            if risk <= 0:
                return None
            return Signal(
                side=LONG,
                stop_loss=line,
                target=price + self.rr * risk,
                reason=f"Supertrend flipped up at {line:.2f}",
            )
        if dir_prev > 0 and dir_now < 0:
            risk = line - price
            if risk <= 0:
                return None
            return Signal(
                side=SHORT,
                stop_loss=line,
                target=price - self.rr * risk,
                reason=f"Supertrend flipped down at {line:.2f}",
            )
        return None

    def should_exit(self, history: pd.DataFrame, side: int) -> bool:
        if len(history) < self.period + 2:
            return False
        dir_now = self._st(history)["direction"].iloc[-1]
        if pd.isna(dir_now):
            return False
        return (side == LONG and dir_now < 0) or (side == SHORT and dir_now > 0)
