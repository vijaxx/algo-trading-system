"""Opening Range Breakout (ORB)."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from .base import LONG, SHORT, Signal, Strategy


class OpeningRangeBreakout(Strategy):
    """Trade the break of the first N minutes' high/low.

    The opening range is the high/low of the first `range_minutes` of the
    session. Once that window is complete, a close above the range high is a
    long breakout and a close below the range low is a short breakdown.

    Stop is the opposite side of the range; target is `rr` times the risk.
    Only `max_trades_per_day` entries are allowed, which stops the strategy
    from churning on a choppy day that keeps poking through the range.
    """

    name = "orb"

    def __init__(
        self,
        range_minutes: int = 15,
        rr: float = 1.5,
        buffer_pct: float = 0.0005,
        max_trades_per_day: int = 1,
    ) -> None:
        super().__init__(
            range_minutes=range_minutes,
            rr=rr,
            buffer_pct=buffer_pct,
            max_trades_per_day=max_trades_per_day,
        )
        self.range_minutes = range_minutes
        self.rr = rr
        self.buffer_pct = buffer_pct
        self.max_trades_per_day = max_trades_per_day
        self._trades_today = 0

    def on_session_start(self, day: date) -> None:
        self._trades_today = 0

    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        if self._trades_today >= self.max_trades_per_day:
            return None

        session_start = history.index[0]
        cutoff = session_start + pd.Timedelta(minutes=self.range_minutes)
        opening = history.loc[history.index < cutoff]
        # Need the opening range to be *complete*: at least one bar must have
        # closed after the cutoff, otherwise we are still inside the range.
        if len(opening) == 0 or len(history) == len(opening):
            return None

        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        if or_high <= or_low:
            return None

        last_close = float(history["close"].iloc[-1])
        up_trigger = or_high * (1.0 + self.buffer_pct)
        dn_trigger = or_low * (1.0 - self.buffer_pct)

        if last_close > up_trigger:
            risk = last_close - or_low
            if risk <= 0:
                return None
            self._trades_today += 1
            return Signal(
                side=LONG,
                stop_loss=or_low,
                target=last_close + self.rr * risk,
                reason=f"close {last_close:.2f} > OR high {or_high:.2f}",
            )

        if last_close < dn_trigger:
            risk = or_high - last_close
            if risk <= 0:
                return None
            self._trades_today += 1
            return Signal(
                side=SHORT,
                stop_loss=or_high,
                target=last_close - self.rr * risk,
                reason=f"close {last_close:.2f} < OR low {or_low:.2f}",
            )

        return None
