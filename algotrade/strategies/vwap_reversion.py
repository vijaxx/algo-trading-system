"""VWAP mean-reversion."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..indicators import vwap
from .base import LONG, SHORT, Signal, Strategy


class VwapReversion(Strategy):
    """Fade stretched moves away from the session VWAP.

    Institutional intraday execution is benchmarked against VWAP, which gives
    price a tendency to be pulled back toward it. When price is more than
    `entry_dev_pct` below VWAP we buy expecting a snap back; when it is that
    far above we sell.

    Target is the VWAP itself (a moving target -- recomputed each bar would be
    more faithful, but the engine takes a fixed target price, so we use the
    VWAP level at signal time). Stop is placed a further `stop_dev_pct` beyond
    the entry, so the trade is cut if the "stretch" turns into a trend.

    `min_bars` prevents entries in the first few minutes, when VWAP is
    computed off two or three bars and is extremely noisy.
    """

    name = "vwap_reversion"

    def __init__(
        self,
        entry_dev_pct: float = 0.4,
        stop_dev_pct: float = 0.35,
        min_bars: int = 6,
        max_trades_per_day: int = 3,
    ) -> None:
        super().__init__(
            entry_dev_pct=entry_dev_pct,
            stop_dev_pct=stop_dev_pct,
            min_bars=min_bars,
            max_trades_per_day=max_trades_per_day,
        )
        self.entry_dev_pct = entry_dev_pct
        self.stop_dev_pct = stop_dev_pct
        self.min_bars = min_bars
        self.max_trades_per_day = max_trades_per_day
        self._trades_today = 0

    def on_session_start(self, day) -> None:
        self._trades_today = 0

    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        if len(history) < self.min_bars:
            return None
        if self._trades_today >= self.max_trades_per_day:
            return None

        # VWAP is computed on the session slice only -- it resets daily, which
        # is exactly what `history` gives us.
        vw = vwap(history["high"], history["low"], history["close"], history["volume"])
        vwap_now = float(vw.iloc[-1])
        price = float(history["close"].iloc[-1])
        if pd.isna(vwap_now) or vwap_now <= 0:
            return None

        dev_pct = 100.0 * (price - vwap_now) / vwap_now

        if dev_pct <= -self.entry_dev_pct:
            self._trades_today += 1
            return Signal(
                side=LONG,
                stop_loss=price * (1.0 - self.stop_dev_pct / 100.0),
                target=vwap_now,
                reason=f"{dev_pct:.2f}% below VWAP {vwap_now:.2f}",
            )
        if dev_pct >= self.entry_dev_pct:
            self._trades_today += 1
            return Signal(
                side=SHORT,
                stop_loss=price * (1.0 + self.stop_dev_pct / 100.0),
                target=vwap_now,
                reason=f"{dev_pct:.2f}% above VWAP {vwap_now:.2f}",
            )
        return None
