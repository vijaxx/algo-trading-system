"""EMA crossover with an RSI regime filter."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..indicators import atr, ema, rsi
from .base import LONG, SHORT, Signal, Strategy


class EmaRsiCrossover(Strategy):
    """Fast/slow EMA crossover, gated by RSI to avoid counter-trend entries.

    Long when the fast EMA crosses above the slow EMA *and* RSI is above
    `rsi_long_min` (momentum confirms). Short on the mirror condition.

    The RSI gate is what makes this different from a naive crossover: in a
    sideways session the EMAs cross constantly, and requiring RSI to be
    decisively on the same side removes a large share of those whipsaws.

    Stops are ATR-based rather than a fixed percentage, so the same parameters
    work on a Rs 790 bank stock and a Rs 3900 IT stock.
    """

    name = "ema_rsi"

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        rsi_period: int = 14,
        rsi_long_min: float = 55.0,
        rsi_short_max: float = 45.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        rr: float = 1.5,
    ) -> None:
        super().__init__(
            fast=fast,
            slow=slow,
            rsi_period=rsi_period,
            rsi_long_min=rsi_long_min,
            rsi_short_max=rsi_short_max,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            rr=rr,
        )
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_long_min = rsi_long_min
        self.rsi_short_max = rsi_short_max
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.rr = rr

    def _min_bars(self) -> int:
        return max(self.slow, self.rsi_period, self.atr_period) + 2

    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        if len(history) < self._min_bars():
            return None

        close = history["close"]
        ema_fast = ema(close, self.fast)
        ema_slow = ema(close, self.slow)
        rsi_series = rsi(close, self.rsi_period)
        atr_series = atr(history["high"], history["low"], close, self.atr_period)

        f_now, f_prev = float(ema_fast.iloc[-1]), float(ema_fast.iloc[-2])
        s_now, s_prev = float(ema_slow.iloc[-1]), float(ema_slow.iloc[-2])
        r_now = float(rsi_series.iloc[-1])
        a_now = float(atr_series.iloc[-1])
        price = float(close.iloc[-1])

        if pd.isna(r_now) or pd.isna(a_now) or a_now <= 0:
            return None

        crossed_up = f_prev <= s_prev and f_now > s_now
        crossed_dn = f_prev >= s_prev and f_now < s_now
        risk = self.atr_stop_mult * a_now

        if crossed_up and r_now >= self.rsi_long_min:
            return Signal(
                side=LONG,
                stop_loss=price - risk,
                target=price + self.rr * risk,
                reason=f"EMA{self.fast} crossed above EMA{self.slow}, RSI {r_now:.1f}",
            )
        if crossed_dn and r_now <= self.rsi_short_max:
            return Signal(
                side=SHORT,
                stop_loss=price + risk,
                target=price - self.rr * risk,
                reason=f"EMA{self.fast} crossed below EMA{self.slow}, RSI {r_now:.1f}",
            )
        return None

    def should_exit(self, history: pd.DataFrame, side: int) -> bool:
        """Exit early if the EMAs cross back against the position."""
        if len(history) < self._min_bars():
            return False
        close = history["close"]
        f = float(ema(close, self.fast).iloc[-1])
        s = float(ema(close, self.slow).iloc[-1])
        return (side == LONG and f < s) or (side == SHORT and f > s)
