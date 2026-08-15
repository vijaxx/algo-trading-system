"""Bollinger Band breakout (volatility expansion)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..indicators import bollinger_bands
from .base import LONG, SHORT, Signal, Strategy


class BollingerBreakout(Strategy):
    """Buy closes above the upper band, sell closes below the lower band.

    Note this is the *opposite* posture to the classic "buy the lower band"
    reversion trade: a close outside a 2-sigma band on intraday data is more
    often a volatility expansion than an exhaustion point, so this variant
    trades with the break, not against it.

    A squeeze filter (`max_bandwidth_pct`) requires the bands to have been
    narrow just before the break, which is the setup the strategy is actually
    trying to capture -- breaking out of an already-wide band is usually late.
    """

    name = "bollinger"

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        rr: float = 1.5,
        max_bandwidth_pct: float = 1.2,
        squeeze_lookback: int = 3,
    ) -> None:
        super().__init__(
            period=period,
            num_std=num_std,
            rr=rr,
            max_bandwidth_pct=max_bandwidth_pct,
            squeeze_lookback=squeeze_lookback,
        )
        self.period = period
        self.num_std = num_std
        self.rr = rr
        self.max_bandwidth_pct = max_bandwidth_pct
        self.squeeze_lookback = squeeze_lookback

    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        need = self.period + self.squeeze_lookback + 1
        if len(history) < need:
            return None

        close = history["close"]
        bands = bollinger_bands(close, self.period, self.num_std)

        upper = float(bands["upper"].iloc[-1])
        lower = float(bands["lower"].iloc[-1])
        mid = float(bands["mid"].iloc[-1])
        if pd.isna(upper) or pd.isna(lower) or pd.isna(mid):
            return None

        # Squeeze check on the bars *before* the breakout bar.
        prior_bw = bands["bandwidth"].iloc[-(self.squeeze_lookback + 1) : -1]
        if prior_bw.isna().any():
            return None
        if float(prior_bw.min()) * 100.0 > self.max_bandwidth_pct:
            return None

        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        prev_upper = float(bands["upper"].iloc[-2])
        prev_lower = float(bands["lower"].iloc[-2])

        # Require an actual crossing this bar, not merely "outside the band"
        # (which would re-fire on every bar of an extended move).
        if price > upper and prev_close <= prev_upper:
            risk = price - mid
            if risk <= 0:
                return None
            return Signal(
                side=LONG,
                stop_loss=mid,
                target=price + self.rr * risk,
                reason=f"close {price:.2f} broke upper band {upper:.2f}",
            )
        if price < lower and prev_close >= prev_lower:
            risk = mid - price
            if risk <= 0:
                return None
            return Signal(
                side=SHORT,
                stop_loss=mid,
                target=price - self.rr * risk,
                reason=f"close {price:.2f} broke lower band {lower:.2f}",
            )
        return None
