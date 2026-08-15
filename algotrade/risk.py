"""Risk management and SEBI-style compliance guards.

These are modelled on the controls a SEBI-registered algo desk is expected to
have in its risk management system (RMS) before an order reaches the exchange.
None of them are "advice"; they are hard pre-trade blocks.

Guards implemented:
  1. max daily loss  -- kill switch for the session once realised P&L breaches
     a percentage of starting capital.
  2. max open positions -- concentration cap.
  3. position sizing -- allocation per trade as a percentage of equity, with a
     separate stop-based risk cap.
  4. square-off cutoff -- no new intraday entries after `no_entry_after`, and
     forced exit at `square_off_time` (broker MIS auto-square-off).
  5. circuit breaker -- refuse to trade an instrument that has moved beyond
     the exchange price band from its session reference price.

Every check returns a `RiskDecision` so the caller (and the tests) can see
exactly which rule fired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time
from typing import Optional

SQUARE_OFF_DEFAULT = time(15, 15)
NO_ENTRY_AFTER_DEFAULT = time(15, 0)


@dataclass(frozen=True)
class RiskDecision:
    """Result of a pre-trade check."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class RiskConfig:
    """Risk limits. All percentages are of *starting capital* unless noted."""

    starting_capital: float = 500_000.0
    max_daily_loss_pct: float = 2.0
    max_positions: int = 2
    capital_per_trade_pct: float = 20.0
    max_risk_per_trade_pct: float = 1.0
    circuit_limit_pct: float = 10.0
    square_off_time: time = SQUARE_OFF_DEFAULT
    no_entry_after: time = NO_ENTRY_AFTER_DEFAULT

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if not 0 < self.capital_per_trade_pct <= 100:
            raise ValueError("capital_per_trade_pct must be in (0, 100]")
        if self.no_entry_after > self.square_off_time:
            raise ValueError("no_entry_after must not be later than square_off_time")


class RiskManager:
    """Stateful per-session risk engine."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self.equity = self.config.starting_capital
        self.day_start_equity = self.config.starting_capital
        self.realised_pnl_today = 0.0
        self.open_positions = 0
        self.halted_reason = ""
        self.blocked_counts: dict = {}

    # ---- session lifecycle ---------------------------------------------
    def start_day(self) -> None:
        self.day_start_equity = self.equity
        self.realised_pnl_today = 0.0
        self.halted_reason = ""

    def record_trade(self, net_pnl: float) -> None:
        """Book a completed round trip (net of costs) against the day's P&L."""
        self.realised_pnl_today += net_pnl
        self.equity += net_pnl
        if self.daily_loss_breached():
            self.halted_reason = (
                f"daily loss limit hit: {self.realised_pnl_today:,.2f} "
                f"<= -{self.max_daily_loss_amount():,.2f}"
            )

    # ---- individual guards ---------------------------------------------
    def max_daily_loss_amount(self) -> float:
        return self.config.starting_capital * self.config.max_daily_loss_pct / 100.0

    def daily_loss_breached(self) -> bool:
        return self.realised_pnl_today <= -self.max_daily_loss_amount()

    def check_daily_loss(self) -> RiskDecision:
        if self.daily_loss_breached():
            return RiskDecision(
                False,
                f"max daily loss breached (realised {self.realised_pnl_today:,.2f}, "
                f"limit {-self.max_daily_loss_amount():,.2f})",
            )
        return RiskDecision(True)

    def check_position_count(self) -> RiskDecision:
        if self.open_positions >= self.config.max_positions:
            return RiskDecision(
                False,
                f"max positions reached ({self.open_positions}/"
                f"{self.config.max_positions})",
            )
        return RiskDecision(True)

    def check_entry_time(self, now: time) -> RiskDecision:
        if now >= self.config.square_off_time:
            return RiskDecision(
                False, f"past square-off time {self.config.square_off_time}"
            )
        if now >= self.config.no_entry_after:
            return RiskDecision(
                False, f"past no-new-entry cutoff {self.config.no_entry_after}"
            )
        return RiskDecision(True)

    def must_square_off(self, now: time) -> bool:
        """True once the broker's MIS auto-square-off window is reached."""
        return now >= self.config.square_off_time

    def check_circuit_breaker(
        self, price: float, reference_price: float
    ) -> RiskDecision:
        """Block trading if price is outside the exchange price band.

        NSE applies 2/5/10/20% price bands per security; a stock frozen at its
        band is effectively untradeable in the intended direction, and an algo
        that keeps firing orders into a band freeze is exactly what an RMS is
        meant to stop. `circuit_limit_pct` is the applicable band.
        """
        if reference_price <= 0:
            return RiskDecision(False, "invalid reference price")
        move_pct = abs(price - reference_price) / reference_price * 100.0
        if move_pct >= self.config.circuit_limit_pct:
            return RiskDecision(
                False,
                f"circuit breaker: {move_pct:.2f}% move exceeds "
                f"{self.config.circuit_limit_pct:.2f}% band",
            )
        return RiskDecision(True)

    # ---- composite ------------------------------------------------------
    def can_enter(
        self, now: time, price: float, reference_price: float
    ) -> RiskDecision:
        """Run every pre-trade guard, in the order an RMS would."""
        for decision in (
            self.check_daily_loss(),
            self.check_position_count(),
            self.check_entry_time(now),
            self.check_circuit_breaker(price, reference_price),
        ):
            if not decision.allowed:
                key = decision.reason.split(":")[0].split("(")[0].strip()
                self.blocked_counts[key] = self.blocked_counts.get(key, 0) + 1
                return decision
        return RiskDecision(True)

    # ---- sizing ---------------------------------------------------------
    def position_size(self, price: float, stop_loss: float) -> int:
        """Quantity to trade, as the tighter of two independent caps.

        1. Capital cap: `capital_per_trade_pct` of current equity / price.
        2. Risk cap: `max_risk_per_trade_pct` of current equity divided by the
           per-share distance to the stop. This is the cap that actually
           matters -- it keeps the rupee loss on a stopped-out trade constant
           regardless of how wide the stop is.

        Returns 0 when neither cap allows even one share, which the engine
        treats as "no trade".
        """
        if price <= 0:
            return 0
        capital_alloc = self.equity * self.config.capital_per_trade_pct / 100.0
        qty_capital = math.floor(capital_alloc / price)

        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return 0
        risk_budget = self.equity * self.config.max_risk_per_trade_pct / 100.0
        qty_risk = math.floor(risk_budget / risk_per_share)

        return max(int(min(qty_capital, qty_risk)), 0)
