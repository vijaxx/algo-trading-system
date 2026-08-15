"""Performance metrics.

Deliberately computed on **net** P&L (after Indian transaction costs) and
reported alongside the gross figures, because the whole point of the cost
model is the gap between the two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    strategy: str
    side: int  # +1 long, -1 short
    quantity: int
    entry_time: object
    entry_price: float
    exit_time: object
    exit_price: float
    exit_reason: str
    gross_pnl: float
    costs: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["net_pnl"] = self.net_pnl
        d["side"] = "LONG" if self.side > 0 else "SHORT"
        return d


@dataclass
class PerformanceReport:
    num_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    cost_drag_pct_of_gross: float
    return_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    trading_days: int
    starting_capital: float
    ending_capital: float

    def as_dict(self) -> Dict:
        return asdict(self)

    def as_rows(self) -> List:
        """(label, formatted value) pairs for CLI table printing."""
        return [
            ("Trades", f"{self.num_trades}"),
            ("Wins / Losses", f"{self.wins} / {self.losses}"),
            ("Win rate", f"{self.win_rate_pct:.2f}%"),
            ("Gross P&L", f"Rs {self.gross_pnl:,.2f}"),
            ("Total costs", f"Rs {self.total_costs:,.2f}"),
            ("Net P&L", f"Rs {self.net_pnl:,.2f}"),
            ("Cost drag (% of gross)", f"{self.cost_drag_pct_of_gross:.2f}%"),
            ("Net return on capital", f"{self.return_pct:.2f}%"),
            ("Average win", f"Rs {self.avg_win:,.2f}"),
            ("Average loss", f"Rs {self.avg_loss:,.2f}"),
            ("Profit factor", f"{self.profit_factor:.3f}"),
            ("Expectancy / trade", f"Rs {self.expectancy:,.2f}"),
            ("Sharpe (annualised)", f"{self.sharpe_ratio:.3f}"),
            ("Max drawdown", f"Rs {self.max_drawdown:,.2f}"),
            ("Max drawdown %", f"{self.max_drawdown_pct:.2f}%"),
            ("Trading days", f"{self.trading_days}"),
            ("Starting capital", f"Rs {self.starting_capital:,.2f}"),
            ("Ending capital", f"Rs {self.ending_capital:,.2f}"),
        ]


def daily_pnl_series(trades: List[Trade]) -> pd.Series:
    """Net P&L aggregated by exit date."""
    if not trades:
        return pd.Series(dtype=float)
    rows = [(pd.Timestamp(t.exit_time).normalize(), t.net_pnl) for t in trades]
    s = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    return s.groupby(level=0).sum().sort_index()


def equity_curve(trades: List[Trade], starting_capital: float) -> pd.Series:
    """Daily mark-to-close equity (intraday-only book, so flat overnight)."""
    daily = daily_pnl_series(trades)
    if daily.empty:
        return pd.Series([starting_capital], dtype=float)
    return starting_capital + daily.cumsum()


def max_drawdown(curve: pd.Series) -> tuple:
    """Largest peak-to-trough fall, in rupees and as a percentage of the peak."""
    if curve.empty:
        return 0.0, 0.0
    running_peak = curve.cummax()
    dd = curve - running_peak
    trough = dd.min()
    if trough >= 0:
        return 0.0, 0.0
    idx = dd.idxmin()
    peak_at_trough = float(running_peak.loc[idx])
    pct = 100.0 * float(trough) / peak_at_trough if peak_at_trough else 0.0
    return abs(float(trough)), abs(pct)


def sharpe_ratio(
    daily_returns: pd.Series, risk_free_annual: float = 0.0
) -> float:
    """Annualised Sharpe from daily returns.

    Sharpe = mean(excess daily return) / stdev(daily return) * sqrt(252).
    Sample stdev (ddof=1) is used, which is the convention for a return
    series. Returns 0.0 when there is no dispersion or fewer than 2 days --
    an undefined Sharpe is reported as 0, never as a large number.
    """
    if daily_returns is None or len(daily_returns) < 2:
        return 0.0
    rf_daily = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    sd = float(daily_returns.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def build_report(
    trades: List[Trade],
    starting_capital: float,
    risk_free_annual: float = 0.0,
) -> PerformanceReport:
    """Compute the full metric set from a list of completed trades."""
    n = len(trades)
    gross = float(sum(t.gross_pnl for t in trades))
    costs = float(sum(t.costs for t in trades))
    net = gross - costs

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    win_rate = 100.0 * len(wins) / n if n else 0.0
    avg_win = float(np.mean([t.net_pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.net_pnl for t in losses])) if losses else 0.0

    gross_profit = float(sum(t.net_pnl for t in wins))
    gross_loss = abs(float(sum(t.net_pnl for t in losses)))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    curve = equity_curve(trades, starting_capital)
    dd_abs, dd_pct = max_drawdown(curve)

    daily = daily_pnl_series(trades)
    if len(daily) >= 2:
        prior_equity = (starting_capital + daily.cumsum().shift(1)).fillna(
            starting_capital
        )
        daily_returns = daily / prior_equity
    else:
        daily_returns = pd.Series(dtype=float)
    sharpe = sharpe_ratio(daily_returns, risk_free_annual)

    cost_drag = 100.0 * costs / abs(gross) if gross != 0 else 0.0

    return PerformanceReport(
        num_trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=win_rate,
        gross_pnl=gross,
        total_costs=costs,
        net_pnl=net,
        cost_drag_pct_of_gross=cost_drag,
        return_pct=100.0 * net / starting_capital if starting_capital else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=net / n if n else 0.0,
        sharpe_ratio=sharpe,
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        trading_days=int(len(daily)),
        starting_capital=starting_capital,
        ending_capital=starting_capital + net,
    )


def trades_to_frame(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy",
                "side",
                "quantity",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "exit_reason",
                "gross_pnl",
                "costs",
                "net_pnl",
            ]
        )
    return pd.DataFrame([t.as_dict() for t in trades])
