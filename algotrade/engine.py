"""Bar-by-bar backtesting engine.

Design goals, in priority order:

1. **No lookahead.** A strategy is handed a DataFrame slice ending at the bar
   that just closed. Rows after it are not in the object, so a bug cannot
   silently read them. Signals produced at bar `i` are filled at the OPEN of
   bar `i + 1`. Nothing in the fill path reads bar `i + 1`'s high, low or
   close.

2. **Honest fills.** Market orders cross the spread (slippage), and every fill
   pays the full Indian intraday charge stack.

3. **Risk first.** Every entry passes through the RiskManager. If a guard
   blocks it, the trade does not happen and the block is counted.

Intrabar assumption: OHLC bars do not record the path within the bar. When
both the stop and the target lie inside a bar's range we assume the STOP was
hit first. That is the pessimistic choice and it is deliberate -- assuming the
target first is a classic way to manufacture a backtest that cannot be
reproduced live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from .broker import BUY, SELL, BrokerAdapter, Order, PaperBroker
from .costs import DEFAULT_COSTS, IndianEquityIntradayCosts
from .metrics import PerformanceReport, Trade, build_report, trades_to_frame
from .risk import RiskConfig, RiskManager
from .strategies.base import LONG, SHORT, Signal, Strategy

EXIT_STOP = "stop_loss"
EXIT_TARGET = "target"
EXIT_SQUARE_OFF = "square_off"
EXIT_EOD = "end_of_day"
EXIT_STRATEGY = "strategy_exit"


@dataclass
class OpenPosition:
    symbol: str
    side: int
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    entry_costs: float
    reason: str = ""


@dataclass
class PendingEntry:
    symbol: str
    side: int
    quantity: int
    stop_loss: float
    target: float
    reason: str


@dataclass
class BacktestResult:
    strategy: str
    symbols: List[str]
    trades: List[Trade]
    report: PerformanceReport
    blocked: Dict[str, int]
    signals_generated: int
    signals_sized_out: int
    starting_capital: float

    def trades_frame(self) -> pd.DataFrame:
        return trades_to_frame(self.trades)

    def per_symbol(self) -> pd.DataFrame:
        df = self.trades_frame()
        if df.empty:
            return df
        grouped = df.groupby("symbol").agg(
            trades=("net_pnl", "size"),
            gross_pnl=("gross_pnl", "sum"),
            costs=("costs", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda s: 100.0 * (s > 0).mean()),
        )
        return grouped.sort_values("net_pnl", ascending=False)


class BacktestEngine:
    """Event-driven intraday backtester across a basket of symbols."""

    def __init__(
        self,
        strategy_factory,
        risk_config: Optional[RiskConfig] = None,
        cost_model: Optional[IndianEquityIntradayCosts] = None,
        slippage_bps: float = 2.0,
        broker: Optional[BrokerAdapter] = None,
        apply_costs: bool = True,
    ) -> None:
        """
        strategy_factory: zero-arg callable returning a fresh Strategy. One
            instance is created per symbol so per-day state (opening ranges,
            trade counters) never leaks between instruments.
        apply_costs: set False to measure the same run with zero transaction
            costs -- this is how the README's cost-impact table is produced.
        """
        self.strategy_factory = strategy_factory
        self.risk_config = risk_config or RiskConfig()
        self.cost_model = cost_model or DEFAULT_COSTS
        self.slippage_bps = slippage_bps
        self.apply_costs = apply_costs
        self.broker = broker or PaperBroker(
            starting_funds=self.risk_config.starting_capital,
            slippage_bps=slippage_bps,
            cost_model=self.cost_model,
        )

    # ------------------------------------------------------------------
    def run(self, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        if not data:
            raise ValueError("no data supplied")

        self.broker.connect()
        risk = RiskManager(self.risk_config)
        strategies: Dict[str, Strategy] = {s: self.strategy_factory() for s in data}
        strategy_name = next(iter(strategies.values())).name

        symbols = sorted(data)
        trades: List[Trade] = []
        signals_generated = 0
        signals_sized_out = 0

        positions: Dict[str, OpenPosition] = {}
        pending_entries: Dict[str, Optional[PendingEntry]] = {s: None for s in symbols}
        pending_exits: Dict[str, Optional[str]] = {s: None for s in symbols}
        prev_close: Dict[str, Optional[float]] = {s: None for s in symbols}

        all_days = sorted(
            {ts.date() for df in data.values() for ts in df.index}
        )

        for day in all_days:
            risk.start_day()
            day_data = {
                s: data[s].loc[
                    (data[s].index >= pd.Timestamp(day))
                    & (data[s].index < pd.Timestamp(day) + pd.Timedelta(days=1))
                ]
                for s in symbols
            }
            day_data = {s: df for s, df in day_data.items() if len(df) > 0}
            if not day_data:
                continue

            for sym, strat in strategies.items():
                strat.on_session_start(day)
            for sym in symbols:
                pending_entries[sym] = None
                pending_exits[sym] = None

            n_bars = max(len(df) for df in day_data.values())

            for i in range(n_bars):
                for sym, df in day_data.items():
                    if i >= len(df):
                        continue
                    row = df.iloc[i]
                    ts = df.index[i]
                    is_last_bar = i == len(df) - 1
                    bar_open = float(row["open"])
                    reference = prev_close[sym] if prev_close[sym] else bar_open

                    # ---- A. fill orders queued at the previous bar's close
                    if pending_exits[sym] is not None and sym in positions:
                        reason = pending_exits[sym]
                        pending_exits[sym] = None
                        trades.append(
                            self._close(positions.pop(sym), bar_open, ts, reason, risk)
                        )
                    pending_exits[sym] = None

                    if pending_entries[sym] is not None and sym not in positions:
                        pe = pending_entries[sym]
                        pending_entries[sym] = None
                        pos = self._open(pe, bar_open, ts, risk)
                        if pos is not None:
                            positions[sym] = pos
                    pending_entries[sym] = None

                    # ---- B. manage an open position on THIS bar
                    pos = positions.get(sym)
                    if pos is not None:
                        if risk.must_square_off(ts.time()):
                            trades.append(
                                self._close(
                                    positions.pop(sym),
                                    bar_open,
                                    ts,
                                    EXIT_SQUARE_OFF,
                                    risk,
                                )
                            )
                        else:
                            exit_price, reason = self._check_stop_target(pos, row)
                            if reason is not None:
                                trades.append(
                                    self._close(
                                        positions.pop(sym), exit_price, ts, reason, risk
                                    )
                                )

                    # ---- C. hard end-of-day flatten (safety net)
                    if is_last_bar and sym in positions:
                        trades.append(
                            self._close(
                                positions.pop(sym),
                                float(row["close"]),
                                ts,
                                EXIT_EOD,
                                risk,
                            )
                        )

                    # ---- D/E. decisions at this bar's CLOSE, using history
                    # that ends at this bar. Fills happen next bar.
                    history = df.iloc[: i + 1]
                    strat = strategies[sym]

                    if sym in positions:
                        if strat.should_exit(history, positions[sym].side):
                            pending_exits[sym] = EXIT_STRATEGY
                        continue

                    if is_last_bar:
                        continue  # nothing to fill tomorrow morning

                    decision = risk.can_enter(ts.time(), float(row["close"]), reference)
                    if not decision.allowed:
                        continue

                    signal = strat.generate_signal(history)
                    if signal is None:
                        continue
                    signals_generated += 1

                    qty = risk.position_size(float(row["close"]), signal.stop_loss)
                    if qty <= 0:
                        signals_sized_out += 1
                        continue

                    pending_entries[sym] = PendingEntry(
                        symbol=sym,
                        side=signal.side,
                        quantity=qty,
                        stop_loss=signal.stop_loss,
                        target=signal.target,
                        reason=signal.reason,
                    )
                    # Reserve the position slot NOW, not when the fill
                    # happens next bar. Without this, every symbol processed
                    # within the same bar iteration sees the same (stale)
                    # risk.open_positions count and the max_positions guard
                    # can be oversubscribed across symbols on one bar.
                    risk.open_positions += 1

                # end symbols loop
            # end bars loop

            for sym, df in day_data.items():
                prev_close[sym] = float(df["close"].iloc[-1])

        for t in trades:
            t.strategy = strategy_name

        report = build_report(trades, self.risk_config.starting_capital)
        return BacktestResult(
            strategy=strategy_name,
            symbols=symbols,
            trades=trades,
            report=report,
            blocked=dict(risk.blocked_counts),
            signals_generated=signals_generated,
            signals_sized_out=signals_sized_out,
            starting_capital=self.risk_config.starting_capital,
        )

    # ------------------------------------------------------------------
    def _check_stop_target(self, pos: OpenPosition, row) -> tuple:
        """Return (exit_price, reason) or (None, None).

        Stop is evaluated before target -- see the module docstring.
        """
        high = float(row["high"])
        low = float(row["low"])
        if pos.side == LONG:
            if low <= pos.stop_loss:
                return pos.stop_loss, EXIT_STOP
            if high >= pos.target:
                return pos.target, EXIT_TARGET
        else:
            if high >= pos.stop_loss:
                return pos.stop_loss, EXIT_STOP
            if low <= pos.target:
                return pos.target, EXIT_TARGET
        return None, None

    def _open(
        self, pe: PendingEntry, price: float, ts: datetime, risk: RiskManager
    ) -> Optional[OpenPosition]:
        side_str = BUY if pe.side == LONG else SELL
        fill = self.broker.place_order(
            Order(symbol=pe.symbol, side=side_str, quantity=pe.quantity, timestamp=ts),
            reference_price=price,
        )
        # Guard against a gap through the stop at the open: if the fill price
        # is already past the stop the trade makes no sense, so skip it.
        if pe.side == LONG and fill.price <= pe.stop_loss:
            self._unwind(pe, fill, ts)
            risk.open_positions = max(risk.open_positions - 1, 0)  # release reservation
            return None
        if pe.side == SHORT and fill.price >= pe.stop_loss:
            self._unwind(pe, fill, ts)
            risk.open_positions = max(risk.open_positions - 1, 0)  # release reservation
            return None

        # NOTE: risk.open_positions was already incremented at signal time
        # (the reservation) -- do not increment it again here.
        return OpenPosition(
            symbol=pe.symbol,
            side=pe.side,
            quantity=pe.quantity,
            entry_price=fill.price,
            entry_time=ts,
            stop_loss=pe.stop_loss,
            target=pe.target,
            entry_costs=fill.charges.total if self.apply_costs else 0.0,
            reason=pe.reason,
        )

    def _unwind(self, pe: PendingEntry, fill, ts) -> None:
        """Immediately reverse an entry we decided not to keep."""
        opposite = SELL if pe.side == LONG else BUY
        self.broker.place_order(
            Order(symbol=pe.symbol, side=opposite, quantity=pe.quantity, timestamp=ts),
            reference_price=fill.price,
        )

    def _close(
        self,
        pos: OpenPosition,
        price: float,
        ts: datetime,
        reason: str,
        risk: RiskManager,
    ) -> Trade:
        side_str = SELL if pos.side == LONG else BUY
        fill = self.broker.place_order(
            Order(symbol=pos.symbol, side=side_str, quantity=pos.quantity, timestamp=ts),
            reference_price=price,
        )
        exit_costs = fill.charges.total if self.apply_costs else 0.0
        gross = pos.side * (fill.price - pos.entry_price) * pos.quantity
        total_costs = pos.entry_costs + exit_costs

        risk.open_positions = max(risk.open_positions - 1, 0)
        risk.record_trade(gross - total_costs)

        return Trade(
            symbol=pos.symbol,
            strategy="",
            side=pos.side,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=ts,
            exit_price=fill.price,
            exit_reason=reason,
            gross_pnl=gross,
            costs=total_costs,
        )
