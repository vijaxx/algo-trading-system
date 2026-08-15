"""Command line interface.

    python -m algotrade.cli backtest --strategy orb
    python -m algotrade.cli backtest --strategy all --days 60
    python -m algotrade.cli costs --price 1000 --qty 100
    python -m algotrade.cli list-strategies
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import List

from .costs import DEFAULT_COSTS
from .data import SYMBOLS, generate_dataset
from .engine import BacktestEngine
from .risk import RiskConfig
from .strategies import STRATEGY_REGISTRY, build_strategy


# ---------------------------------------------------------------- printing
def _table(rows: List, headers: List) -> str:
    """Minimal fixed-width table (no external dependency)."""
    all_rows = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    sep = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), sep]
    for r in all_rows[1:]:
        out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def _rule(title: str = "") -> str:
    if title:
        return f"\n{'=' * 78}\n{title}\n{'=' * 78}"
    return "=" * 78


# ---------------------------------------------------------------- commands
def cmd_list_strategies(args) -> int:
    rows = []
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        rows.append([name, cls.__name__, doc])
    print(_table(rows, ["key", "class", "description"]))
    return 0


def cmd_costs(args) -> int:
    model = DEFAULT_COSTS
    buy = model.order_cost(args.price, args.qty, "BUY")
    sell_price = args.price * (1 + args.move_pct / 100.0)
    sell = model.order_cost(sell_price, args.qty, "SELL")

    print(_rule(f"Round trip: BUY {args.qty} @ {args.price:.2f} -> "
                f"SELL @ {sell_price:.2f}"))
    rows = []
    for label, key in [
        ("Turnover", "turnover"),
        ("Brokerage", "brokerage"),
        ("STT", "stt"),
        ("Exchange txn charges", "exchange_txn"),
        ("SEBI turnover fees", "sebi_fees"),
        ("Stamp duty", "stamp_duty"),
        ("GST @ 18%", "gst"),
        ("TOTAL", "total"),
    ]:
        b = buy.as_dict()[key]
        s = sell.as_dict()[key]
        rows.append([label, f"{b:,.4f}", f"{s:,.4f}", f"{b + s:,.4f}"])
    print(_table(rows, ["component", "buy leg (Rs)", "sell leg (Rs)", "total (Rs)"]))

    gross = (sell_price - args.price) * args.qty
    total_cost = buy.total + sell.total
    print()
    print(f"Gross P&L        : Rs {gross:,.2f}")
    print(f"Total charges    : Rs {total_cost:,.2f}")
    print(f"Net P&L          : Rs {gross - total_cost:,.2f}")
    print(
        f"Breakeven move   : {model.breakeven_move_pct(args.price, args.qty, 1):.4f}% "
        f"(round trip, long)"
    )
    return 0


def _run_one(name: str, data, capital: float, slippage: float, verbose: bool):
    risk_cfg = RiskConfig(starting_capital=capital)
    with_costs = BacktestEngine(
        strategy_factory=lambda: build_strategy(name),
        risk_config=risk_cfg,
        slippage_bps=slippage,
        apply_costs=True,
    ).run(data)
    without_costs = BacktestEngine(
        strategy_factory=lambda: build_strategy(name),
        risk_config=RiskConfig(starting_capital=capital),
        slippage_bps=slippage,
        apply_costs=False,
    ).run(data)
    return with_costs, without_costs


def cmd_backtest(args) -> int:
    if args.strategy == "all":
        names = sorted(STRATEGY_REGISTRY)
    else:
        if args.strategy not in STRATEGY_REGISTRY:
            print(
                f"unknown strategy {args.strategy!r}; "
                f"available: {', '.join(sorted(STRATEGY_REGISTRY))}, all",
                file=sys.stderr,
            )
            return 2
        names = [args.strategy]

    symbols = args.symbols or list(SYMBOLS)
    data = generate_dataset(
        symbols=symbols,
        n_days=args.days,
        bar_minutes=args.bar_minutes,
        seed=args.seed,
    )
    total_bars = sum(len(df) for df in data.values())

    print(_rule("Automated Algorithmic Trading System -- backtest"))
    print(f"Data      : synthetic, seed={args.seed}, {args.bar_minutes}-min bars")
    print(f"Symbols   : {', '.join(sorted(data))}")
    print(f"Sessions  : {args.days} trading days   Bars: {total_bars:,}")
    print(f"Capital   : Rs {args.capital:,.2f}   Slippage: {args.slippage} bps")
    print("Costs     : Indian equity intraday (brokerage, STT, exchange, "
          "SEBI, stamp, GST)")

    summary_rows = []
    for name in names:
        net_run, gross_run = _run_one(
            name, data, args.capital, args.slippage, args.verbose
        )
        r = net_run.report
        g = gross_run.report

        print(_rule(f"Strategy: {name}"))
        print(_table([[k, v] for k, v in r.as_rows()], ["metric", "value"]))

        if net_run.blocked:
            print("\nRisk blocks (pre-trade rejections):")
            print(
                _table(
                    [[k, str(v)] for k, v in sorted(net_run.blocked.items())],
                    ["guard", "count"],
                )
            )

        if args.verbose and net_run.trades:
            print("\nExit reason breakdown:")
            df = net_run.trades_frame()
            counts = df["exit_reason"].value_counts()
            print(
                _table(
                    [[k, str(v)] for k, v in counts.items()],
                    ["exit reason", "trades"],
                )
            )
            ps = net_run.per_symbol()
            if not ps.empty:
                print("\nPer symbol (net of costs):")
                print(
                    _table(
                        [
                            [
                                idx,
                                str(int(row["trades"])),
                                f"{row['gross_pnl']:,.2f}",
                                f"{row['costs']:,.2f}",
                                f"{row['net_pnl']:,.2f}",
                                f"{row['win_rate']:.1f}%",
                            ]
                            for idx, row in ps.iterrows()
                        ],
                        ["symbol", "trades", "gross", "costs", "net", "win%"],
                    )
                )

        summary_rows.append(
            [
                name,
                str(r.num_trades),
                f"{r.win_rate_pct:.1f}%",
                f"{g.net_pnl:,.0f}",
                f"{r.total_costs:,.0f}",
                f"{r.net_pnl:,.0f}",
                f"{r.return_pct:.2f}%",
                f"{r.sharpe_ratio:.2f}",
                f"{r.max_drawdown_pct:.2f}%",
                f"{r.profit_factor:.2f}",
            ]
        )

    print(_rule("Summary -- gross (zero-cost run) vs net (full Indian costs)"))
    print(
        _table(
            summary_rows,
            [
                "strategy",
                "trades",
                "win%",
                "gross P&L",
                "costs",
                "net P&L",
                "net ret%",
                "Sharpe",
                "maxDD%",
                "PF",
            ],
        )
    )
    print("\nPaper/simulation only. Synthetic data. Not investment advice.")
    return 0


# ---------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="algotrade",
        description="Indian intraday backtesting / paper-trading engine "
        "(simulation only -- cannot place live orders).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="run a backtest over synthetic NSE data")
    b.add_argument(
        "--strategy",
        default="all",
        help="strategy key, or 'all' (default: all). "
        f"Options: {', '.join(sorted(STRATEGY_REGISTRY))}",
    )
    b.add_argument("--symbols", nargs="*", default=None, help=f"subset of {list(SYMBOLS)}")
    b.add_argument("--days", type=int, default=60, help="trading sessions (default 60)")
    b.add_argument("--bar-minutes", type=int, default=5, dest="bar_minutes")
    b.add_argument("--capital", type=float, default=500_000.0)
    b.add_argument("--slippage", type=float, default=2.0, help="bps per fill")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("costs", help="print a worked transaction-cost example")
    c.add_argument("--price", type=float, default=1000.0)
    c.add_argument("--qty", type=int, default=100)
    c.add_argument("--move-pct", type=float, default=1.0, dest="move_pct")
    c.set_defaults(func=cmd_costs)

    l = sub.add_parser("list-strategies", help="list available strategies")
    l.set_defaults(func=cmd_list_strategies)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
