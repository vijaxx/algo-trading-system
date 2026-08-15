# algo-trading-system

An intraday strategy backtesting and paper-trading engine for the Indian
equity market (NSE/BSE), built to be **runnable, honest, and cost-accurate**
rather than a demo that only looks good in a slide deck.

Five intraday strategies, a bar-by-bar backtest engine with an explicit
no-lookahead proof, a full Indian equity intraday transaction-cost model
(brokerage, STT, exchange charges, SEBI fees, stamp duty, GST), SEBI-style
pre-trade risk guards, and a CLI that prints a results table. Everything runs
against a deterministic synthetic data generator — no API key, no data
download, no network call, no live order routing anywhere in the codebase.

```
git clone <this repo>
cd algo-trading-system
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m algotrade.cli backtest --strategy all --days 60
./.venv/bin/python -m pytest -q
```

## Why this project

Built as a portfolio piece for fintech application (S&P Global, Broadridge
style roles). The interesting engineering problems in retail algo trading
aren't the indicators — EMA and RSI are a page of pandas each — they're:
correctly modeling the cost structure that erodes a strategy's edge, proving
a backtest isn't cheating by looking at the future, and building risk
controls that actually block trades rather than just logging a warning.
Those three things are where this project puts its effort.

## Architecture

```
algotrade/
  data.py          synthetic OHLCV generator (deterministic, seeded)
  indicators.py    SMA/EMA/RSI/ATR/Bollinger/VWAP/Supertrend, pandas/numpy only
  strategies/      5 strategies behind a common Strategy interface
  engine.py        bar-by-bar backtest loop, no-lookahead by construction
  costs.py         Indian equity intraday transaction-cost model
  risk.py          SEBI-style pre-trade risk guards
  broker.py        BrokerAdapter interface + PaperBroker + unfinished live sketch
  metrics.py       P&L, win rate, Sharpe, drawdown, profit factor
  cli.py           `python -m algotrade.cli backtest|costs|list-strategies`
tests/             92 pytest tests
```

### Strategies (all implemented from scratch, no TA-Lib)

| key | strategy | idea |
|---|---|---|
| `orb` | Opening Range Breakout | trade a close beyond the first 15 minutes' high/low |
| `ema_rsi` | EMA crossover + RSI filter | fast/slow EMA cross, gated by RSI momentum confirmation |
| `vwap_reversion` | VWAP mean reversion | fade a stretch away from session VWAP back toward it |
| `supertrend` | Supertrend | ATR-banded trend-following, enter/exit on direction flips |
| `bollinger` | Bollinger Band breakout | trade a squeeze-then-expansion break of the bands |

Every indicator (`algotrade/indicators.py`) is implemented directly on
pandas/numpy — no TA-Lib, which needs a compiled C library and would break a
plain `pip install` on a clean machine.

## No-lookahead: why it matters and how it's enforced

The single most common bug in retail backtesting is a strategy that,
knowingly or not, reads a bar that hasn't happened yet — e.g. computing a
day's high/low and then "predicting" a breakout of it before the close that
actually created that high/low. Backtests built this way show fantastic
Sharpe ratios and lose money the moment they go live, because the information
they traded on didn't exist yet.

This engine prevents it structurally, not by convention:

1. `Strategy.generate_signal(history)` is passed a DataFrame **slice** that
   ends at the bar that just closed (`algotrade/strategies/base.py`). Rows
   after that bar are not present in the object — there is nothing to
   accidentally index into.
2. A `Signal` produced from bar `i`'s close is filled at bar `i + 1`'s
   **open**, never at bar `i`'s own close (`algotrade/engine.py`, `_open`).
   In live trading, a bar's close is only known once the bar is already
   gone — filling at the signal bar's own close is a subtler lookahead bug.
3. Indicators are recomputed on the growing slice at each bar rather than
   precomputed once over the full series and then indexed — this catches any
   accidental use of a centered/backward-looking pandas operation that would
   otherwise leak future values into a "past" index position.

This is verified, not just asserted, by `tests/test_no_lookahead.py`:

- **Contract test** (`test_history_slice_never_extends_past_current_bar`,
  run against all 5 strategies): confirms the history slice handed to a
  strategy at bar `i` has length exactly `i + 1` and its last timestamp is
  strictly before the next bar's timestamp.
- **Mutation test** (`test_mutating_future_bars_does_not_change_past_signals`):
  scrambles every price/volume value in the *second half* of every session
  and re-runs the strategy. Signals generated in the untouched *first half*
  are asserted byte-identical between the original and scrambled runs. If a
  strategy could see the future, scrambling it would change past decisions —
  it doesn't.
- **Fill-timing test** (`test_engine_fills_at_next_bar_open_not_signal_bar_close`):
  runs the ORB strategy with zero slippage and confirms every recorded
  `entry_price` equals the *open* of its fill bar exactly, not the close of
  an earlier bar.

## The Indian transaction-cost model

This is the part most toy backtesters skip, and it's usually the difference
between a strategy that looks good and one that actually works. All rates
below are the FY 2024-25 retail discount-broker structure for NSE equity
**intraday** (MIS) trades; see `algotrade/costs.py` for the full citations
in comments.

| component | rate | side |
|---|---|---|
| Brokerage | min(₹20, 0.03% of turnover) per executed order | both |
| STT | 0.025% of turnover | sell only |
| NSE exchange transaction charge | 0.00297% of turnover | both |
| SEBI turnover fee | ₹10 per crore (0.0001%) | both |
| Stamp duty | 0.003% of turnover | buy only |
| GST | 18% of (brokerage + exchange charge + SEBI fee) | — |

GST is **not** charged on STT or stamp duty — a mistake several hobby cost
models make. Note also that a short round trip pays STT on the *opening*
sell and stamp duty on the *closing* buy — the sides are swapped relative to
a long, which is why `round_trip_cost(direction=-1)` computes the legs
separately rather than doubling one side's cost.

### Worked example

`python -m algotrade.cli costs --price 1000 --qty 100 --move-pct 1.0` — buy
100 shares at ₹1,000, sell at ₹1,010 (a 1% favorable move):

```
component            | buy leg (Rs) | sell leg (Rs) | total (Rs)
----------------------+--------------+---------------+-------------
Turnover              | 100,000.0000 | 101,000.0000  | 201,000.0000
Brokerage              | 20.0000      | 20.0000       | 40.0000
STT                    | 0.0000       | 25.2500       | 25.2500
Exchange txn charges   | 2.9700       | 2.9997        | 5.9697
SEBI turnover fees     | 0.1000       | 0.1010        | 0.2010
Stamp duty              | 3.0000       | 0.0000        | 3.0000
GST @ 18%               | 4.1526       | 4.1581        | 8.3107
TOTAL                   | 30.2226      | 52.5088       | 82.7314

Gross P&L        : Rs 1,000.00
Total charges    : Rs 82.73
Net P&L          : Rs 917.27
Breakeven move    : 0.0824% (round trip, long)
```

Costs ate 8.3% of the gross profit on a fairly large (₹1 lakh) ticket with a
clean 1% move. On a smaller ticket the flat ₹20 brokerage floor dominates and
the breakeven move is much larger — see `test_breakeven_move_shrinks_with_larger_ticket`
in `tests/test_costs.py`.

## Risk / SEBI-style compliance guards

`algotrade/risk.py` implements pre-trade guards modeled on what a broker's
risk management system enforces before an order reaches the exchange. These
**block** trades — they don't just log a warning:

- **Max daily loss** — a kill switch once realised P&L breaches a % of
  starting capital for the session.
- **Max open positions** — a concentration cap across the whole book.
- **Position sizing** — the smaller of (a) a % of equity allocated per trade
  and (b) a per-trade risk budget divided by the stop distance, so a wide
  stop doesn't silently size up the rupee risk.
- **Square-off enforcement** — no new entries after 15:00, forced flatten at
  15:15 IST (broker MIS auto-square-off convention), and a hard end-of-day
  flatten in the engine as a second safety net.
- **Circuit breaker check** — refuses to trade a symbol that has already
  moved beyond the configured exchange price band from its session reference
  price.

All five are exercised by `tests/test_risk.py`, and `tests/test_engine.py`
proves them end to end against real backtest runs — e.g.
`test_max_positions_cap_is_never_exceeded` reconstructs concurrent open
positions from the trade log and asserts the cap was never exceeded across a
30-day, 3-symbol run.

**A real bug this caught:** the first version of the engine incremented the
"open positions" counter only when a position actually filled (one bar after
the signal), so multiple symbols evaluated within the *same* bar could each
see a stale, pre-increment count and all queue an entry — oversubscribing
the `max_positions` cap. The fix reserves the slot at signal time instead of
at fill time. `test_max_positions_cap_is_never_exceeded` failed against the
original code and passes against the fix — exactly the kind of bug an
integration test across the risk guard and the engine is meant to catch.

## Measured results

All numbers below are from an actual run, not invented. Reproduce with:

```
./.venv/bin/python -m algotrade.cli backtest --strategy all --days 60 --seed 42
```

60 trading sessions, 6 synthetic NSE symbols (RELIANCE, TCS, HDFCBANK, INFY,
SBIN, TATAMOTORS), 5-minute bars, ₹500,000 starting capital, 2 bps slippage,
`max_positions=2`, `max_daily_loss_pct=2`, default strategy parameters.

| strategy | trades | win% | gross P&L | costs | **net P&L** | net return | Sharpe | max DD% | profit factor |
|---|---|---|---|---|---|---|---|---|---|
| supertrend | 322 | 51.9% | ₹123,348 | ₹27,491 | **₹95,857** | 19.17% | 10.56 | 0.97% | 2.09 |
| ema_rsi | 349 | 57.0% | ₹68,897 | ₹29,182 | **₹39,715** | 7.94% | 7.46 | 0.67% | 1.50 |
| bollinger | 399 | 58.6% | ₹67,674 | ₹33,272 | **₹34,401** | 6.88% | 6.01 | 0.70% | 1.38 |
| orb | 257 | 48.2% | ₹17,535 | ₹20,943 | **-₹3,408** | -0.68% | -0.19 | 5.01% | 0.98 |
| vwap_reversion | 1,012 | 22.6% | -₹112,396 | ₹75,494 | **-₹187,890** | -37.58% | -30.03 | 37.19% | 0.37 |

Full per-strategy output (all metrics, plus the risk-guard block counts) is
reproduced by the command above.

Honest read of these numbers:

- **ORB is roughly cost-neutral and net negative.** It made ₹17.5k gross
  over 60 days on 257 trades and lost more than that back to costs
  (₹20.9k), for a net loss of ₹3.4k. Cost drag (costs / gross) was 119%.
  This is the realistic outcome for a strategy with a modest per-trade edge
  and a moderate trade count — it is *not* tuned to look good.
- **VWAP reversion is a clear loser, both gross and net.** 22.6% win rate,
  profit factor 0.37, -37.6% return, and a 37% max drawdown on 1,012
  trades. Reported as-is rather than dropped from the table — a strategy
  losing money after costs is a more credible result than a suspiciously
  clean sweep of winners, and this repo would rather show that than hide it.
- **Supertrend, EMA/RSI and Bollinger were net winners** on this synthetic
  data, with cost drag between 22% and 49% of gross profit — a concrete
  illustration of how much of an intraday strategy's edge the Indian cost
  stack can consume even when the strategy is net profitable.
- **The Sharpe ratios above (6–10 for the winners) are high for what would
  be a "real" strategy** and should not be read as evidence of genuine
  edge — see Limitations. They reflect that this synthetic data's AR(1)
  price process has cleaner trend/reversion structure than real markets, and
  the winning strategies are the ones whose bias (trend-following /
  momentum) matches that structure.
- **Every entry is routed through the risk guards**, and they fire
  constantly: in the full run, `max_positions reached` blocked between
  ~2,900 and ~15,500 signals per strategy (cap of 2 concurrent positions
  across 6 symbols), `past no-new-entry cutoff` and `past square-off time`
  block signals near the close every single day, and the circuit breaker
  fired a handful of times per strategy. These aren't decorative — see the
  bug note above.

## Test suite

```
./.venv/bin/python -m pytest -q
```

**92 tests, all passing**, covering:

- `test_indicators.py` (17) — SMA/EMA/RSI/ATR/Bollinger/VWAP/Supertrend
  against hand-computed values (e.g. RSI = 100 on an all-up series, EMA
  matched against a manual recursion, Bollinger bands against `mean ± 2·std`).
- `test_costs.py` (13) — brokerage floor vs percentage, STT sell-only, stamp
  duty buy-only, GST base, round-trip long vs short leg swapping, breakeven
  move scaling with ticket size.
- `test_risk.py` (16) — every guard (daily loss, position count, entry-time
  cutoff, square-off, circuit breaker) proven to actually block, plus
  position sizing arithmetic.
- `test_no_lookahead.py` (11, parametrized across strategies) — the
  contract, mutation, and fill-timing proofs described above.
- `test_metrics.py` (14) — win rate, profit factor (including the
  no-losses/no-trades edge cases), Sharpe, max drawdown, cost drag, all
  against hand-computed values.
- `test_data.py` (8) — determinism (same seed ⇒ byte-identical output),
  OHLC invariants, session/weekday handling.
- `test_broker.py` (8) — PaperBroker fills/slippage/funds/position netting,
  and a proof that `SketchLiveBrokerAdapter` raises `NotImplementedError` on
  every method that would touch a real account.
- `test_engine.py` (8) — square-off enforcement, the max-positions cap
  under concurrency, daily-loss-limit trade-count reduction, cost
  application on/off, and per-symbol P&L reconciling to the total.

## Broker integration — paper only, by design

`algotrade/broker.py` defines `BrokerAdapter` as an abstract interface with
one working implementation: `PaperBroker`, a full simulation with a
slippage model and the Indian cost model wired in.

`SketchLiveBrokerAdapter` is included to show where a real broker (Angel One
SmartAPI, Fyers API v3, Upstox API v2) would plug into the same interface.
**Every method on it raises `NotImplementedError`.** There is no HTTP
client, no auth flow, no API key handling, and no endpoint URL anywhere in
this class or in this repository. This is enforced by
`test_sketch_live_adapter_cannot_connect_or_trade` in `tests/test_broker.py`.

**This project cannot place a real order and never will, by design.**

## Limitations (honest)

- **Synthetic data only.** The market data is a deterministic seeded
  generator (`algotrade/data.py`) with an AR(1) intraday shock process,
  overnight gaps, and a U-shaped volume/volatility profile — it captures
  the *structural* features intraday strategies react to, but it is not
  real NSE data and the measured results above say nothing about live
  performance. In particular the strategies' edges likely reflect fitting
  the generator's own trend/reversion parameter, not a real market
  inefficiency.
- **Paper trading only, no live order routing.** See above.
- **Slippage model is a flat basis-point spread** applied to every market
  order (2 bps default), not a real order-book/impact model. It does not
  scale with order size, volatility, or liquidity.
- **No partial fills or queue position.** Every order fills completely at
  one price.
- **Stop-before-target intrabar assumption.** When a bar's range contains
  both the stop and the target, the engine assumes the stop was hit first
  (the pessimistic assumption, documented in `engine.py`) because OHLC bars
  don't record the intrabar path. This is conservative but still an
  approximation of the true fill.
- **No exchange holiday calendar** — the data generator treats every weekday
  as a trading day.
- **Single-machine backtest, not a live execution loop.** There is no
  scheduler, no real-time data feed handling, and no reconnection/retry
  logic — none of which would matter without a real broker connection
  anyway.
- **Position sizing and risk parameters are illustrative defaults**, not
  calibrated to any real account or regulatory minimum.

## Requirements

- Python 3.9+
- `pandas`, `numpy` (runtime); `pytest` (tests)
- No TA-Lib, no API keys, no network access needed at any point.

## License

MIT.
