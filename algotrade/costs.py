"""Indian equity **intraday** transaction-cost model.

This is the piece most toy backtesters get wrong or skip entirely. On NSE/BSE
intraday equity the round-trip friction on a small ticket is dominated by the
flat brokerage floor and STT, and it is very easy for a strategy with a
positive gross edge to be net negative after costs.

Rate sources (all rates as of FY 2024-25, discount-broker retail equity
intraday; see README for links):

  brokerage      : min(Rs 20, 0.03% of turnover) per executed order.
                   Zerodha / Upstox / Angel One / Groww all use this same
                   "Rs 20 or 0.03%, whichever is lower" intraday slab.
  STT / CTT      : 0.025% on the SELL side turnover only (equity intraday).
                   Securities Transaction Tax Act; delivery trades are charged
                   0.1% on both sides -- not modelled here, this engine is
                   intraday-only.
  exchange txn   : NSE 0.00297% of turnover, both sides (NSE equity cash
                   segment transaction charge, Rs 297 per crore).
                   BSE equity uses the same 0.00375%/0.00297% group structure;
                   we model the NSE rate and expose it as a parameter.
  SEBI turnover  : Rs 10 per crore = 0.0001% of turnover, both sides.
  stamp duty     : 0.003% on the BUY side turnover only (intraday equity),
                   uniform across states since the Finance Act 2019
                   centralised collection.
  GST            : 18% on (brokerage + exchange transaction charges + SEBI
                   turnover fees). GST is *not* charged on STT or stamp duty.

  DP charges     : NOT applicable -- those hit delivery sells only.

Everything is computed in rupees. No rounding is applied per component (the
brokerage floor is applied per order, which is where the rounding that
actually matters happens).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- default rates (fractions of turnover, not percentages) -----------------
BROKERAGE_FLAT = 20.0
BROKERAGE_PCT = 0.0003  # 0.03%
STT_SELL_PCT = 0.00025  # 0.025% sell side, intraday equity
EXCHANGE_TXN_PCT = 0.0000297  # NSE 0.00297%
SEBI_TURNOVER_PCT = 0.000001  # Rs 10 per crore
STAMP_DUTY_BUY_PCT = 0.00003  # 0.003% buy side
GST_PCT = 0.18


@dataclass(frozen=True)
class CostBreakdown:
    """Rupee cost of a single executed order, itemised."""

    turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fees: float
    stamp_duty: float
    gst: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fees
            + self.stamp_duty
            + self.gst
        )

    def as_dict(self) -> dict:
        d = {
            "turnover": self.turnover,
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange_txn": self.exchange_txn,
            "sebi_fees": self.sebi_fees,
            "stamp_duty": self.stamp_duty,
            "gst": self.gst,
            "total": self.total,
        }
        return {k: round(v, 4) for k, v in d.items()}


@dataclass(frozen=True)
class IndianEquityIntradayCosts:
    """Configurable cost model. Defaults = discount broker, NSE, FY25 rates."""

    brokerage_flat: float = BROKERAGE_FLAT
    brokerage_pct: float = BROKERAGE_PCT
    stt_sell_pct: float = STT_SELL_PCT
    exchange_txn_pct: float = EXCHANGE_TXN_PCT
    sebi_turnover_pct: float = SEBI_TURNOVER_PCT
    stamp_duty_buy_pct: float = STAMP_DUTY_BUY_PCT
    gst_pct: float = GST_PCT

    def order_cost(self, price: float, quantity: int, side: str) -> CostBreakdown:
        """Itemised cost of one executed order.

        side: "BUY" or "SELL" (case-insensitive).
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if price < 0:
            raise ValueError("price must be non-negative")

        turnover = float(price) * int(quantity)
        if turnover == 0.0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        brokerage = min(self.brokerage_flat, self.brokerage_pct * turnover)
        exchange_txn = self.exchange_txn_pct * turnover
        sebi_fees = self.sebi_turnover_pct * turnover
        stt = self.stt_sell_pct * turnover if side_u == "SELL" else 0.0
        stamp_duty = self.stamp_duty_buy_pct * turnover if side_u == "BUY" else 0.0
        # GST applies to brokerage + exchange charges + SEBI fees only.
        gst = self.gst_pct * (brokerage + exchange_txn + sebi_fees)

        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            gst=gst,
        )

    def round_trip_cost(
        self, entry_price: float, exit_price: float, quantity: int, direction: int
    ) -> float:
        """Total rupee cost of a full round trip.

        direction: +1 for a long (buy then sell), -1 for a short (sell then
        buy). Note the *short* leg pays STT on the opening sell and stamp duty
        on the closing buy -- the sides swap, which is why direction matters
        and a naive "2 x one-side cost" approximation is wrong.
        """
        if direction not in (1, -1):
            raise ValueError("direction must be +1 (long) or -1 (short)")
        if direction == 1:
            entry = self.order_cost(entry_price, quantity, "BUY")
            exit_ = self.order_cost(exit_price, quantity, "SELL")
        else:
            entry = self.order_cost(entry_price, quantity, "SELL")
            exit_ = self.order_cost(exit_price, quantity, "BUY")
        return entry.total + exit_.total

    def breakeven_move_pct(self, price: float, quantity: int, direction: int) -> float:
        """Percentage price move needed just to cover costs.

        Handy sanity check: with a Rs 20 flat brokerage floor, a Rs 25,000
        ticket needs a far bigger move than a Rs 5,00,000 one.
        """
        cost = self.round_trip_cost(price, price, quantity, direction)
        turnover = price * quantity
        if turnover == 0:
            return 0.0
        return 100.0 * cost / turnover


DEFAULT_COSTS = IndianEquityIntradayCosts()
