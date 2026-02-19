"""
config.py — Strategy configuration and defaults for the Gamma Scalping Dashboard.

All tunable parameters are encapsulated in a dataclass for clean propagation
through indicators, strategy, and backtest modules.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class StrategyConfig:
    """Central configuration for the IV-filtered ATM Straddle Gamma Scalping strategy."""

    # ── Part 1: IV Filter ──────────────────────────────────────────────
    iv_percentile_window: int = 20          # Rolling window for IV percentile (trading days)
    iv_percentile_threshold: float = 0.35   # Max IV percentile to allow entry

    # ── Part 2: Expected Move Filter ───────────────────────────────────
    expected_move_threshold: float = 0.4    # Min expected move % (IV / 16)

    # ── Part 3: 5-min Bollinger Compression ───────────────────────────
    bb_length_5min: int = 20                # Bollinger period for 5-min data
    bb_std_5min: float = 2.0                # Bollinger std dev for 5-min data
    compression_lookback: int = 20          # Candles to look back for compression

    # ── Part 4: 1-min Entry ───────────────────────────────────────────
    bb_length_1min: int = 20                # Bollinger period for 1-min data
    bb_std_1min: float = 2.0                # Bollinger std dev for 1-min data
    body_avg_lookback: int = 10             # Candles for average body comparison

    # ── Part 5: Exit Rules ─────────────────────────────────────────────
    take_profit_pct: float = 3.0            # Take profit threshold (%)
    stop_loss_pct: float = 1.5              # Stop loss threshold (%)
    range_bound_minutes: int = 30           # Minutes of flat PnL before timeout exit
    hard_exit_time: str = "15:15"           # Hard exit time (HH:MM)

    # ── Part 6: Re-Entry ──────────────────────────────────────────────
    reentry_wait_minutes: int = 15          # Minutes to wait after exit before re-entry
    max_trades_per_day: int = 3             # Maximum trades per trading day

    # ── Transaction Costs ──────────────────────────────────────────────
    slippage_bps: float = 0.0               # Slippage in basis points
    brokerage_per_trade: float = 0.0        # Flat brokerage per trade (₹)

    # ── Market Timing ─────────────────────────────────────────────────
    market_open: str = "09:15"
    market_close: str = "15:30"
    iv_close_time: str = "15:15"            # Time to sample daily closing IV

    # ── Column Name Mappings ──────────────────────────────────────────
    col_datetime: str = "datetime"
    col_date: str = "date"
    col_time: str = "time"
    col_spot: str = "spot"
    col_iv: str = "iv"
    col_straddle: str = "Straddle_Price"
    col_ce_open: str = "CE_open"
    col_ce_high: str = "CE_high"
    col_ce_low: str = "CE_low"
    col_ce_close: str = "CE_close"
    col_pe_open: str = "PE_open"
    col_pe_high: str = "PE_high"
    col_pe_low: str = "PE_low"
    col_pe_close: str = "PE_close"


# ── Optimization Grid Defaults ────────────────────────────────────────

@dataclass
class OptimizationGrid:
    """Default parameter ranges for grid search optimization."""
    iv_percentile_range: List[float] = field(
        default_factory=lambda: [0.20, 0.25, 0.30, 0.35, 0.40]
    )
    expected_move_range: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6]
    )
    take_profit_range: List[float] = field(
        default_factory=lambda: [2.0, 3.0, 4.0, 5.0]
    )
    stop_loss_range: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5]
    )
    bb_length_range: List[int] = field(
        default_factory=lambda: [15, 20, 25, 30]
    )
