"""
strategy.py — Entry detection, exit logic, and re-entry rules for the Gamma Scalping strategy.

This module contains the core trading logic, separated from the backtest engine.
All functions are deterministic and operate on pre-computed indicator data.
"""

import pandas as pd
import numpy as np
from config import StrategyConfig


def detect_entry(spot: float, upper_bb: float, lower_bb: float,
                 candle_body: float, avg_body: float,
                 compression_active: bool,
                 daily_filter_pass: bool) -> bool:
    """
    Check if all entry conditions are satisfied at the current bar.

    Conditions (all must be True):
        1. Daily IV filter + expected move filter passed (daily_filter_pass)
        2. 5-min Bollinger compression is active (compression_active)
        3. Spot closes above upper BB OR below lower BB
        4. Current candle body > average body of last 10 candles

    Args:
        spot: Current spot close price (1-min).
        upper_bb: Current 1-min upper Bollinger Band.
        lower_bb: Current 1-min lower Bollinger Band.
        candle_body: Current candle body (abs(close - open)).
        avg_body: Average candle body of last N candles.
        compression_active: Whether 5-min BB compression is detected.
        daily_filter_pass: Whether the day passes IV + expected move filters.

    Returns:
        True if entry signal fires.
    """
    if not daily_filter_pass:
        return False

    if not compression_active:
        return False

    # Check for BB breakout
    bb_breakout = (spot > upper_bb) or (spot < lower_bb)
    if not bb_breakout:
        return False

    # Check body condition — ensure avg_body is valid and > 0
    if pd.isna(avg_body) or avg_body <= 0:
        return False

    if candle_body <= avg_body:
        return False

    return True


def check_exit(entry_straddle: float, current_straddle: float,
               bars_in_trade: int, pnl_history: list,
               current_time_str: str,
               config: StrategyConfig) -> tuple:
    """
    Check if any exit condition is triggered.

    Exit conditions:
        1. PnL >= take_profit_pct%
        2. PnL <= -stop_loss_pct%
        3. PnL range-bound for range_bound_minutes consecutive minutes
        4. Time >= hard_exit_time

    Args:
        entry_straddle: Entry straddle price.
        current_straddle: Current straddle price.
        bars_in_trade: Number of 1-min bars since entry.
        pnl_history: List of PnL % values for each bar in trade.
        current_time_str: Current time as "HH:MM" string.
        config: Strategy configuration.

    Returns:
        Tuple of (should_exit: bool, reason: str).
    """
    if pd.isna(current_straddle) or pd.isna(entry_straddle) or entry_straddle <= 0:
        return False, ""

    pnl_pct = (current_straddle - entry_straddle) / entry_straddle * 100.0

    # 1. Take profit
    if pnl_pct >= config.take_profit_pct:
        return True, "take_profit"

    # 2. Stop loss
    if pnl_pct <= -config.stop_loss_pct:
        return True, "stop_loss"

    # 3. Hard exit at specified time
    if current_time_str >= config.hard_exit_time:
        return True, "hard_exit"

    # 4. Range-bound check
    if len(pnl_history) >= config.range_bound_minutes:
        recent = pnl_history[-config.range_bound_minutes:]
        pnl_range = max(recent) - min(recent)
        # Range-bound = PnL hasn't moved more than 0.5% in the window
        if pnl_range < 0.5:
            return True, "range_bound"

    return False, ""


def can_reenter(last_exit_bar_idx: int, current_bar_idx: int,
                trades_today: int, config: StrategyConfig) -> bool:
    """
    Check if re-entry is allowed after a previous exit.

    Conditions:
        1. At least reentry_wait_minutes minutes have passed since last exit
        2. Number of trades today < max_trades_per_day

    Args:
        last_exit_bar_idx: Index of the bar when last exit occurred.
        current_bar_idx: Current bar index.
        trades_today: Number of trades executed today so far.
        config: Strategy configuration.

    Returns:
        True if re-entry is allowed.
    """
    if trades_today >= config.max_trades_per_day:
        return False

    minutes_since_exit = current_bar_idx - last_exit_bar_idx
    if minutes_since_exit < config.reentry_wait_minutes:
        return False

    return True
