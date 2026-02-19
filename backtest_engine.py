"""
backtest_engine.py — Event-driven minute-by-minute backtester and performance analytics.

Walks through each trading day chronologically:
  1. Pre-checks daily-level filters (IV percentile, expected move)
  2. Pre-computes 5-min compression flags
  3. Iterates 1-min bars for entry/exit signals
  4. Records trades and computes comprehensive performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from itertools import product

from config import StrategyConfig, OptimizationGrid
from indicators import (
    compute_bollinger,
    compute_candle_body,
    compute_avg_body,
    detect_compression,
    prepare_daily_filters,
)
from strategy import detect_entry, check_exit, can_reenter


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare(filepath: str, config: StrategyConfig) -> pd.DataFrame:
    """
    Load CSV and prepare datetime index with proper types.

    Args:
        filepath: Path to CSV file.
        config: Strategy configuration for column names.

    Returns:
        DataFrame with parsed datetime, sorted chronologically.
    """
    df = pd.read_csv(filepath)

    # Parse datetime
    if config.col_datetime in df.columns:
        df[config.col_datetime] = pd.to_datetime(df[config.col_datetime])
    elif config.col_date in df.columns and config.col_time in df.columns:
        df[config.col_datetime] = pd.to_datetime(
            df[config.col_date].astype(str) + " " + df[config.col_time].astype(str)
        )

    # Ensure date and time columns exist as strings
    if config.col_date not in df.columns:
        df[config.col_date] = df[config.col_datetime].dt.date.astype(str)
    else:
        df[config.col_date] = df[config.col_date].astype(str)

    if config.col_time not in df.columns:
        df[config.col_time] = df[config.col_datetime].dt.strftime("%H:%M")
    else:
        # Normalize time strings to HH:MM
        df[config.col_time] = pd.to_datetime(
            df[config.col_time], format="mixed"
        ).dt.strftime("%H:%M:%S")

    df.sort_values(config.col_datetime, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def precompute_1min_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Add 1-min Bollinger Bands and candle body indicators to DataFrame (in-place columns).
    """
    spot = df[config.col_spot]

    mid, upper, lower, bw = compute_bollinger(spot, config.bb_length_1min, config.bb_std_1min)
    df["bb_mid_1m"] = mid
    df["bb_upper_1m"] = upper
    df["bb_lower_1m"] = lower
    df["bb_bw_1m"] = bw

    # Use spot as proxy for open/close at 1-min level
    # If CE/PE open/close exist we use spot for BB; body uses spot shift
    # Approximate open as previous bar's close
    df["spot_open"] = df[config.col_spot].shift(1)
    df["candle_body"] = compute_candle_body(df["spot_open"], spot)
    df["avg_body"] = compute_avg_body(df["candle_body"], config.body_avg_lookback)

    return df


def precompute_5min_compression(df_5min: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Add compression flags to 5-min DataFrame.
    """
    spot = df_5min[config.col_spot]
    _, _, _, bw = compute_bollinger(spot, config.bb_length_5min, config.bb_std_5min)
    df_5min["bb_bw_5m"] = bw
    df_5min["compression"] = detect_compression(bw, config.compression_lookback)
    return df_5min


# ═══════════════════════════════════════════════════════════════════════
# MAIN BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(df_1min: pd.DataFrame,
                 df_5min: pd.DataFrame,
                 config: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Execute the full event-driven backtest.

    Args:
        df_1min: Prepared 1-minute DataFrame.
        df_5min: Prepared 5-minute DataFrame with compression flags.
        config: Strategy configuration.

    Returns:
        Tuple of (trade_log DataFrame, equity_curve DataFrame, metrics dict).
    """
    # ── Step 1: Compute daily filters ──────────────────────────────────
    daily_filters = prepare_daily_filters(
        df_1min,
        iv_window=config.iv_percentile_window,
        iv_threshold=config.iv_percentile_threshold,
        em_threshold=config.expected_move_threshold,
        col_date=config.col_date,
        col_time=config.col_time,
        col_iv=config.col_iv,
        close_time=config.iv_close_time,
    )

    # ── Step 2: Pre-compute 1-min indicators ──────────────────────────
    df_1min = precompute_1min_indicators(df_1min, config)

    # ── Step 3: Build 5-min compression lookup ─────────────────────────
    df_5min = precompute_5min_compression(df_5min, config)

    # Create a lookup: for each 1-min datetime, find the most recent 5-min compression flag
    compression_lookup = _build_compression_lookup(df_1min, df_5min, config)

    # ── Step 4: Event-driven walk ──────────────────────────────────────
    trades: List[Dict] = []
    equity_points: List[Dict] = []

    trading_days = df_1min[config.col_date].unique()
    cumulative_pnl = 0.0

    for day in trading_days:
        # Check daily filter
        if day not in daily_filters.index:
            continue

        day_filter = daily_filters.loc[day]
        daily_allowed = bool(day_filter.get("trade_allowed", False))

        day_mask = df_1min[config.col_date] == day
        day_data = df_1min.loc[day_mask].copy()

        if len(day_data) == 0:
            continue

        # State variables for the day
        in_position = False
        entry_price = 0.0
        entry_time = None
        entry_bar_idx = 0
        pnl_history: List[float] = []
        trades_today = 0
        last_exit_bar_idx = -999

        for i, (idx, row) in enumerate(day_data.iterrows()):
            current_time = str(row[config.col_time])[:5]  # HH:MM
            straddle = row[config.col_straddle]
            spot = row[config.col_spot]

            if pd.isna(straddle) or pd.isna(spot):
                continue

            if in_position:
                # ── CHECK EXIT ──────────────────────────────────────
                pnl_pct = (straddle - entry_price) / entry_price * 100.0
                pnl_history.append(pnl_pct)

                should_exit, reason = check_exit(
                    entry_price, straddle, i - entry_bar_idx,
                    pnl_history, current_time, config
                )

                if should_exit:
                    # Apply slippage
                    exit_price = _apply_slippage(straddle, config.slippage_bps, is_exit=True)
                    raw_pnl = exit_price - entry_price
                    net_pnl = raw_pnl - config.brokerage_per_trade
                    net_pnl_pct = (net_pnl / entry_price) * 100.0

                    cumulative_pnl += net_pnl

                    trades.append({
                        "trade_num": trades_today + 1,
                        "date": day,
                        "entry_time": entry_time,
                        "exit_time": row[config.col_datetime],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": net_pnl,
                        "pnl_pct": net_pnl_pct,
                        "holding_minutes": i - entry_bar_idx,
                        "exit_reason": reason,
                        "cumulative_pnl": cumulative_pnl,
                    })

                    in_position = False
                    trades_today += 1
                    last_exit_bar_idx = i
                    pnl_history = []

            else:
                # ── CHECK ENTRY ─────────────────────────────────────
                # Re-entry gating
                if trades_today > 0:
                    if not can_reenter(last_exit_bar_idx, i, trades_today, config):
                        continue

                # Don't enter too close to market close
                if current_time >= config.hard_exit_time:
                    continue

                # Get compression status
                comp_active = compression_lookup.get(idx, False)

                # Check entry
                upper_bb = row.get("bb_upper_1m", np.nan)
                lower_bb = row.get("bb_lower_1m", np.nan)
                body = row.get("candle_body", np.nan)
                avg_b = row.get("avg_body", np.nan)

                if pd.isna(upper_bb) or pd.isna(lower_bb):
                    continue

                if detect_entry(spot, upper_bb, lower_bb, body, avg_b,
                                comp_active, daily_allowed):
                    entry_price = _apply_slippage(straddle, config.slippage_bps, is_exit=False)
                    entry_time = row[config.col_datetime]
                    entry_bar_idx = i
                    in_position = True
                    pnl_history = []

            # Record equity point
            equity_points.append({
                "datetime": row[config.col_datetime],
                "cumulative_pnl": cumulative_pnl,
                "spot": spot,
            })

        # ── Force close at EOD if still in position ────────────────────
        if in_position and len(day_data) > 0:
            last_row = day_data.iloc[-1]
            exit_price = _apply_slippage(last_row[config.col_straddle],
                                          config.slippage_bps, is_exit=True)
            raw_pnl = exit_price - entry_price
            net_pnl = raw_pnl - config.brokerage_per_trade
            net_pnl_pct = (net_pnl / entry_price) * 100.0
            cumulative_pnl += net_pnl

            trades.append({
                "trade_num": trades_today + 1,
                "date": day,
                "entry_time": entry_time,
                "exit_time": last_row[config.col_datetime],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": net_pnl,
                "pnl_pct": net_pnl_pct,
                "holding_minutes": len(day_data) - entry_bar_idx - 1,
                "exit_reason": "eod_force_close",
                "cumulative_pnl": cumulative_pnl,
            })

    # ── Build outputs ──────────────────────────────────────────────────
    trade_log = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "trade_num", "date", "entry_time", "exit_time", "entry_price",
        "exit_price", "pnl", "pnl_pct", "holding_minutes", "exit_reason",
        "cumulative_pnl"
    ])

    equity_df = pd.DataFrame(equity_points) if equity_points else pd.DataFrame(
        columns=["datetime", "cumulative_pnl", "spot"]
    )

    metrics = compute_metrics(trade_log, equity_df)

    return trade_log, equity_df, metrics


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _build_compression_lookup(df_1min: pd.DataFrame,
                               df_5min: pd.DataFrame,
                               config: StrategyConfig) -> Dict:
    """
    Build a lookup dict mapping 1-min DataFrame index → compression status.

    Uses merge_asof to assign each 1-min bar the most recent 5-min compression flag.
    """
    if df_5min.empty or "compression" not in df_5min.columns:
        return {}

    df5 = df_5min[[config.col_datetime, "compression"]].copy()
    df5.sort_values(config.col_datetime, inplace=True)

    df1 = df_1min[[config.col_datetime]].copy()
    df1["_orig_idx"] = df_1min.index
    df1.sort_values(config.col_datetime, inplace=True)

    merged = pd.merge_asof(
        df1, df5,
        on=config.col_datetime,
        direction="backward"
    )

    return dict(zip(merged["_orig_idx"], merged["compression"].fillna(False)))


def _apply_slippage(price: float, slippage_bps: float, is_exit: bool) -> float:
    """
    Apply slippage to a price. Slippage hurts: higher entry, lower exit.
    """
    if slippage_bps <= 0:
        return price
    factor = slippage_bps / 10000.0
    if is_exit:
        return price * (1 - factor)
    else:
        return price * (1 + factor)


# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(trade_log: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive performance metrics from the trade log.

    Returns:
        Dictionary of performance metrics.
    """
    metrics: Dict = {}

    if trade_log.empty:
        return {
            "total_trades": 0, "total_return": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "max_drawdown": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0,
            "avg_holding_minutes": 0, "cagr": 0,
        }

    pnls = trade_log["pnl"]
    pnl_pcts = trade_log["pnl_pct"]

    # Basic counts
    metrics["total_trades"] = len(trade_log)
    metrics["total_return"] = pnls.sum()
    metrics["total_return_pct"] = pnl_pcts.sum()

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    metrics["winning_trades"] = len(wins)
    metrics["losing_trades"] = len(losses)
    metrics["win_rate"] = len(wins) / len(trade_log) * 100 if len(trade_log) > 0 else 0

    metrics["avg_win"] = wins.mean() if len(wins) > 0 else 0
    metrics["avg_loss"] = losses.mean() if len(losses) > 0 else 0
    metrics["avg_win_pct"] = pnl_pcts[pnls > 0].mean() if len(wins) > 0 else 0
    metrics["avg_loss_pct"] = pnl_pcts[pnls <= 0].mean() if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    if not equity_df.empty and "cumulative_pnl" in equity_df.columns:
        eq = equity_df["cumulative_pnl"]
        running_max = eq.cummax()
        drawdown = eq - running_max
        metrics["max_drawdown"] = drawdown.min()
        metrics["max_drawdown_pct"] = (
            (drawdown / running_max.replace(0, np.nan)).min() * 100
            if running_max.max() > 0 else 0
        )
    else:
        metrics["max_drawdown"] = 0
        metrics["max_drawdown_pct"] = 0

    # Daily returns for Sharpe/Sortino
    if "date" in trade_log.columns:
        daily_pnl = trade_log.groupby("date")["pnl"].sum()
        if len(daily_pnl) > 1:
            metrics["sharpe_ratio"] = (
                daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                if daily_pnl.std() > 0 else 0
            )
            downside = daily_pnl[daily_pnl < 0]
            downside_std = downside.std() if len(downside) > 1 else 0
            metrics["sortino_ratio"] = (
                daily_pnl.mean() / downside_std * np.sqrt(252)
                if downside_std > 0 else 0
            )
        else:
            metrics["sharpe_ratio"] = 0
            metrics["sortino_ratio"] = 0
    else:
        metrics["sharpe_ratio"] = 0
        metrics["sortino_ratio"] = 0

    # Holding time
    metrics["avg_holding_minutes"] = trade_log["holding_minutes"].mean()
    metrics["max_holding_minutes"] = trade_log["holding_minutes"].max()

    # CAGR
    if "entry_time" in trade_log.columns and len(trade_log) > 0:
        trade_log_dt = trade_log.copy()
        trade_log_dt["entry_time"] = pd.to_datetime(trade_log_dt["entry_time"])
        trade_log_dt["exit_time"] = pd.to_datetime(trade_log_dt["exit_time"])
        first_date = trade_log_dt["entry_time"].min()
        last_date = trade_log_dt["exit_time"].max()
        years = (last_date - first_date).days / 365.25
        if years > 0 and metrics["total_return"] > 0:
            # Use initial capital assumption of first trade entry price
            initial = trade_log["entry_price"].iloc[0]
            final = initial + metrics["total_return"]
            if final > 0 and initial > 0:
                metrics["cagr"] = ((final / initial) ** (1 / years) - 1) * 100
            else:
                metrics["cagr"] = 0
        else:
            metrics["cagr"] = 0
    else:
        metrics["cagr"] = 0

    # Exit reason breakdown
    if "exit_reason" in trade_log.columns:
        metrics["exit_reasons"] = trade_log["exit_reason"].value_counts().to_dict()

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# PARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def run_optimization(df_1min: pd.DataFrame,
                     df_5min: pd.DataFrame,
                     base_config: StrategyConfig,
                     param1_name: str,
                     param1_values: list,
                     param2_name: str,
                     param2_values: list,
                     progress_callback=None) -> pd.DataFrame:
    """
    Grid search optimization over two parameters, computing Sharpe for each combo.

    Args:
        df_1min: Prepared 1-minute DataFrame.
        df_5min: Prepared 5-minute DataFrame.
        base_config: Base strategy configuration.
        param1_name: First parameter name (must be StrategyConfig field).
        param1_values: List of values for first parameter.
        param2_name: Second parameter name.
        param2_values: List of values for second parameter.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        DataFrame with columns: param1, param2, sharpe, total_return, win_rate, total_trades
    """
    results = []
    total = len(param1_values) * len(param2_values)
    current = 0

    for v1, v2 in product(param1_values, param2_values):
        # Clone config and set parameters
        config_dict = asdict(base_config)
        config_dict[param1_name] = v1
        config_dict[param2_name] = v2
        cfg = StrategyConfig(**config_dict)

        try:
            trade_log, equity_df, metrics = run_backtest(
                df_1min.copy(), df_5min.copy(), cfg
            )
            results.append({
                param1_name: v1,
                param2_name: v2,
                "sharpe": metrics.get("sharpe_ratio", 0),
                "total_return": metrics.get("total_return", 0),
                "win_rate": metrics.get("win_rate", 0),
                "total_trades": metrics.get("total_trades", 0),
                "profit_factor": metrics.get("profit_factor", 0),
            })
        except Exception:
            results.append({
                param1_name: v1,
                param2_name: v2,
                "sharpe": 0, "total_return": 0,
                "win_rate": 0, "total_trades": 0,
                "profit_factor": 0,
            })

        current += 1
        if progress_callback:
            progress_callback(current, total)

    return pd.DataFrame(results)
