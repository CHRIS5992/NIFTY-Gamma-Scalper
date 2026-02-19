"""
indicators.py — Technical indicator computations for the Gamma Scalping strategy.

All functions are pure, stateless, and operate on pandas Series/DataFrames.
Vectorized operations are used wherever possible for performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands on a price series.

    Args:
        series: Price series (typically spot close).
        length: Rolling window for SMA.
        std: Number of standard deviations for bands.

    Returns:
        Tuple of (middle_band, upper_band, lower_band, bandwidth).
        bandwidth = (upper - lower) / middle
    """
    middle = series.rolling(window=length, min_periods=length).mean()
    rolling_std = series.rolling(window=length, min_periods=length).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    bandwidth = (upper - lower) / middle
    return middle, upper, lower, bandwidth


def compute_iv_percentile(daily_iv: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling IV percentile using min-max normalization.

    IV_percentile = (IV_today - rolling_min) / (rolling_max - rolling_min)

    Args:
        daily_iv: Series of daily closing IV values (one per trading day).
        window: Lookback window in trading days.

    Returns:
        Series of IV percentile values (0 to 1).
    """
    rolling_min = daily_iv.rolling(window=window, min_periods=window).min()
    rolling_max = daily_iv.rolling(window=window, min_periods=window).max()
    denominator = rolling_max - rolling_min
    # Avoid division by zero — set percentile to 0.5 when range is zero
    iv_pctile = np.where(
        denominator > 0,
        (daily_iv - rolling_min) / denominator,
        0.5
    )
    return pd.Series(iv_pctile, index=daily_iv.index, name="iv_percentile")


def compute_expected_move(iv: pd.Series) -> pd.Series:
    """
    Compute expected daily move using the rule-of-16 approximation.

    Expected_Move_% = IV / 16

    Args:
        iv: IV values (annualized, in percentage terms, e.g. 15.0 = 15%).

    Returns:
        Series of expected move percentages.
    """
    return (iv / 16.0).rename("expected_move_pct")


def detect_compression(bandwidth: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Detect Bollinger Band compression on a 5-minute timeframe.

    Compression = current bandwidth is the lowest in the last `lookback` candles.

    Args:
        bandwidth: Bollinger bandwidth series.
        lookback: Number of candles to look back.

    Returns:
        Boolean Series (True where compression is detected).
    """
    rolling_min_bw = bandwidth.rolling(window=lookback, min_periods=lookback).min()
    # Compression when current bandwidth equals the rolling minimum
    return (bandwidth <= rolling_min_bw) & bandwidth.notna()


def compute_candle_body(open_series: pd.Series, close_series: pd.Series) -> pd.Series:
    """
    Compute absolute candle body size.

    Body = abs(close - open)

    Args:
        open_series: Open prices.
        close_series: Close prices.

    Returns:
        Series of absolute body sizes.
    """
    return (close_series - open_series).abs().rename("candle_body")


def compute_avg_body(body_series: pd.Series, lookback: int = 10) -> pd.Series:
    """
    Compute rolling average of candle body.

    Args:
        body_series: Series of candle body sizes.
        lookback: Number of candles to average.

    Returns:
        Series of rolling average body sizes.
    """
    return body_series.rolling(window=lookback, min_periods=lookback).mean().rename("avg_body")


def extract_daily_closing_iv(df: pd.DataFrame,
                              col_date: str = "date",
                              col_time: str = "time",
                              col_iv: str = "iv",
                              close_time: str = "15:15") -> pd.Series:
    """
    Extract IV at a specific time each day (default 3:15 PM).

    If that exact timestamp is missing, takes the last available IV for the day.

    Args:
        df: DataFrame with date, time, and iv columns.
        col_date: Name of the date column.
        col_time: Name of the time column.
        col_iv: Name of the IV column.
        close_time: Time string (HH:MM) at which to sample IV.

    Returns:
        Series indexed by date with daily closing IV.
    """
    # Try exact match first
    mask_exact = df[col_time] == close_time
    exact = df.loc[mask_exact].groupby(col_date)[col_iv].last()

    # For days where exact time is missing, use last available
    all_days = df.groupby(col_date)[col_iv].last()
    daily_iv = all_days.copy()
    daily_iv.update(exact)

    return daily_iv.rename("daily_closing_iv")


def prepare_daily_filters(df_1min: pd.DataFrame,
                           iv_window: int = 20,
                           iv_threshold: float = 0.35,
                           em_threshold: float = 0.4,
                           col_date: str = "date",
                           col_time: str = "time",
                           col_iv: str = "iv",
                           close_time: str = "15:15") -> pd.DataFrame:
    """
    Build a daily-level DataFrame with IV filter and expected move filter results.

    Args:
        df_1min: Full 1-minute DataFrame.
        iv_window: Rolling window for IV percentile.
        iv_threshold: Maximum IV percentile to trade.
        em_threshold: Minimum expected move % to trade.
        col_date, col_time, col_iv: Column names.
        close_time: Time to sample daily closing IV.

    Returns:
        DataFrame indexed by date with columns:
        daily_iv, iv_percentile, expected_move_pct, iv_pass, em_pass, trade_allowed
    """
    daily_iv = extract_daily_closing_iv(df_1min, col_date, col_time, col_iv, close_time)
    iv_pctile = compute_iv_percentile(daily_iv, window=iv_window)
    exp_move = compute_expected_move(daily_iv)

    daily = pd.DataFrame({
        "daily_iv": daily_iv,
        "iv_percentile": iv_pctile,
        "expected_move_pct": exp_move,
    })

    daily["iv_pass"] = daily["iv_percentile"] < iv_threshold
    daily["em_pass"] = daily["expected_move_pct"] > em_threshold
    daily["trade_allowed"] = daily["iv_pass"] & daily["em_pass"]

    return daily
