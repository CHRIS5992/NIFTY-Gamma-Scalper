"""
visualization.py — Plotly chart builders for the Gamma Scalping Dashboard.

Each function returns a Plotly Figure object for rendering in Streamlit.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Color Palette ─────────────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#00D2FF",
    "accent": "#FF6B6B",
    "success": "#00E676",
    "warning": "#FFD600",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1D23",
    "text": "#E0E0E0",
    "grid": "#2A2D35",
    "up": "#00E676",
    "down": "#FF5252",
    "bb_fill": "rgba(108, 99, 255, 0.1)",
    "equity": "#00D2FF",
    "drawdown": "#FF5252",
}

LAYOUT_TEMPLATE = dict(
    paper_bgcolor=COLORS["bg_dark"],
    plot_bgcolor=COLORS["bg_dark"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    margin=dict(l=60, r=30, t=50, b=40),
    hovermode="x unified",
)


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply consistent dark theme layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        **LAYOUT_TEMPLATE,
        **kwargs,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# EQUITY & DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════

def plot_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
    """Line chart of cumulative PnL over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["datetime"],
        y=equity_df["cumulative_pnl"],
        mode="lines",
        line=dict(color=COLORS["equity"], width=2),
        name="Cumulative PnL",
        fill="tozeroy",
        fillcolor="rgba(0, 210, 255, 0.08)",
    ))
    return _apply_layout(fig, "📈 Equity Curve", yaxis_title="Cumulative PnL (₹)")


def plot_drawdown(equity_df: pd.DataFrame) -> go.Figure:
    """Drawdown chart (filled area)."""
    eq = equity_df["cumulative_pnl"]
    running_max = eq.cummax()
    drawdown = eq - running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["datetime"],
        y=drawdown,
        mode="lines",
        line=dict(color=COLORS["drawdown"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 82, 82, 0.15)",
        name="Drawdown",
    ))
    return _apply_layout(fig, "📉 Drawdown", yaxis_title="Drawdown (₹)")


def plot_cumulative_returns(trade_log: pd.DataFrame) -> go.Figure:
    """Cumulative returns percentage from trade log."""
    if trade_log.empty:
        return go.Figure()

    cumulative = trade_log["pnl_pct"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative) + 1)),
        y=cumulative,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=4, color=COLORS["primary"]),
        name="Cumulative Return %",
    ))
    return _apply_layout(fig, "📊 Cumulative Returns (%)",
                         xaxis_title="Trade #", yaxis_title="Cumulative Return %")


# ═══════════════════════════════════════════════════════════════════════
# MONTHLY HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(trade_log: pd.DataFrame) -> go.Figure:
    """Monthly returns heatmap (Year × Month)."""
    if trade_log.empty:
        return go.Figure()

    tl = trade_log.copy()
    tl["entry_time"] = pd.to_datetime(tl["entry_time"])
    tl["year"] = tl["entry_time"].dt.year
    tl["month"] = tl["entry_time"].dt.month

    monthly = tl.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Ensure all 12 months present
    for m in range(1, 13):
        if m not in monthly.columns:
            monthly[m] = 0
    monthly = monthly[sorted(monthly.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=[month_names[m - 1] for m in monthly.columns],
        y=[str(y) for y in monthly.index],
        colorscale=[[0, COLORS["down"]], [0.5, COLORS["bg_dark"]], [1, COLORS["up"]]],
        zmid=0,
        text=np.round(monthly.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hoverongaps=False,
    ))
    return _apply_layout(fig, "🗓️ Monthly PnL Heatmap", height=300)


# ═══════════════════════════════════════════════════════════════════════
# TRADE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def plot_trade_distribution(trade_log: pd.DataFrame) -> go.Figure:
    """Histogram of trade PnL %."""
    if trade_log.empty:
        return go.Figure()

    colors = [COLORS["up"] if x > 0 else COLORS["down"] for x in trade_log["pnl_pct"]]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trade_log["pnl_pct"],
        nbinsx=30,
        marker_color=COLORS["primary"],
        opacity=0.8,
        name="PnL %",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["warning"], line_width=1)
    return _apply_layout(fig, "📊 Trade PnL Distribution",
                         xaxis_title="PnL %", yaxis_title="Count")


def plot_holding_time_histogram(trade_log: pd.DataFrame) -> go.Figure:
    """Histogram of trade holding times in minutes."""
    if trade_log.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trade_log["holding_minutes"],
        nbinsx=25,
        marker_color=COLORS["secondary"],
        opacity=0.8,
        name="Holding Time",
    ))
    return _apply_layout(fig, "⏱️ Holding Time Distribution",
                         xaxis_title="Minutes", yaxis_title="Count")


def plot_win_loss_breakdown(trade_log: pd.DataFrame) -> go.Figure:
    """Win/Loss/Breakeven breakdown as a bar + pie combo."""
    if trade_log.empty:
        return go.Figure()

    wins = len(trade_log[trade_log["pnl"] > 0])
    losses = len(trade_log[trade_log["pnl"] < 0])
    breakeven = len(trade_log[trade_log["pnl"] == 0])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Wins", "Losses", "Breakeven"],
        y=[wins, losses, breakeven],
        marker_color=[COLORS["up"], COLORS["down"], COLORS["warning"]],
        text=[wins, losses, breakeven],
        textposition="auto",
    ))
    return _apply_layout(fig, "🏆 Win/Loss Breakdown", yaxis_title="Count")


def plot_exit_reasons(trade_log: pd.DataFrame) -> go.Figure:
    """Pie chart of exit reasons."""
    if trade_log.empty or "exit_reason" not in trade_log.columns:
        return go.Figure()

    reasons = trade_log["exit_reason"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=reasons.index,
        values=reasons.values,
        hole=0.4,
        marker=dict(colors=[COLORS["success"], COLORS["accent"],
                            COLORS["warning"], COLORS["secondary"],
                            COLORS["primary"]]),
    )])
    return _apply_layout(fig, "🎯 Exit Reasons")


# ═══════════════════════════════════════════════════════════════════════
# DAY REPLAY
# ═══════════════════════════════════════════════════════════════════════

def plot_day_replay(day_1min: pd.DataFrame,
                    day_trades: pd.DataFrame,
                    col_spot: str = "spot",
                    col_straddle: str = "Straddle_Price",
                    col_datetime: str = "datetime") -> go.Figure:
    """
    Interactive day replay chart with:
    - Spot price + Bollinger Bands overlay
    - Entry & exit markers
    - Straddle price subplot
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Spot Price + Bollinger Bands", "Straddle Price"),
    )

    dt = day_1min[col_datetime]

    # ── Spot price ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dt, y=day_1min[col_spot],
        mode="lines", line=dict(color=COLORS["text"], width=1.5),
        name="Spot",
    ), row=1, col=1)

    # ── Bollinger Bands ────────────────────────────────────────────
    if "bb_upper_1m" in day_1min.columns:
        fig.add_trace(go.Scatter(
            x=dt, y=day_1min["bb_upper_1m"],
            mode="lines", line=dict(color=COLORS["primary"], width=1, dash="dot"),
            name="Upper BB", showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dt, y=day_1min["bb_lower_1m"],
            mode="lines", line=dict(color=COLORS["primary"], width=1, dash="dot"),
            name="Lower BB", fill="tonexty", fillcolor=COLORS["bb_fill"],
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dt, y=day_1min["bb_mid_1m"],
            mode="lines", line=dict(color=COLORS["warning"], width=1, dash="dash"),
            name="Middle BB",
        ), row=1, col=1)

    # ── Trade markers ──────────────────────────────────────────────
    if not day_trades.empty:
        # Entry markers
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(day_trades["entry_time"]),
            y=[day_1min.loc[
                day_1min[col_datetime] == pd.to_datetime(t), col_spot
            ].values[0] if len(day_1min.loc[
                day_1min[col_datetime] == pd.to_datetime(t)
            ]) > 0 else np.nan for t in day_trades["entry_time"]],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=COLORS["up"],
                        line=dict(width=2, color="white")),
            name="Entry",
        ), row=1, col=1)

        # Exit markers
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(day_trades["exit_time"]),
            y=[day_1min.loc[
                day_1min[col_datetime] == pd.to_datetime(t), col_spot
            ].values[0] if len(day_1min.loc[
                day_1min[col_datetime] == pd.to_datetime(t)
            ]) > 0 else np.nan for t in day_trades["exit_time"]],
            mode="markers",
            marker=dict(symbol="triangle-down", size=14, color=COLORS["down"],
                        line=dict(width=2, color="white")),
            name="Exit",
        ), row=1, col=1)

    # ── Straddle price subplot ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dt, y=day_1min[col_straddle],
        mode="lines", line=dict(color=COLORS["secondary"], width=1.5),
        name="Straddle Price",
    ), row=2, col=1)

    # ── Layout ─────────────────────────────────────────────────────
    fig.update_layout(
        height=650,
        **LAYOUT_TEMPLATE,
        title=dict(text="🔍 Day Replay", font=dict(size=16)),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])

    return fig


# ═══════════════════════════════════════════════════════════════════════
# IV ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

def plot_iv_analytics(daily_filters: pd.DataFrame) -> go.Figure:
    """Rolling IV and IV percentile dual-axis chart."""
    if daily_filters.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=("Daily Closing IV", "IV Percentile"),
    )

    dates = daily_filters.index

    fig.add_trace(go.Scatter(
        x=dates, y=daily_filters["daily_iv"],
        mode="lines", line=dict(color=COLORS["primary"], width=1.5),
        name="IV",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=daily_filters["iv_percentile"],
        mode="lines", line=dict(color=COLORS["secondary"], width=1.5),
        name="IV Percentile",
        fill="tozeroy", fillcolor="rgba(0, 210, 255, 0.1)",
    ), row=2, col=1)

    # Add threshold line
    fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["accent"],
                  row=2, col=1, annotation_text="Threshold")

    fig.update_layout(
        height=500,
        **LAYOUT_TEMPLATE,
        title=dict(text="📊 IV Analytics", font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])

    return fig


def plot_expected_move(daily_filters: pd.DataFrame) -> go.Figure:
    """Expected move chart with threshold."""
    if daily_filters.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_filters.index,
        y=daily_filters["expected_move_pct"],
        marker_color=[
            COLORS["up"] if v > 0.4 else COLORS["down"]
            for v in daily_filters["expected_move_pct"]
        ],
        opacity=0.7,
        name="Expected Move %",
    ))
    fig.add_hline(y=0.4, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Threshold")
    return _apply_layout(fig, "📐 Expected Move (IV/16)",
                         yaxis_title="Expected Move %", height=350)


def plot_iv_regime(daily_filters: pd.DataFrame) -> go.Figure:
    """IV regime classification bar chart."""
    if daily_filters.empty:
        return go.Figure()

    df = daily_filters.copy()
    df["regime"] = pd.cut(
        df["iv_percentile"],
        bins=[0, 0.2, 0.35, 0.5, 0.8, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        include_lowest=True,
    )
    regime_counts = df["regime"].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=regime_counts.index.astype(str),
        y=regime_counts.values,
        marker_color=[COLORS["success"], COLORS["up"], COLORS["warning"],
                      COLORS["accent"], COLORS["down"]],
        text=regime_counts.values,
        textposition="auto",
    ))
    return _apply_layout(fig, "🔬 IV Regime Breakdown", yaxis_title="Days")


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════

def plot_optimization_heatmap(opt_df: pd.DataFrame,
                               param1_name: str,
                               param2_name: str,
                               metric: str = "sharpe") -> go.Figure:
    """
    Heatmap of a metric across two parameter axes.
    """
    if opt_df.empty:
        return go.Figure()

    pivot = opt_df.pivot_table(
        index=param1_name, columns=param2_name, values=metric, aggfunc="first"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="Viridis",
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    return _apply_layout(
        fig,
        f"🧪 Optimization: {metric.title()} by {param1_name} × {param2_name}",
        xaxis_title=param2_name,
        yaxis_title=param1_name,
        height=400,
    )
