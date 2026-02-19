"""
app.py — Streamlit Dashboard for IV-Filtered ATM Straddle Gamma Scalping Strategy.

Professional-grade UI with 5 tabs, sidebar controls, parameter optimization,
and interactive trade replay.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from config import StrategyConfig, OptimizationGrid
from indicators import prepare_daily_filters, compute_bollinger, detect_compression
from backtest_engine import (
    load_and_prepare,
    precompute_1min_indicators,
    precompute_5min_compression,
    run_backtest,
    run_optimization,
)
from visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_cumulative_returns,
    plot_monthly_heatmap,
    plot_trade_distribution,
    plot_holding_time_histogram,
    plot_win_loss_breakdown,
    plot_exit_reasons,
    plot_day_replay,
    plot_iv_analytics,
    plot_expected_move,
    plot_iv_regime,
    plot_optimization_heatmap,
)


# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NIFTY Gamma Scalper",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1D23 0%, #252830 100%);
        border: 1px solid #2A2D35;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .positive { color: #00E676; }
    .negative { color: #FF5252; }
    .neutral { color: #00D2FF; }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 700;
        margin-bottom: 0;
    }
    .dashboard-sub {
        color: #888;
        font-size: 0.95em;
        margin-top: -5px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Strategy info box */
    .info-box {
        background: #1A1D23;
        border-left: 4px solid #6C63FF;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }

    /* Trade table */
    .dataframe { font-size: 13px !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1D23 100%);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Strategy Parameters")
    st.markdown("---")

    st.markdown("### 📊 IV Filter")
    iv_pctile_thresh = st.slider(
        "IV Percentile Threshold", 0.10, 0.60, 0.35, 0.05,
        help="Trade only when IV percentile < this value (options are cheap)"
    )
    expected_move_thresh = st.slider(
        "Expected Move % Threshold", 0.1, 1.0, 0.4, 0.05,
        help="Min IV/16 to ensure sufficient volatility"
    )

    st.markdown("### 📐 Bollinger Bands")
    bb_length = st.slider("BB Length", 10, 50, 20, 5)
    bb_std = st.slider("BB Std Dev", 1.0, 3.5, 2.0, 0.25)

    st.markdown("### 🎯 Exit Rules")
    take_profit = st.slider("Take Profit %", 1.0, 10.0, 3.0, 0.5)
    stop_loss = st.slider("Stop Loss %", 0.5, 5.0, 1.5, 0.25)
    max_trades = st.slider("Max Trades/Day", 1, 5, 3, 1)

    st.markdown("### 💰 Transaction Costs")
    slippage_on = st.toggle("Enable Slippage", value=False)
    slippage_bps = st.number_input(
        "Slippage (bps)", 0.0, 50.0, 5.0, 1.0,
        disabled=not slippage_on
    ) if slippage_on else 0.0
    brokerage = st.number_input("Brokerage per Trade (₹)", 0.0, 500.0, 0.0, 10.0)

    st.markdown("---")

    # Build config from sidebar
    config = StrategyConfig(
        iv_percentile_threshold=iv_pctile_thresh,
        expected_move_threshold=expected_move_thresh,
        bb_length_1min=bb_length,
        bb_std_1min=bb_std,
        bb_length_5min=bb_length,
        bb_std_5min=bb_std,
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        max_trades_per_day=max_trades,
        slippage_bps=slippage_bps if slippage_on else 0.0,
        brokerage_per_trade=brokerage,
    )


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<p class="dashboard-header">⚡ NIFTY Gamma Scalper</p>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-sub">IV-Filtered Intraday ATM Straddle Gamma Scalping Engine</p>',
            unsafe_allow_html=True)
st.markdown("")


@st.cache_data(show_spinner="Loading data...")
def load_data_from_path(path_1min: str, path_5min: str):
    """Load and prepare both datasets from file paths."""
    cfg = StrategyConfig()
    df1 = load_and_prepare(path_1min, cfg)
    df5 = load_and_prepare(path_5min, cfg)
    return df1, df5


@st.cache_data(show_spinner="Loading uploaded data...")
def load_data_from_upload(file_1min_bytes: bytes, file_5min_bytes: bytes):
    """Load and prepare both datasets from uploaded file bytes."""
    import io
    cfg = StrategyConfig()
    df1 = pd.read_csv(io.BytesIO(file_1min_bytes))
    df5 = pd.read_csv(io.BytesIO(file_5min_bytes))

    # Apply same preparation as load_and_prepare
    for df in [df1, df5]:
        if cfg.col_datetime in df.columns:
            df[cfg.col_datetime] = pd.to_datetime(df[cfg.col_datetime])
        elif cfg.col_date in df.columns and cfg.col_time in df.columns:
            df[cfg.col_datetime] = pd.to_datetime(
                df[cfg.col_date].astype(str) + " " + df[cfg.col_time].astype(str)
            )
        if cfg.col_date not in df.columns:
            df[cfg.col_date] = df[cfg.col_datetime].dt.date.astype(str)
        else:
            df[cfg.col_date] = df[cfg.col_date].astype(str)
        if cfg.col_time not in df.columns:
            df[cfg.col_time] = df[cfg.col_datetime].dt.strftime("%H:%M")
        else:
            df[cfg.col_time] = pd.to_datetime(
                df[cfg.col_time], format="mixed"
            ).dt.strftime("%H:%M:%S")
        df.sort_values(cfg.col_datetime, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df1, df5


# Auto-detect local files (multiple search paths for flexibility)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_PATHS = [
    APP_DIR,
    os.path.join(APP_DIR, "data"),
]

def _find_local_file(filename: str):
    """Search for a CSV file in known directories."""
    for d in SEARCH_PATHS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

LOCAL_1MIN = _find_local_file("FINAL_NIFTY_MASTER_ATM.csv")
LOCAL_5MIN = _find_local_file("FINAL_NIFTY_MASTER_ATM_5min.csv")
HAS_LOCAL = LOCAL_1MIN is not None and LOCAL_5MIN is not None

with st.sidebar:
    st.markdown("### 📁 Data Files")

    if HAS_LOCAL:
        use_upload = st.toggle("Upload CSV files", value=False,
                               help="Toggle to upload custom files instead of using local files")
    else:
        st.info("📤 Upload your CSV files to get started")
        use_upload = True  # Force upload mode on cloud

    if use_upload:
        file_1min = st.file_uploader("1-Minute Data", type=["csv"], key="f1")
        file_5min = st.file_uploader("5-Minute Data", type=["csv"], key="f5")

data_loaded = False

if use_upload:
    if file_1min is not None and file_5min is not None:
        df_1min, df_5min = load_data_from_upload(
            file_1min.getvalue(), file_5min.getvalue()
        )
        data_loaded = True
    else:
        st.warning("⚠️ Please upload both CSV files (1-min and 5-min data).")
        st.markdown("""
        **Required files:**
        - `FINAL_NIFTY_MASTER_ATM.csv` — 1-minute NIFTY data
        - `FINAL_NIFTY_MASTER_ATM_5min.csv` — 5-minute NIFTY data

        **Expected columns:** `datetime, spot, iv, CE_open/high/low/close, PE_open/high/low/close, Straddle_Price`
        """)
        st.stop()
else:
    df_1min, df_5min = load_data_from_path(LOCAL_1MIN, LOCAL_5MIN)
    data_loaded = True


# ═══════════════════════════════════════════════════════════════════════
# RUN BACKTEST
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(_df1, _df5, config_dict):
    """Cache backtest results keyed by config parameters."""
    cfg = StrategyConfig(**config_dict)
    trade_log, equity_df, metrics = run_backtest(_df1.copy(), _df5.copy(), cfg)
    return trade_log, equity_df, metrics


@st.cache_data(show_spinner="Computing daily filters...")
def cached_daily_filters(_df1, config_dict):
    cfg = StrategyConfig(**config_dict)
    return prepare_daily_filters(
        _df1.copy(),
        iv_window=cfg.iv_percentile_window,
        iv_threshold=cfg.iv_percentile_threshold,
        em_threshold=cfg.expected_move_threshold,
        col_date=cfg.col_date,
        col_time=cfg.col_time,
        col_iv=cfg.col_iv,
        close_time=cfg.iv_close_time,
    )


# Convert config to dict for caching
config_dict = {
    k: v for k, v in config.__dict__.items()
}

# Run backtest
trade_log, equity_df, metrics = cached_backtest(df_1min, df_5min, config_dict)
daily_filters = cached_daily_filters(df_1min, config_dict)


# ═══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Strategy Overview",
    "📈 Backtest Results",
    "🔎 Trade Analysis",
    "🔍 Day Replay",
    "📊 IV Analytics",
])


# ───────────────────────────────────────────────────────────────────────
# TAB 1 — STRATEGY OVERVIEW
# ───────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Strategy Overview")
    st.markdown("""
    <div class="info-box">
    <strong>IV-Filtered Intraday ATM Straddle Gamma Scalping</strong><br>
    This strategy buys ATM straddles intraday on NIFTY when options are cheap (low IV percentile)
    and a volatility breakout is imminent (Bollinger Band compression + breakout).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 Strategy Logic")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **Part 1 — IV Filter (Daily)**
        - Extract IV at 3:15 PM each day
        - Compute 20-day rolling IV percentile
        - Trade only when IV percentile < threshold
        - *Ensures options are relatively cheap*

        **Part 2 — Expected Move Filter**
        - Expected Move % = IV / 16
        - Trade only when Expected Move > threshold
        - *Ensures sufficient volatility potential*

        **Part 3 — Compression Filter (5-min)**
        - Bollinger Bands (20,2) on 5-min spot
        - Bandwidth = (Upper - Lower) / Middle
        - Compression = bandwidth at 20-candle low
        - *Detects volatility squeeze*
        """)

    with col_b:
        st.markdown("""
        **Part 4 — Entry Logic (1-min)**
        - Bollinger Bands (20,2) on 1-min spot
        - Spot breaks above upper BB or below lower BB
        - Candle body > avg body of last 10 candles
        - *Confirms breakout with momentum*

        **Part 5 — Exit Rules**
        - Take profit at +3% PnL
        - Stop loss at -1.5% PnL
        - Range-bound exit after 30 min flat PnL
        - Hard exit at 3:15 PM

        **Part 6 — Re-Entry**
        - Wait 15 min after exit
        - Re-check all conditions
        - Max 3 trades per day
        """)

    st.markdown("### ⚙️ Current Parameters")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("IV Percentile Threshold", f"{config.iv_percentile_threshold:.2f}")
        st.metric("Expected Move Threshold", f"{config.expected_move_threshold:.1f}%")
    with p2:
        st.metric("BB Length", config.bb_length_1min)
        st.metric("BB Std Dev", f"{config.bb_std_1min:.1f}")
    with p3:
        st.metric("Take Profit", f"{config.take_profit_pct:.1f}%")
        st.metric("Stop Loss", f"{config.stop_loss_pct:.1f}%")
    with p4:
        st.metric("Max Trades/Day", config.max_trades_per_day)
        st.metric("Brokerage", f"₹{config.brokerage_per_trade:.0f}")

    st.markdown("### 📊 Data Summary")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**1-Minute Dataset**")
        st.write(f"- Rows: **{len(df_1min):,}**")
        st.write(f"- Date Range: **{df_1min[config.col_date].min()}** to **{df_1min[config.col_date].max()}**")
        st.write(f"- Trading Days: **{df_1min[config.col_date].nunique()}**")
    with d2:
        st.markdown("**5-Minute Dataset**")
        st.write(f"- Rows: **{len(df_5min):,}**")
        st.write(f"- Date Range: **{df_5min[config.col_date].min()}** to **{df_5min[config.col_date].max()}**")
        st.write(f"- Trading Days: **{df_5min[config.col_date].nunique()}**")


# ───────────────────────────────────────────────────────────────────────
# TAB 2 — BACKTEST RESULTS
# ───────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Backtest Results")

    if trade_log.empty:
        st.warning("No trades were generated with the current parameters. "
                    "Try adjusting IV percentile or expected move thresholds.")
    else:
        # ── Metric Cards ──────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        total_ret = metrics.get("total_return", 0)
        ret_class = "positive" if total_ret > 0 else "negative"

        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {ret_class}">₹{total_ret:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            wr = metrics.get("win_rate", 0)
            wr_class = "positive" if wr >= 50 else "negative"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {wr_class}">{wr:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            pf = metrics.get("profit_factor", 0)
            pf_class = "positive" if pf >= 1 else "negative"
            pf_str = f"{pf:.2f}" if pf < 100 else "∞"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {pf_class}">{pf_str}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            sr = metrics.get("sharpe_ratio", 0)
            sr_class = "positive" if sr > 0 else "negative"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {sr_class}">{sr:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            mdd = metrics.get("max_drawdown", 0)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">₹{mdd:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with m6:
            tt = metrics.get("total_trades", 0)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value neutral">{tt}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Charts ────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_equity_curve(equity_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_drawdown(equity_df), use_container_width=True)

        st.plotly_chart(plot_cumulative_returns(trade_log), use_container_width=True)
        st.plotly_chart(plot_monthly_heatmap(trade_log), use_container_width=True)

        # ── Detailed Metrics Table ────────────────────────────────────
        st.markdown("### 📋 Performance Metrics")
        metrics_display = {
            "Total Return (₹)": f"{metrics.get('total_return', 0):,.2f}",
            "Total Return (%)": f"{metrics.get('total_return_pct', 0):,.2f}%",
            "CAGR": f"{metrics.get('cagr', 0):.2f}%",
            "Win Rate": f"{metrics.get('win_rate', 0):.1f}%",
            "Winning Trades": metrics.get("winning_trades", 0),
            "Losing Trades": metrics.get("losing_trades", 0),
            "Avg Win (₹)": f"{metrics.get('avg_win', 0):,.2f}",
            "Avg Loss (₹)": f"{metrics.get('avg_loss', 0):,.2f}",
            "Avg Win (%)": f"{metrics.get('avg_win_pct', 0):.2f}%",
            "Avg Loss (%)": f"{metrics.get('avg_loss_pct', 0):.2f}%",
            "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
            "Max Drawdown (₹)": f"{metrics.get('max_drawdown', 0):,.2f}",
            "Avg Holding (min)": f"{metrics.get('avg_holding_minutes', 0):.1f}",
            "Max Holding (min)": f"{metrics.get('max_holding_minutes', 0):.0f}",
        }
        df_metrics = pd.DataFrame(
            list(metrics_display.items()),
            columns=["Metric", "Value"]
        )
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # ── Optimization Section ──────────────────────────────────────
        with st.expander("🧪 Parameter Optimization", expanded=False):
            st.markdown("Select two parameters to optimize and their ranges:")

            opt_params = [
                "iv_percentile_threshold", "expected_move_threshold",
                "take_profit_pct", "stop_loss_pct",
                "bb_length_1min", "bb_std_1min",
            ]

            oc1, oc2 = st.columns(2)
            with oc1:
                p1_name = st.selectbox("Parameter 1", opt_params, index=0)
                p1_vals_str = st.text_input(
                    "Values (comma-separated)",
                    "0.20, 0.25, 0.30, 0.35, 0.40",
                    key="p1v"
                )
            with oc2:
                p2_name = st.selectbox("Parameter 2", opt_params, index=3)
                p2_vals_str = st.text_input(
                    "Values (comma-separated)",
                    "1.0, 1.5, 2.0, 2.5",
                    key="p2v"
                )

            if st.button("🚀 Run Optimization", type="primary"):
                try:
                    p1_vals = [float(x.strip()) for x in p1_vals_str.split(",")]
                    p2_vals = [float(x.strip()) for x in p2_vals_str.split(",")]

                    # Convert to int if needed
                    if "length" in p1_name:
                        p1_vals = [int(x) for x in p1_vals]
                    if "length" in p2_name:
                        p2_vals = [int(x) for x in p2_vals]

                    total_combos = len(p1_vals) * len(p2_vals)
                    progress = st.progress(0, f"Running {total_combos} combinations...")

                    def update_progress(current, total):
                        progress.progress(current / total,
                                          f"Combo {current}/{total}")

                    opt_results = run_optimization(
                        df_1min.copy(), df_5min.copy(), config,
                        p1_name, p1_vals,
                        p2_name, p2_vals,
                        progress_callback=update_progress,
                    )
                    progress.empty()

                    st.plotly_chart(
                        plot_optimization_heatmap(opt_results, p1_name, p2_name, "sharpe"),
                        use_container_width=True
                    )
                    st.plotly_chart(
                        plot_optimization_heatmap(opt_results, p1_name, p2_name, "total_return"),
                        use_container_width=True
                    )

                    st.markdown("#### Full Results")
                    st.dataframe(opt_results.round(3), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Optimization error: {e}")


# ───────────────────────────────────────────────────────────────────────
# TAB 3 — TRADE ANALYSIS
# ───────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Trade Analysis")

    if trade_log.empty:
        st.warning("No trades to analyze.")
    else:
        # ── Trade List ────────────────────────────────────────────────
        st.markdown("### 📋 Trade Log")

        display_cols = [
            "trade_num", "date", "entry_time", "exit_time",
            "entry_price", "exit_price", "pnl", "pnl_pct",
            "holding_minutes", "exit_reason"
        ]
        available = [c for c in display_cols if c in trade_log.columns]
        tl_display = trade_log[available].copy()

        # Format for display
        for col in ["entry_price", "exit_price", "pnl"]:
            if col in tl_display.columns:
                tl_display[col] = tl_display[col].round(2)
        if "pnl_pct" in tl_display.columns:
            tl_display["pnl_pct"] = tl_display["pnl_pct"].round(2)

        st.dataframe(
            tl_display.style.applymap(
                lambda v: "color: #00E676" if isinstance(v, (int, float)) and v > 0
                else ("color: #FF5252" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["pnl", "pnl_pct"] if "pnl" in tl_display.columns else []
            ),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # ── CSV Export ────────────────────────────────────────────────
        csv_data = trade_log.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log CSV",
            data=csv_data,
            file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_trade_distribution(trade_log), use_container_width=True)
        with c2:
            st.plotly_chart(plot_holding_time_histogram(trade_log), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_win_loss_breakdown(trade_log), use_container_width=True)
        with c4:
            st.plotly_chart(plot_exit_reasons(trade_log), use_container_width=True)

        # ── Trade-by-Trade Replay Slider ──────────────────────────────
        st.markdown("### 🎬 Trade-by-Trade Replay")
        trade_idx = st.slider(
            "Select Trade",
            1, len(trade_log), 1,
            help="Slide through individual trades"
        )
        selected_trade = trade_log.iloc[trade_idx - 1]

        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Entry Time", str(selected_trade.get("entry_time", ""))[:19])
        with tc2:
            st.metric("Exit Time", str(selected_trade.get("exit_time", ""))[:19])
        with tc3:
            pnl_val = selected_trade.get("pnl", 0)
            st.metric("PnL", f"₹{pnl_val:,.2f}",
                       delta=f"{selected_trade.get('pnl_pct', 0):.2f}%")
        with tc4:
            st.metric("Exit Reason", selected_trade.get("exit_reason", ""))


# ───────────────────────────────────────────────────────────────────────
# TAB 4 — DAY REPLAY
# ───────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Day Replay")

    trading_days = sorted(df_1min[config.col_date].unique())

    if len(trading_days) == 0:
        st.warning("No trading days in data.")
    else:
        selected_day = st.selectbox(
            "Select Trading Day",
            trading_days,
            index=len(trading_days) - 1,
        )

        # Get day data
        day_mask = df_1min[config.col_date] == selected_day
        day_data = df_1min.loc[day_mask].copy()

        # Pre-compute BB indicators for display
        day_data = precompute_1min_indicators(day_data, config)

        # Get trades for this day
        if not trade_log.empty and "date" in trade_log.columns:
            day_trades = trade_log[trade_log["date"] == selected_day]
        else:
            day_trades = pd.DataFrame()

        # Day info
        di1, di2, di3, di4 = st.columns(4)
        with di1:
            st.metric("Spot Open", f"₹{day_data[config.col_spot].iloc[0]:,.2f}"
                       if len(day_data) > 0 else "N/A")
        with di2:
            st.metric("Spot Close", f"₹{day_data[config.col_spot].iloc[-1]:,.2f}"
                       if len(day_data) > 0 else "N/A")
        with di3:
            spot_range = day_data[config.col_spot].max() - day_data[config.col_spot].min()
            st.metric("Day Range", f"₹{spot_range:,.2f}")
        with di4:
            st.metric("Trades on Day", len(day_trades))

        # Plot
        fig = plot_day_replay(
            day_data, day_trades,
            col_spot=config.col_spot,
            col_straddle=config.col_straddle,
            col_datetime=config.col_datetime,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade details for this day
        if not day_trades.empty:
            st.markdown("### 📋 Day's Trades")
            st.dataframe(day_trades, use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 5 — IV ANALYTICS
# ───────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## IV Analytics")

    if daily_filters.empty:
        st.warning("No IV data available.")
    else:
        # ── Summary Stats ─────────────────────────────────────────────
        iv1, iv2, iv3, iv4 = st.columns(4)
        with iv1:
            st.metric("Avg IV",
                       f"{daily_filters['daily_iv'].mean():.2f}")
        with iv2:
            st.metric("Current IV Percentile",
                       f"{daily_filters['iv_percentile'].iloc[-1]:.2f}"
                       if len(daily_filters) > 0 else "N/A")
        with iv3:
            days_allowed = daily_filters["trade_allowed"].sum()
            st.metric("Days Allowed to Trade",
                       f"{days_allowed} / {len(daily_filters)}")
        with iv4:
            pct_allowed = days_allowed / len(daily_filters) * 100 if len(daily_filters) > 0 else 0
            st.metric("% Days Allowed", f"{pct_allowed:.1f}%")

        # ── Charts ────────────────────────────────────────────────────
        st.plotly_chart(plot_iv_analytics(daily_filters), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_expected_move(daily_filters), use_container_width=True)
        with c2:
            st.plotly_chart(plot_iv_regime(daily_filters), use_container_width=True)

        # ── Daily Filter Table ────────────────────────────────────────
        with st.expander("📊 Daily Filter Data", expanded=False):
            df_display = daily_filters.copy()
            df_display = df_display.round(4)
            st.dataframe(df_display, use_container_width=True, height=400)
