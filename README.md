# ⚡ NIFTY Gamma Scalper

**IV-Filtered Intraday ATM Straddle Gamma Scalping Dashboard**

A professional-grade Streamlit dashboard that backtests an ATM straddle gamma scalping strategy on NIFTY 1-minute and 5-minute options data.

## 🚀 Deploy on Streamlit Cloud

1. **Push this repo to GitHub** (exclude CSV data files — they're in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Deploy!

> **Note:** CSV data files are NOT included in the repo. Upload them via the sidebar after deploying.

## 📁 Project Structure

```
├── app.py                  # Streamlit dashboard (main entry point)
├── config.py               # Strategy parameters
├── indicators.py           # Technical indicators
├── strategy.py             # Entry/exit/re-entry logic
├── backtest_engine.py      # Event-driven backtester
├── visualization.py        # Plotly chart builders
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Theme & server config
└── .gitignore
```

## 📊 Required Data

Upload these CSV files via the sidebar:

| File | Description |
|------|-------------|
| `FINAL_NIFTY_MASTER_ATM.csv` | 1-minute NIFTY data |
| `FINAL_NIFTY_MASTER_ATM_5min.csv` | 5-minute NIFTY data |

**Expected columns:** `datetime, spot, iv, CE_open, CE_high, CE_low, CE_close, PE_open, PE_high, PE_low, PE_close, Straddle_Price`

## 🏃 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Strategy

Six-part IV-filtered intraday ATM straddle gamma scalping:

1. **IV Filter** — 20-day rolling IV percentile < 0.35
2. **Expected Move** — IV/16 > 0.4%
3. **Compression** — 5-min Bollinger bandwidth at 20-candle low
4. **Entry** — 1-min BB breakout + large candle body
5. **Exit** — TP 3% / SL 1.5% / 30-min flat / 3:15 PM hard exit
6. **Re-entry** — 15 min wait, max 3 trades/day
