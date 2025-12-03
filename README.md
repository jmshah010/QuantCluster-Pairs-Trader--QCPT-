````markdown
# Nifty 50 Pairs Trading Terminal (EOD)

Professional **Streamlit-based pairs trading terminals** for the **Nifty 50** universe built on end-of-day (EOD) data.

This repository contains **two versions** of the same idea:

1. **Core Version – Pairs Trading Terminal (this script)**  
2. **Advanced Version – Institutional-Style Pairs Trading Terminal** (with Kalman filter, half-life, VaR, regime detection, etc.)

Both versions are designed for **research / educational** use around **statistical arbitrage and pairs trading** in Indian equities.

> ⚠️ **Disclaimer**: This project is for **educational and research purposes only**.  
> It does **not** constitute investment advice and must not be used directly for live trading without your own testing, validation, and risk controls.

---

## 🔁 1. Core Version – Pairs Trading System (this file)

This is the **base engine** you shared (simpler, easier to understand and explain).  
It focuses on the **core quant logic** of pairs trading using EOD data.

### Key Capabilities

- **Universe**: All Nifty 50 tickers + Nifty index (`^NSEI`) via `yfinance`
- **Return & Feature Computation**
  - Daily returns
  - Annualized **Volatility**
  - Annualized **Mean Return**
  - **Beta** vs `^NSEI`
  - **Sharpe Ratio**
  - **Volatility Stability** (stability of rolling volatility)

- **Unsupervised Clustering (K-Means)**
  - Features are standardized via `StandardScaler`
  - K-Means is run for `k = 2 … max_k`
  - **Optimal number of clusters** chosen via **silhouette score**
  - Each stock gets a **cluster label**

- **Cointegrated Pair Discovery (Within Cluster)**
  - For a chosen cluster, all stock combinations are tested with `statsmodels.coint`
  - Pairs with **p-value < 0.05** are considered **cointegrated**
  - For each candidate pair, price **correlation** is also reported
  - Top pairs are displayed in a sorted table by p-value

- **Spread & Signal Logic**
  - Spread is defined using a **price ratio**:
    - `ratio = Stock1 / Stock2`
    - `spread = ratio – mean(ratio)`
    - `z-score = (spread – mean(spread)) / std(spread)`
  - Trading rules:
    - `z > +entry_z` → **Short spread** (Short Stock1, Long Stock2)  
    - `z < –entry_z` → **Long spread** (Long Stock1, Short Stock2)  
    - `|z| < exit_z` → **Exit / Flat**

- **Backtesting**
  - Backtest uses the z-score-based entry/exit rules
  - Metrics:
    - Total Return (%)
    - Sharpe Ratio
    - Max Drawdown (%)
    - Number of Trades
  - Charts:
    - Cumulative strategy return
    - Z-score with entry/exit levels

- **Live EOD Signals**
  - Uses the **latest available EOD close** for each pair
  - Generates text-based signals:
    - 🔴 SHORT Spread
    - 🟢 LONG Spread
    - ⚪ EXIT
    - ⏸️ HOLD

### UI Layout (Core)

Tabs:

1. **📊 Cluster Analysis**  
   - Cluster metrics, summary stats, and scatter plots  
2. **🎯 Pair Selection**  
   - Cluster-wise cointegrated pairs with p-value & correlation  
3. **📈 Backtest**  
   - Performance metrics + plots for a chosen pair  
4. **📉 Live Signals**  
   - Current z-score and EOD trading signals for all pairs  

---

## 🧠 2. Advanced Version – Institutional-Style Pairs Trading (Enhanced Script)

> This is the **enhanced version** you built next, with **unique differentiators**:
> - Kalman filter hedge ratio  
> - Half-life of mean reversion  
> - Regime detection  
> - VaR-based risk sizing  
> - Transaction cost modelling  
> - P&L curves and trade rationale text

Compared to the core version, the **logic is upgraded from “simple ratio spread” to an institutional-style mean-reversion framework**.

### Additional / Advanced Features

- **Dynamic Hedge Ratio – Kalman Filter**
  - Instead of a fixed price ratio, hedge ratio is estimated **dynamically over time**
  - Spread is defined as:
    - `spread_t = Price1_t – HedgeRatio_t × Price2_t`
  - Helps the strategy adapt to changing relative behavior between the two stocks

- **Mean-Reversion Diagnostics**
  - **Half-Life of Mean Reversion**
    - OU-style regression: Δspread vs lagged spread
    - Half-life ~ number of days for spread to revert halfway to its mean
  - **Regime Detection using ADF**
    - ADF test on recent spread
    - Classifies environment as:
      - “Mean-Reverting”  (good for pairs trading)
      - “Trending / Random Walk” (regime shift / higher risk)

- **Risk & Position Sizing**
  - **Parametric VaR (95%)** on daily strategy returns
  - User sets **Risk per Trade (% of capital)**
  - From VaR and risk budget, the app suggests a **recommended position size (₹)**

- **Transaction Costs**
  - User-configurable transaction cost in **bps per leg**
  - Costs applied when the position changes (entry/exit/reversal)
  - P&L and returns net of these costs

- **Performance & PnL Tracking**
  - Daily strategy returns
  - Cumulative returns (return index)
  - Currency PnL and **cumulative PnL (₹)**

- **Explanatory Block for Each Pair**
  - For the selected pair, the app explains:
    - Why the pair is selected (cointegration & correlation)
    - What the half-life and regime say about mean reversion
    - Clear entry/exit interpretation based on current thresholds

- **UI is similar**, but:
  - Backtest tab is richer with risk metrics (VaR, half-life, regime, recommended capital)
  - Extra visualizations: PnL curve, spread with trade markers
  - Live EOD signals still show **long/short/exit** recommendations using latest close

---

## 📦 Tech Stack

Common to both versions:

- **Language**: Python 3.8+
- **Data**: `yfinance` (Nifty 50 + ^NSEI EOD prices)
- **Core Libraries**:
  - `pandas`, `numpy`
  - `scikit-learn` (StandardScaler, KMeans, silhouette_score)
  - `statsmodels` (cointegration, ADF test)
  - `matplotlib`, `seaborn`
  - `streamlit`
  - `scipy`

---

## 🚀 How to Run

Assuming:

- `app_basic.py`  → core version (the script you pasted last)  
- `app_advanced.py` → institutional version (enhanced Kalman/VaR/half-life version)

You can adjust filenames as per your repo.

### 1. Install Dependencies

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn streamlit statsmodels scipy
````

### 2. Run Core Version

```bash
streamlit run app_basic.py
```

### 3. Run Advanced Version

```bash
streamlit run app_advanced.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 📂 Suggested Repository Structure

```text
.
├── Quant Cluster.py        #(Basic)   Core pairs trading terminal (ratio spread version)
├── QCPT.py                 #(Advance) Institutional version (Kalman, half-life, VaR, regime)
├── README.md           # This file
└── requirements.txt    # Optional: dependencies
```

---

## 🔮 Possible Extensions

* Portfolio-level **multi-pair risk aggregation** (portfolio VaR / CVaR)
* **Stress testing** scenarios (index –3%, sector shock, volatility spike)
* Integration with **intraday or order-book data** for execution modelling
* Export of **signal files** for live/demo trading systems

---

## ⚠️ Final Note

Both versions are meant to **teach and demonstrate** how professional-style pairs trading engines are structured:

* **Core Version** → Clean, simple logic; great for learning the basics.
* **Advanced Version** → Adds realistic institutional layers: dynamic hedge, mean-reversion diagnostics, risk, and costs.

Use them as a **research toolkit**, adapt the logic, and always validate with your own tests before using anything in real markets.

```
```
