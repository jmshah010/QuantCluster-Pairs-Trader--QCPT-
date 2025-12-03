# QuantCluster Pairs Trader (QCPT)

````markdown
# Nifty 50 Institutional-Style Pairs Trading Terminal (EOD)

A professional **Streamlit-based pairs trading terminal** for the **Nifty 50** universe, built for **end-of-day (EOD)** analysis and **institutional-style risk management**.

This project combines **machine learning**, **time-series econometrics**, and **portfolio risk techniques** to discover and trade statistically robust pair relationships in Indian equities.

---

## 🔍 Overview

This application:

- Downloads **EOD prices** for all Nifty 50 stocks and the **Nifty index (^NSEI)** via `yfinance`
- Builds **risk/return features** for each stock
- Uses **K-Means clustering** to group similar stocks
- Finds **cointegrated pairs** within each cluster
- Trades the spread with:
  - **Kalman filter–based dynamic hedge ratio**
  - **Z-score based mean reversion signals**
  - **Half-life estimation**
  - **Regime detection** (mean-reverting vs trending)
  - **VaR-based risk sizing**
  - **Transaction cost modelling**
- Provides a clean **Streamlit UI** for:
  - Cluster analysis
  - Pair selection
  - Backtesting
  - “Live” (latest EOD) signals

> ⚠️ **Disclaimer**: This project is for **educational and research purposes only**. It is **not** investment advice. Do not use in live trading without your own validation and risk checks.

---

## ✨ Key Features

### 1. Universe & Data

- Nifty 50 stocks, plus **Nifty index (^NSEI)** for market benchmark
- **End-of-day close prices** fetched via `yfinance`
- Automatically handles **configurable lookback periods**

### 2. Feature Engineering & Clustering

- Per-stock features:
  - Annualized **Volatility**
  - Annualized **Mean Return**
  - **Beta** vs Nifty (^NSEI)
  - **Sharpe Ratio**
  - **Volatility Stability** (rolling vol variability)
- **K-Means clustering** on standardized features
- Automatic selection of **optimal cluster count** using **silhouette score**
- Cluster-level summary with:
  - Mean volatility, return, beta, Sharpe
  - Stock count per cluster

### 3. Pair Selection (Within Cluster)

- For a selected cluster:
  - Tests **cointegration** for all stock pairs via `statsmodels.tsa.stattools.coint`
  - Filters significant pairs with **p-value < 0.05**
  - Computes **price correlation** for each pair
- Display of **top cointegrated pairs** (sorted by p-value) with:
  - `Stock1`, `Stock2`
  - `P-Value`
  - `Correlation`

### 4. Trading Logic (Institutional-Style)

For each selected pair:

- **Kalman Filter Hedge Ratio**
  - Dynamic hedge ratio between the two legs
  - Spread defined as:  
    `spread_t = Price1_t – HedgeRatio_t × Price2_t`

- **Spread Z-Score**
  - Normalization of spread to identify statistically extreme deviations

- **Signal Rules (Standard)**
  - If `Z > +EntryZ` → **Short spread**  
    (Short Stock1, Long Stock2)
  - If `Z < -EntryZ` → **Long spread**  
    (Long Stock1, Short Stock2)
  - If `|Z| < ExitZ` → **Exit / Close** positions

- **Mean Reversion Half-Life**
  - OU-style regression of Δspread on lagged spread
  - Half-life = `-ln(2) / b`  
    (approximate days for half mean-reversion)

- **Regime Detection**
  - ADF test on recent spread
  - If p-value < 0.05 → **Mean-Reverting**
  - Else → **Trending / Random Walk**

### 5. Risk Management & P&L

- **Parametric VaR (95%)** on daily strategy returns
- Suggestion of **position size** based on:
  - Total capital
  - Desired **risk-per-trade (%)**
  - Estimated daily VaR
- **Transaction Costs**:
  - User-defined **bps per leg**
  - Applied whenever the position changes (entry/exit/reversal)
- P&L:
  - Daily strategy returns
  - Cumulative returns
  - Currency **PnL and cumulative PnL**

---

## 🧱 Application Structure

Single Streamlit script (example):

```text
.
└── QCPT.py        # Main Streamlit application with all logic
````

You can rename `app.py` as you like, but ensure you use the correct file name when running Streamlit.

---

## 📦 Requirements

* Python 3.8+
* Libraries:

  * `yfinance`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`
  * `seaborn`
  * `streamlit`
  * `statsmodels`
  * `scipy`

Install them via:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn streamlit statsmodels scipy
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(If you don’t have a `requirements.txt`, you can create one using the package list above.)*

### 3. Run the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

---

## 🧭 Using the App

The UI is organized into **four main tabs**:

### 1️⃣ Cluster Analysis

* Shows:

  * Optimal number of clusters
  * Total stocks
  * Date range used
* Cluster summary table:

  * Mean Volatility, Mean Return, Beta, Sharpe, Count
* Scatter plots:

  * **Beta vs Volatility** (colored by cluster)
  * **Mean Return vs Volatility** (colored by cluster)

Use this tab to understand **risk/return clusters** before selecting pairs.

---

### 2️⃣ Pair Selection

* Choose a **cluster** from the dropdown.
* See:

  * All stocks in that cluster (with Volatility, Mean Return, Beta, Sharpe)
* App then:

  * Tests for **cointegration** among stocks in that cluster
  * Displays **top cointegrated pairs** with:

    * `Stock1`, `Stock2`, `P-Value`, `Correlation`

This tab answers:

> “Which pairs inside this risk/return cluster are **statistically connected** and suitable for mean-reversion trading?”

---

### 3️⃣ Backtest

* Select a pair: `Stock1 / Stock2`
* Configure:

  * Entry Z-score
  * Exit Z-score
  * Total capital (₹)
  * Risk per trade (%)
  * Transaction cost (bps per leg)
* The app runs a **full backtest** using:

  * Kalman filter hedge ratio
  * Z-score entry/exit rules
  * Transaction cost adjustments

**Outputs:**

* Metrics:

  * Total Return (%)
  * Sharpe Ratio
  * Max Drawdown (%)
  * Number of Trades
  * Half-life (days)
  * Regime (Mean-Reverting / Trending)
  * Daily VaR (95%)
  * Suggested Position Size (₹)
* Charts:

  * **Cumulative strategy return**
  * **Z-score with entry/exit bands**
  * **Cumulative PnL (₹)**
  * **Spread with marked entry/exit points**

**Explanation Block** (“Why are we trading this pair together?”):

* Cointegration p-value and interpretation
* Correlation between legs
* Half-life and regime description
* Clear description of signal logic (when we long/short/exit)

---

### 4️⃣ Live EOD Signals

* Uses **latest EOD prices** for all selected cointegrated pairs
* For each pair:

  * Computes current spread and Z-score
  * Generates:

    * 🔴 SHORT Spread
    * 🟢 LONG Spread
    * ⚪ EXIT
    * ⏸️ HOLD
  * Shows recommended action (e.g., “Short Stock1, Long Stock2”)
* Dedicated section for:

  * **Active signals only**
  * Full table of all pairs with current Z-score and p-value

This tab turns the engine into a **simple overnight signal dashboard** based on end-of-day data.

---

## 🧠 Methodology (High-Level)

1. **Feature Space Construction**

   * Build a risk/return feature vector for each stock.
   * Scale features and perform **K-Means** to group similar stocks.

2. **Within-Cluster Cointegration**

   * Focus only on “similar” stocks in each cluster.
   * Run pairwise cointegration tests; keep statistically robust pairs.

3. **Dynamic Hedge Ratio via Kalman Filter**

   * Instead of static OLS, use a **Kalman filter** to estimate time-varying hedge ratio.
   * Allows the spread definition to adapt as relative stock dynamics change.

4. **Mean-Reversion Framework**

   * Define spread via hedge ratio.
   * Normalize it to Z-score.
   * Use **half-life** and **ADF-based regime test** to confirm mean-reversion.

5. **Risk Management**

   * Parametric **VaR (95%)** on strategy returns.
   * Convert risk budget (% of capital) and VaR into **suggested position size**.
   * Apply **transaction costs** in bps per leg for more realistic PnL.

---

## 📌 Limitations & Future Enhancements

* Uses **EOD data only** (no intraday or order-book microstructure)
* VaR is **normal-distribution-based** (no fat-tail / non-parametric modelling)
* Assumes **constant transaction cost** in bps
* No portfolio-level multi-pair risk aggregation yet

Possible future improvements:

* Multi-pair **portfolio VaR / CVaR**
* **Stress testing** (index/sector shocks)
* **Multi-timeframe confirmation** using higher-frequency data
* Extension to **sector indices** and global markets

---

## ⚠️ Disclaimer

This repository is intended **solely for educational, academic, and research use**.
Past performance of any backtested strategy **does not guarantee** future results.

You are responsible for:

* Verifying all results,
* Understanding the risks,
* Complying with all regulatory requirements in your jurisdiction.

Use at your own risk.

---

```
```
