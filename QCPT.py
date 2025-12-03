import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Nifty 50 tickers
# -----------------------------
nifty50_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

# -----------------------------
# DATA DOWNLOAD
# -----------------------------
@st.cache_data
def download_data(start_date, end_date):
    """Download historical stock data (EOD Close)."""
    data = yf.download(nifty50_tickers + ["^NSEI"], start=start_date, end=end_date, progress=False)['Close']
    returns = data.pct_change().dropna()
    return data, returns

# -----------------------------
# FEATURE COMPUTATION
# -----------------------------
def compute_features(returns, market_returns, window=60):
    """Compute clustering features."""
    features = pd.DataFrame(index=returns.columns)
    
    # Volatility (annualized)
    features['Volatility'] = returns.std() * np.sqrt(252)
    
    # Mean Return (annualized)
    features['Mean_Return'] = returns.mean() * 252
    
    # Beta
    market_var = market_returns.var()
    features['Beta'] = returns.apply(lambda x: x.cov(market_returns) / market_var)
    
    # Sharpe Ratio
    features['Sharpe'] = (features['Mean_Return'] / features['Volatility']).replace([np.inf, -np.inf], 0)
    
    # Rolling volatility stability
    rolling_vol = returns.rolling(window).std()
    features['Vol_Stability'] = 1 / (rolling_vol.std() + 1e-6)
    
    return features.dropna()

# -----------------------------
# CLUSTERING
# -----------------------------
def perform_clustering(features, max_k=7):
    """Perform K-means clustering."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    silhouettes = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        silhouettes.append(silhouette_score(scaled_features, clusters))
    
    optimal_k = np.argmax(silhouettes) + 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    features['Cluster'] = clusters
    
    return features, optimal_k, scaler

# -----------------------------
# COINTEGRATION
# -----------------------------
def test_cointegration(price_data, stock1, stock2):
    """Test cointegration between two stocks."""
    try:
        score, pvalue, _ = coint(price_data[stock1], price_data[stock2])
        return pvalue
    except:
        return 1.0

def find_cointegrated_pairs(price_data, cluster_stocks, top_n=5):
    """Find cointegrated pairs within a cluster."""
    pairs = []
    n = len(cluster_stocks)
    
    for i in range(n):
        for j in range(i+1, n):
            stock1, stock2 = cluster_stocks[i], cluster_stocks[j]
            pvalue = test_cointegration(price_data, stock1, stock2)
            
            if pvalue < 0.05:  # Significant cointegration
                corr = price_data[[stock1, stock2]].corr().iloc[0, 1]
                pairs.append({
                    'Stock1': stock1,
                    'Stock2': stock2,
                    'P-Value': pvalue,
                    'Correlation': corr
                })
    
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values('P-Value').head(top_n)
    return pairs_df

# -----------------------------
# KALMAN FILTER HEDGE RATIO
# -----------------------------
def kalman_filter_hedge_ratio(y, x, delta=1e-5, vt=1e-3):
    """
    Simple Kalman filter to estimate time-varying hedge ratio between y and x.
    y = stock1 price, x = stock2 price.
    """
    y = y.astype(float)
    x = x.astype(float)
    
    beta = np.zeros(len(y))
    P = 1.0
    Q = delta / (1 - delta)  # process variance
    beta_prev = 0.0
    
    for t in range(len(y)):
        # Prediction
        R = P + Q
        # If missing data, carry forward
        if np.isnan(x.iloc[t]) or np.isnan(y.iloc[t]):
            beta[t] = beta_prev
            P = R
            continue
        
        At = x.iloc[t]
        # Kalman gain
        K = R * At / (At * At * R + vt)
        # Prediction error
        e = y.iloc[t] - beta_prev * At
        # Update beta
        beta_curr = beta_prev + K * e
        # Update covariance
        P = (1 - K * At) * R
        
        beta[t] = beta_curr
        beta_prev = beta_curr
    
    return pd.Series(beta, index=y.index, name='Hedge_Ratio')

# -----------------------------
# SPREAD, HALF-LIFE, REGIME
# -----------------------------
def calculate_spread(price_data, stock1, stock2, method="kalman"):
    """
    Calculate spread using dynamic hedge ratio from Kalman filter.
    spread = stock1 - hedge_ratio * stock2
    """
    y = price_data[stock1]
    x = price_data[stock2]
    
    if method == "kalman":
        hedge_ratio = kalman_filter_hedge_ratio(y, x)
    else:
        # Fallback: static OLS hedge ratio
        beta_ols = np.polyfit(x.dropna(), y.dropna(), 1)[0]
        hedge_ratio = pd.Series(beta_ols, index=y.index)
    
    spread = y - hedge_ratio * x
    spread = spread.dropna()
    
    z_score = (spread - spread.mean()) / spread.std()
    return spread, z_score, hedge_ratio.loc[spread.index]

def calculate_half_life(spread):
    """
    Estimate half-life of mean reversion using an OU-like model.
    Δs_t = a + b * s_{t-1} + ε
    Half-life = -ln(2) / b
    """
    s = spread.dropna()
    if len(s) < 30:
        return np.nan
    
    s_lag = s.shift(1).dropna()
    s_ret = s.diff().dropna()
    
    s_lag = s_lag.loc[s_ret.index]
    
    # Regress Δs on s_{t-1}
    b = np.polyfit(s_lag.values, s_ret.values, 1)[0]
    if b >= 0:
        return np.inf
    
    half_life = -np.log(2) / b
    return max(1, half_life)

def detect_regime(spread, lookback=252):
    """
    Simple regime detection using ADF test on recent spread.
    If p-value < 0.05 → Mean-Reverting regime
    Else → Trending / Random Walk regime
    """
    s = spread.dropna()
    if len(s) > lookback:
        s = s[-lookback:]
    if len(s) < 30:
        return "Not Enough Data", np.nan
    
    try:
        adf_result = adfuller(s)
        pvalue = adf_result[1]
        if pvalue < 0.05:
            regime = "Mean-Reverting"
        else:
            regime = "Trending / Random Walk"
        return regime, pvalue
    except:
        return "Unknown", np.nan

# -----------------------------
# RISK: VaR
# -----------------------------
def calculate_var(returns, conf=0.95):
    """
    Parametric VaR (normal) on daily strategy returns.
    Returns absolute VaR (positive number, fraction of capital).
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    mu = r.mean()
    sigma = r.std()
    var = norm.ppf(1 - conf, loc=mu, scale=sigma)  # typically negative
    return abs(var)

# -----------------------------
# SIGNALS
# -----------------------------
def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """Generate trading signals based on z-score."""
    signals = pd.Series(0, index=z_score.index)
    signals[z_score > entry_threshold] = -1  # Short spread
    signals[z_score < -entry_threshold] = 1  # Long spread
    signals[abs(z_score) < exit_threshold] = 0  # Exit
    return signals

# -----------------------------
# BACKTEST WITH INSTITUTIONAL FEATURES
# -----------------------------
def backtest_pair(price_data,
                  stock1,
                  stock2,
                  entry_z=2.0,
                  exit_z=0.5,
                  capital=100000,
                  transaction_cost_bps=0,
                  risk_per_trade_pct=1.0):
    """
    Backtest pair using dynamic Kalman hedge ratio, TC, VaR, half-life, regime.
    """
    # Spread + Z-score + Hedge ratio
    spread, z_score, hedge_ratio = calculate_spread(price_data, stock1, stock2, method="kalman")
    
    # Align price data to spread index
    prices = price_data.loc[spread.index, [stock1, stock2]]
    
    # Signals
    signals = generate_signals(z_score, entry_z, exit_z)
    
    # Returns
    returns = pd.DataFrame(index=spread.index)
    returns['Stock1_Ret'] = prices[stock1].pct_change()
    returns['Stock2_Ret'] = prices[stock2].pct_change()
    
    # Position in spread (1 = long spread, -1 = short spread)
    position = signals.shift(1).fillna(0)
    
    # Strategy returns (per unit capital, before TC)
    # Long spread: +stock1 - hedge*stock2
    # Short spread: -stock1 + hedge*stock2
    returns['Strategy_Gross'] = position * (
        returns['Stock1_Ret'] - hedge_ratio * returns['Stock2_Ret']
    )
    
    # Transaction costs (bps per leg) applied when position changes
    tc_per_leg = transaction_cost_bps / 10000.0
    trade_flags = position.diff().abs()  # 0 → no change, 1 or 2 → change
    # Approx: 2 legs traded when we enter/exit/reverse
    returns['TC'] = -2 * tc_per_leg * trade_flags
    returns['Strategy'] = returns['Strategy_Gross'] + returns['TC']
    
    # Cumulative returns
    returns['Cumulative_Return'] = (1 + returns['Strategy'].fillna(0)).cumprod()
    
    # PnL in currency
    returns['PnL'] = returns['Strategy'].fillna(0) * capital
    returns['Cum_PnL'] = returns['PnL'].cumsum()
    
    # Store extra info
    returns['Spread'] = spread
    returns['Z_Score'] = z_score
    returns['Position'] = position
    returns['Hedge_Ratio'] = hedge_ratio
    
    # Performance metrics
    total_return = (returns['Cumulative_Return'].iloc[-1] - 1) * 100
    if returns['Strategy'].std() > 0:
        sharpe = returns['Strategy'].mean() / returns['Strategy'].std() * np.sqrt(252)
    else:
        sharpe = 0
    max_drawdown = ((returns['Cumulative_Return'].cummax() - returns['Cumulative_Return']) /
                    returns['Cumulative_Return'].cummax()).max() * 100
    
    # Mean-reversion half-life & regime
    half_life = calculate_half_life(spread)
    regime, adf_pvalue = detect_regime(spread)
    
    # Risk: VaR and suggested position size
    daily_var_frac = calculate_var(returns['Strategy'])
    if np.isnan(daily_var_frac) or daily_var_frac == 0:
        suggested_position_capital = (risk_per_trade_pct / 100.0) * capital
    else:
        target_loss = (risk_per_trade_pct / 100.0) * capital
        suggested_position_capital = min(capital, target_loss / daily_var_frac)
    
    return {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Trades': int((position.diff() != 0).sum()),
        'Half-life (days)': half_life,
        'Regime': regime,
        'ADF p-value': adf_pvalue,
        'Daily VaR (%)': daily_var_frac * 100 if not np.isnan(daily_var_frac) else np.nan,
        'Recommended Position Capital (₹)': suggested_position_capital,
        'returns': returns,
        'z_score': z_score,
        'signals': signals
    }

# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Pairs Trading System", layout="wide")
    
    st.title("🔄 Professional Pairs Trading System (EOD)")
    st.markdown("### Institutional-Style Statistical Arbitrage on Nifty 50 (End-of-Day Data)")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Date selection
    default_start = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    start_date = st.sidebar.date_input("Start Date", value=datetime.strptime(default_start, '%Y-%m-%d'))
    end_date = st.sidebar.date_input("End Date", value=datetime.strptime(default_end, '%Y-%m-%d'))
    
    # Clustering parameters
    max_clusters = st.sidebar.slider("Max Clusters", 3, 10, 7)
    
    # Trading parameters
    st.sidebar.subheader("Trading Parameters")
    entry_threshold = st.sidebar.slider("Entry Z-Score", 1.5, 3.0, 2.0, 0.1)
    exit_threshold = st.sidebar.slider("Exit Z-Score", 0.0, 1.0, 0.5, 0.1)
    
    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    capital = st.sidebar.number_input("Total Capital (₹)", min_value=100000, max_value=10000000, value=1000000, step=50000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (% of capital)", 0.1, 5.0, 1.0, 0.1)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps per leg)", 0, 50, 5, 1)
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    with st.spinner("Loading market data..."):
        data, returns = download_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
    if len(returns) == 0:
        st.error("❌ No data available. Please check your date range.")
        return
    
    market_returns = returns['^NSEI']
    stock_returns = returns.drop(columns=['^NSEI'])
    price_data = data.drop(columns=['^NSEI'])
    
    # Compute features and cluster
    with st.spinner("Performing clustering analysis..."):
        features = compute_features(stock_returns, market_returns)
        features, optimal_k, scaler = perform_clustering(features, max_clusters)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Analysis", "🎯 Pair Selection", "📈 Backtest", "📉 Live Signals"])
    
    # -------------------------
    # TAB 1: CLUSTER ANALYSIS
    # -------------------------
    with tab1:
        st.header("Cluster Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Clusters", optimal_k)
        col2.metric("Total Stocks", len(features))
        col3.metric("Date Range", f"{(end_date - start_date).days} days")
        
        # Cluster summary
        st.subheader("Cluster Characteristics")
        summary = features.groupby('Cluster').agg({
            'Volatility': 'mean',
            'Mean_Return': 'mean',
            'Beta': 'mean',
            'Sharpe': 'mean'
        }).round(4)
        summary['Count'] = features.groupby('Cluster').size()
        summary = summary.sort_values('Volatility')
        st.dataframe(summary, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(features['Beta'], features['Volatility'], 
                               c=features['Cluster'], cmap='tab10', s=100, alpha=0.6)
            ax.set_xlabel('Beta', fontsize=12)
            ax.set_ylabel('Volatility', fontsize=12)
            ax.set_title('Stock Clusters: Beta vs Volatility', fontsize=14)
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(features['Mean_Return'], features['Volatility'], 
                               c=features['Cluster'], cmap='tab10', s=100, alpha=0.6)
            ax.set_xlabel('Mean Return', fontsize=12)
            ax.set_ylabel('Volatility', fontsize=12)
            ax.set_title('Return vs Risk Profile', fontsize=14)
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
    
    # -------------------------
    # TAB 2: PAIR SELECTION
    # -------------------------
    with tab2:
        st.header("Cointegrated Pair Selection")
        
        # Cluster selection
        selected_cluster = st.selectbox("Select Cluster", 
                                       options=sorted(features['Cluster'].unique()),
                                       format_func=lambda x: f"Cluster {x}")
        
        cluster_stocks = features[features['Cluster'] == selected_cluster].index.tolist()
        
        st.info(f"📌 Cluster {selected_cluster} contains {len(cluster_stocks)} stocks")
        
        # Show stocks in cluster
        with st.expander("View Stocks in Cluster"):
            cluster_df = features[features['Cluster'] == selected_cluster][['Volatility', 'Mean_Return', 'Beta', 'Sharpe']]
            st.dataframe(cluster_df.sort_values('Volatility', ascending=False), use_container_width=True)
        
        # Find cointegrated pairs
        if len(cluster_stocks) >= 2:
            with st.spinner("Testing for cointegrated pairs..."):
                pairs_df = find_cointegrated_pairs(price_data, cluster_stocks, top_n=10)
            
            if not pairs_df.empty:
                st.success(f"✅ Found {len(pairs_df)} cointegrated pairs (p-value < 0.05)")
                st.dataframe(pairs_df.style.format({
                    'P-Value': '{:.4f}',
                    'Correlation': '{:.4f}'
                }), use_container_width=True)
                
                # Store in session state
                st.session_state['pairs_df'] = pairs_df
                st.session_state['cluster_stocks'] = cluster_stocks
            else:
                st.warning("⚠️ No significant cointegrated pairs found in this cluster. Try another cluster.")
        else:
            st.warning("⚠️ Need at least 2 stocks in cluster.")
    
    # -------------------------
    # TAB 3: BACKTEST
    # -------------------------
    with tab3:
        st.header("Pair Backtest with Kalman Hedge, Half-Life, VaR")
        
        if 'pairs_df' in st.session_state and not st.session_state['pairs_df'].empty:
            pairs_df = st.session_state['pairs_df']
            
            # Pair selection
            pair_options = [f"{row['Stock1']} / {row['Stock2']}" for _, row in pairs_df.iterrows()]
            selected_pair = st.selectbox("Select Pair to Backtest", pair_options)
            
            if selected_pair:
                stock1, stock2 = selected_pair.split(' / ')
                pair_row = pairs_df[(pairs_df['Stock1'] == stock1) & (pairs_df['Stock2'] == stock2)].iloc[0]
                
                # Run backtest
                with st.spinner("Running backtest..."):
                    results = backtest_pair(price_data, stock1, stock2,
                                            entry_threshold, exit_threshold,
                                            capital, transaction_cost_bps,
                                            risk_per_trade)
                
                # Display metrics (return + risk)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{results['Total Return (%)']:.2f}%")
                col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                col3.metric("Max Drawdown", f"{results['Max Drawdown (%)']:.2f}%")
                col4.metric("Total Trades", results['Trades'])
                
                col5, col6, col7, col8 = st.columns(4)
                hl = results['Half-life (days)']
                hl_display = "∞" if np.isinf(hl) else f"{hl:.1f}"
                col5.metric("Half-Life (days)", hl_display)
                col6.metric("Regime", results['Regime'])
                if not np.isnan(results['Daily VaR (%)']):
                    col7.metric("Daily VaR (95%)", f"{results['Daily VaR (%)']:.2f}%")
                else:
                    col7.metric("Daily VaR (95%)", "NA")
                col8.metric("Suggested Position (₹)", f"{results['Recommended Position Capital (₹)']:.0f}")
                
                # Explanation: why this pair is traded together
                st.subheader("Why are we trading this pair together?")
                explain_text = f"""
- **Cointegration**: The pair **{stock1} / {stock2}** has a cointegration p-value of **{pair_row['P-Value']:.4f}**,  
  which means they move together in the long run and their **spread is stable**.
- **Correlation**: The price correlation is **{pair_row['Correlation']:.2f}**,  
  so big moves are usually related, not random.
- **Mean Reversion**: The spread has an estimated **half-life of {hl_display} days**  
  in a **{results['Regime']}** regime. So, when the spread moves away,  
  we statistically expect it to come back.
- **Signal Logic** (Standard Signal):
    - If **Z-score > +{entry_threshold:.1f}** → spread is too wide → **SHORT spread**  
      (Short {stock1}, Long {stock2})
    - If **Z-score < -{entry_threshold:.1f}** → spread is too tight/low → **LONG spread**  
      (Long {stock1}, Short {stock2})
    - If |Z-score| < {exit_threshold:.1f} → spread is normal → **EXIT / CLOSE** positions.
"""
                st.markdown(explain_text)
                
                # Plot results
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['returns']['Cumulative_Return'], linewidth=2)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Cumulative Return')
                    ax.set_title('Strategy Performance (Return)')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['z_score'], linewidth=1.5, label='Z-Score')
                    ax.axhline(entry_threshold, linestyle='--', alpha=0.7, label='Entry +')
                    ax.axhline(-entry_threshold, linestyle='--', alpha=0.7, label='Entry -')
                    ax.axhline(exit_threshold, linestyle='--', alpha=0.7, label='Exit band')
                    ax.axhline(-exit_threshold, linestyle='--', alpha=0.7)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Z-Score')
                    ax.set_title('Spread Z-Score with Entry/Exit Bands')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                
                # Extra visuals: PnL + Spread with signals
                with st.expander("📊 View P&L Curve and Spread with Signals"):
                    # P&L
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(results['returns']['Cum_PnL'], linewidth=2)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Cumulative P&L (₹)')
                    ax.set_title('Cumulative P&L (After TC)')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    
                    # Spread + signals
                    fig, ax = plt.subplots(figsize=(10, 5))
                    spread = results['returns']['Spread']
                    z = results['returns']['Z_Score']
                    pos = results['returns']['Position']
                    ax.plot(spread, label='Spread', linewidth=1.5)
                    # Mark long/short entries
                    long_entries = (pos == 1) & (pos.diff() == 1)
                    short_entries = (pos == -1) & (pos.diff() == -1)
                    ax.scatter(spread.index[long_entries], spread[long_entries], marker='^', s=60, label='Long Entry')
                    ax.scatter(spread.index[short_entries], spread[short_entries], marker='v', s=60, label='Short Entry')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Spread')
                    ax.set_title('Spread with Trading Signals')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
        else:
            st.info("👈 Please select a cluster and find cointegrated pairs first.")
    
    # -------------------------
    # TAB 4: LIVE (EOD) SIGNALS
    # -------------------------
    with tab4:
        st.header("Live EOD Trading Signals (based on latest close)")
        
        if 'pairs_df' in st.session_state and not st.session_state['pairs_df'].empty:
            pairs_df = st.session_state['pairs_df']
            
            # Get latest signals for all pairs
            signals_list = []
            for _, row in pairs_df.iterrows():
                stock1, stock2 = row['Stock1'], row['Stock2']
                spread, z_score, hedge_ratio = calculate_spread(price_data, stock1, stock2)
                
                current_z = z_score.iloc[-1]
                
                if current_z > entry_threshold:
                    signal = "🔴 SHORT Spread"
                    action = f"Short {stock1}, Long {stock2}"
                elif current_z < -entry_threshold:
                    signal = "🟢 LONG Spread"
                    action = f"Long {stock1}, Short {stock2}"
                elif abs(current_z) < exit_threshold:
                    signal = "⚪ EXIT"
                    action = "Close positions"
                else:
                    signal = "⏸️ HOLD"
                    action = "No action"
                
                signals_list.append({
                    'Pair': f"{stock1} / {stock2}",
                    'Current Z-Score': current_z,
                    'Signal': signal,
                    'Action': action,
                    'P-Value': row['P-Value']
                })
            
            signals_df = pd.DataFrame(signals_list)
            
            # Filter active signals
            st.subheader("Active Signals (EOD)")
            active = signals_df[signals_df['Signal'].isin(['🔴 SHORT Spread', '🟢 LONG Spread'])]
            
            if not active.empty:
                st.dataframe(active.style.format({'Current Z-Score': '{:.2f}', 'P-Value': '{:.4f}'}), 
                           use_container_width=True)
            else:
                st.info("No active signals at current thresholds based on latest EOD data.")
            
            # All pairs status
            with st.expander("View All Pairs Status"):
                st.dataframe(signals_df.style.format({'Current Z-Score': '{:.2f}', 'P-Value': '{:.4f}'}), 
                           use_container_width=True)
        else:
            st.info("👈 Please select a cluster and find cointegrated pairs first.")
    
    # Export section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Data")
    if st.sidebar.button("Download Cluster Data"):
        csv = features.to_csv()
        st.sidebar.download_button("Download CSV", csv, "clusters.csv", "text/csv")

if __name__ == "__main__":
    main()
