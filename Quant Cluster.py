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
import warnings
warnings.filterwarnings('ignore')

# Nifty 50 tickers
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

@st.cache_data
def download_data(start_date, end_date):
    """Download historical stock data"""
    data = yf.download(nifty50_tickers + ["^NSEI"], start=start_date, end=end_date, progress=False)['Close']
    returns = data.pct_change().dropna()
    return data, returns

def compute_features(returns, market_returns, window=60):
    """Compute clustering features"""
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

def perform_clustering(features, max_k=7):
    """Perform K-means clustering"""
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

def test_cointegration(price_data, stock1, stock2):
    """Test cointegration between two stocks"""
    try:
        score, pvalue, _ = coint(price_data[stock1], price_data[stock2])
        return pvalue
    except:
        return 1.0

def find_cointegrated_pairs(price_data, cluster_stocks, top_n=5):
    """Find cointegrated pairs within a cluster"""
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

def calculate_spread(price_data, stock1, stock2):
    """Calculate normalized spread for pair"""
    ratio = price_data[stock1] / price_data[stock2]
    spread = ratio - ratio.mean()
    z_score = (spread - spread.mean()) / spread.std()
    return spread, z_score

def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """Generate trading signals based on z-score"""
    signals = pd.Series(0, index=z_score.index)
    signals[z_score > entry_threshold] = -1  # Short spread
    signals[z_score < -entry_threshold] = 1   # Long spread
    signals[abs(z_score) < exit_threshold] = 0  # Exit
    return signals

def backtest_pair(price_data, stock1, stock2, entry_z=2.0, exit_z=0.5, capital=100000):
    """Simple backtest for a pair"""
    spread, z_score = calculate_spread(price_data, stock1, stock2)
    signals = generate_signals(z_score, entry_z, exit_z)
    
    # Calculate returns
    returns = pd.DataFrame(index=price_data.index)
    returns['Stock1_Ret'] = price_data[stock1].pct_change()
    returns['Stock2_Ret'] = price_data[stock2].pct_change()
    
    # Calculate strategy returns
    position = signals.shift(1).fillna(0)
    returns['Strategy'] = position * (returns['Stock1_Ret'] - returns['Stock2_Ret'])
    returns['Cumulative'] = (1 + returns['Strategy']).cumprod()
    
    # Performance metrics
    total_return = (returns['Cumulative'].iloc[-1] - 1) * 100
    sharpe = returns['Strategy'].mean() / returns['Strategy'].std() * np.sqrt(252) if returns['Strategy'].std() > 0 else 0
    max_drawdown = ((returns['Cumulative'].cummax() - returns['Cumulative']) / returns['Cumulative'].cummax()).max() * 100
    
    return {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Trades': (position.diff() != 0).sum(),
        'returns': returns,
        'z_score': z_score,
        'signals': signals
    }

# Streamlit App
def main():
    st.set_page_config(page_title="Pairs Trading System", layout="wide")
    
    st.title("🔄 Professional Pairs Trading System")
    st.markdown("### Statistical Arbitrage for Nifty 50 Stocks")
    
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
    
    with tab3:
        st.header("Pair Backtest")
        
        if 'pairs_df' in st.session_state and not st.session_state['pairs_df'].empty:
            pairs_df = st.session_state['pairs_df']
            
            # Pair selection
            pair_options = [f"{row['Stock1']} / {row['Stock2']}" for _, row in pairs_df.iterrows()]
            selected_pair = st.selectbox("Select Pair to Backtest", pair_options)
            
            if selected_pair:
                stock1, stock2 = selected_pair.split(' / ')
                
                # Run backtest
                with st.spinner("Running backtest..."):
                    results = backtest_pair(price_data, stock1, stock2, entry_threshold, exit_threshold)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{results['Total Return (%)']:.2f}%")
                col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                col3.metric("Max Drawdown", f"{results['Max Drawdown (%)']:.2f}%")
                col4.metric("Total Trades", results['Trades'])
                
                # Plot results
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['returns']['Cumulative'], linewidth=2)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Cumulative Return')
                    ax.set_title('Strategy Performance')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['z_score'], linewidth=1.5, label='Z-Score')
                    ax.axhline(entry_threshold, color='r', linestyle='--', alpha=0.7, label='Entry')
                    ax.axhline(-entry_threshold, color='r', linestyle='--', alpha=0.7)
                    ax.axhline(exit_threshold, color='g', linestyle='--', alpha=0.7, label='Exit')
                    ax.axhline(-exit_threshold, color='g', linestyle='--', alpha=0.7)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Z-Score')
                    ax.set_title('Spread Z-Score')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
        else:
            st.info("👈 Please select a cluster and find cointegrated pairs first.")
    
    with tab4:
        st.header("Live Trading Signals")
        
        if 'pairs_df' in st.session_state and not st.session_state['pairs_df'].empty:
            pairs_df = st.session_state['pairs_df']
            
            # Get latest signals for all pairs
            signals_list = []
            for _, row in pairs_df.iterrows():
                stock1, stock2 = row['Stock1'], row['Stock2']
                spread, z_score = calculate_spread(price_data, stock1, stock2)
                
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
            st.subheader("Active Signals")
            active = signals_df[signals_df['Signal'].isin(['🔴 SHORT Spread', '🟢 LONG Spread'])]
            
            if not active.empty:
                st.dataframe(active.style.format({'Current Z-Score': '{:.2f}', 'P-Value': '{:.4f}'}), 
                           use_container_width=True)
            else:
                st.info("No active signals at current thresholds.")
            
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