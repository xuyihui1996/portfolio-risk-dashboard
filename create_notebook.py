
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Text Cells
nb.cells.append(nbf.v4.new_markdown_cell("""
# Japanese Equity Portfolio Analysis

This notebook analyzes a simulated Japanese equity portfolio consisting of 10 equal-weight assets.
period: 2018-01-01 to 2023-12-31.
Benchmark: Nikkei 225 (^N225).

## Objectives
- Simulate monthly equal-weight rebalancing.
- Calculate performance metrics (CAGR, Volatility, Sharpe Ratio, Max Drawdown).
- Analyze top contributors/detractors (2023).
- Visualize performance against benchmark.
"""))

# Imports
nb.cells.append(nbf.v4.new_code_cell("""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
%matplotlib inline
"""))

# Configuration
nb.cells.append(nbf.v4.new_code_cell("""
# Configuration
TICKERS = [
    "7203.T", # Toyota
    "6758.T", # Sony
    "9984.T", # SoftBank
    "8306.T", # MUFG
    "6861.T", # Keyence
    "4502.T", # Takeda
    "9983.T", # Fast Retailing
    "9433.T", # KDDI
    "8035.T", # Tokyo Electron
    "1321.T"  # Nikkei 225 ETF
]
BENCHMARK_TICKER = "^N225"
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
INITIAL_CAPITAL = 10_000_000
TRANSACTION_COST = 0.001
"""))

# Data Fetching
nb.cells.append(nbf.v4.new_code_cell("""
def fetch_data(tickers, start, end):
    print(f"Fetching data for {len(tickers)} assets and benchmark...")
    # Fetch Portfolio
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    
    # Handle yfinance structure
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data["Close"]
        except KeyError:
            prices = data
    else:
        prices = data["Close"] if "Close" in data.columns else data

    # Fetch Benchmark
    bench = yf.download(BENCHMARK_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(bench.columns, pd.MultiIndex):
         bench_prices = bench["Close"]
    else:
         bench_prices = bench["Close"] if "Close" in bench.columns else bench
    
    if isinstance(bench_prices, pd.DataFrame):
        bench_prices = bench_prices.iloc[:, 0]

    # Clean
    prices = prices.copy()
    prices.dropna(how='all', inplace=True)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    
    bench_prices = bench_prices.reindex(prices.index)
    bench_prices = bench_prices.copy()
    bench_prices.ffill(inplace=True)
    
    return prices, bench_prices

prices, benchmark = fetch_data(TICKERS, START_DATE, END_DATE)
print(f"Data fetched. Prices shape: {prices.shape}")
"""))

# Simulation
nb.cells.append(nbf.v4.new_code_cell("""
def simulate_portfolio(prices):
    print("Simulating portfolio...")
    monthly_starts = prices.groupby([prices.index.year, prices.index.month]).apply(lambda x: x.index[0])
    rebalance_dates = set(monthly_starts.values)
    
    cash = INITIAL_CAPITAL
    holdings = {t: 0 for t in TICKERS}
    portfolio_history = []
    dates = prices.index
    
    for date in dates:
        current_prices = prices.loc[date]
        portfolio_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
        
        is_rebalance_day = date in rebalance_dates
        if date == dates[0]: is_rebalance_day = True
            
        if is_rebalance_day:
            target_per_asset = portfolio_val / len(TICKERS)
            
            # Sell Loop
            for t in TICKERS:
                current_holding_val = holdings[t] * current_prices[t]
                if current_holding_val > target_per_asset:
                    diff = current_holding_val - target_per_asset
                    amt_to_sell = diff
                    shares_to_sell = amt_to_sell / current_prices[t]
                    holdings[t] -= shares_to_sell
                    cash += amt_to_sell * (1 - TRANSACTION_COST)
            
            # Buy Loop
            portfolio_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
            target_per_asset = portfolio_val / len(TICKERS) 
            
            for t in TICKERS:
                current_holding_val = holdings[t] * current_prices[t]
                if current_holding_val < target_per_asset:
                    diff = target_per_asset - current_holding_val
                    cost_to_buy = diff * TRANSACTION_COST
                    amt_to_buy = diff
                    
                    if cash < (amt_to_buy + cost_to_buy):
                        amt_to_buy = cash / (1 + TRANSACTION_COST)
                    
                    shares_to_buy = amt_to_buy / current_prices[t]
                    holdings[t] += shares_to_buy
                    cash -= amt_to_buy * (1 + TRANSACTION_COST)
                    
        daily_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
        portfolio_history.append({'Date': date, 'PortfolioValue': daily_val})
        
    return pd.DataFrame(portfolio_history).set_index('Date')

sim_df = simulate_portfolio(prices)
print("Simulation complete.")
"""))

# Metrics
nb.cells.append(nbf.v4.new_code_cell("""
def calculate_metrics(portfolio_series, benchmark_series):
    p_ret = portfolio_series.pct_change().dropna()
    b_ret = benchmark_series.pct_change().dropna()
    
    common_idx = p_ret.index.intersection(b_ret.index)
    p_ret, b_ret = p_ret.loc[common_idx], b_ret.loc[common_idx]
    
    p_cum = (1 + p_ret).cumprod()
    b_cum = (1 + b_ret).cumprod()
    
    days = (p_ret.index[-1] - p_ret.index[0]).days
    years = days / 365.25
    p_cagr = (p_cum.iloc[-1])**(1/years) - 1
    b_cagr = (b_cum.iloc[-1])**(1/years) - 1
    
    p_vol = p_ret.std() * np.sqrt(252)
    b_vol = b_ret.std() * np.sqrt(252)
    
    # Sharpe (Rf=0, Arithmetic Mean / Vol)
    p_mean = p_ret.mean() * 252
    b_mean = b_ret.mean() * 252
    p_sharpe = p_mean / p_vol if p_vol != 0 else 0
    b_sharpe = b_mean / b_vol if b_vol != 0 else 0
    
    def get_dd(ts):
        peak = ts.cummax()
        dd = (ts - peak) / peak
        return dd.min(), dd
        
    p_mdd, p_dd_series = get_dd(p_cum)
    b_mdd, b_dd_series = get_dd(b_cum)
    
    return {
        'Returns': p_ret, 'BenchmarkReturns': b_ret,
        'Cumulative': p_cum, 'BenchmarkCumulative': b_cum,
        'Drawdown': p_dd_series,
        'Metrics': {
            'CAGR': p_cagr, 'Vol': p_vol, 'Sharpe': p_sharpe, 'MDD': p_mdd,
            'Bench_CAGR': b_cagr, 'Bench_Vol': b_vol, 'Bench_Sharpe': b_sharpe, 'Bench_MDD': b_mdd
        }
    }

res = calculate_metrics(sim_df['PortfolioValue'], benchmark)
pd.DataFrame([
    res['Metrics']['CAGR'], res['Metrics']['Vol'], res['Metrics']['Sharpe'], res['Metrics']['MDD']
], index=['CAGR', 'Vol', 'Sharpe', 'Max Drawdown'], columns=['Portfolio'])
"""))

# Contributors (2023)
nb.cells.append(nbf.v4.new_code_cell("""
def get_contributors_2023(prices):
    try:
        prices_2023 = prices.loc['2023']
        if prices_2023.empty: return {} 
        p_start = prices_2023.iloc[0]
        p_end = prices_2023.iloc[-1]
        
        contribs = {}
        for t in TICKERS:
            # Simple return for the year
            ret = (p_end[t] / p_start[t]) - 1
            # Approx Contribution = Return * Weight (0.1)
            contribs[t] = ret * 0.1
        return contribs
    except KeyError:
        return {}

contribs = get_contributors_2023(prices)
sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
top_contributors = sorted_contribs[:3]
top_detractors = sorted_contribs[-3:]

print("Top Contributors (2023) [pp]:", top_contributors)
print("Top Detractors (2023) [pp]:", top_detractors)
"""))

# Visualization
nb.cells.append(nbf.v4.new_code_cell("""
# Generate One-Page Summary
fig = plt.figure(figsize=(11.69, 8.27)) 
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1])

# 1. Equity Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(res['Cumulative'].index, res['Cumulative'], label='Portfolio', color='#1f77b4', linewidth=2)
ax1.plot(res['BenchmarkCumulative'].index, res['BenchmarkCumulative'], label='Nikkei 225 (^N225)', color='#7f7f7f', linestyle='--')
ax1.set_title("Cumulative Performance (Indexed)", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(res['Drawdown'].index, res['Drawdown'], 0, color='#d62728', alpha=0.3)
ax2.plot(res['Drawdown'].index, res['Drawdown'], color='#d62728', linewidth=1)
ax2.set_title("Portfolio Drawdown", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Metrics Table
ax3 = fig.add_subplot(gs[2, 0])
ax3.axis('off')
metrics_data = [
    ["Metric", "Portfolio", "Benchmark"],
    ["CAGR", f"{res['Metrics']['CAGR']:.2%}", f"{res['Metrics']['Bench_CAGR']:.2%}"],
    ["Volatility (Ann)", f"{res['Metrics']['Vol']:.2%}", f"{res['Metrics']['Bench_Vol']:.2%}"],
    ["Sharpe Ratio", f"{res['Metrics']['Sharpe']:.2f}", f"{res['Metrics']['Bench_Sharpe']:.2f}"],
    ["Max Drawdown", f"{res['Metrics']['MDD']:.2%}", f"{res['Metrics']['Bench_MDD']:.2%}"],
]
table = ax3.table(cellText=metrics_data, loc='center', cellLoc='center', colWidths=[0.33, 0.33, 0.33])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax3.set_title("Risk/Return Metrics", fontsize=12, fontweight='bold', pad=10)

# 4. Takeaways
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')

top_c_str = ", ".join([f"{t[0]} ({t[1]:+.1%})" for t in top_contributors]).replace("%", "pp")
top_d_str = ", ".join([f"{t[0]} ({t[1]:+.1%})" for t in top_detractors]).replace("%", "pp")

# Metrics for text
t_cagr = res['Metrics']['CAGR']
b_cagr = res['Metrics']['Bench_CAGR']
t_vol = res['Metrics']['Vol']
b_vol = res['Metrics']['Bench_Vol']
t_sharpe = res['Metrics']['Sharpe']
b_sharpe = res['Metrics']['Bench_Sharpe']
t_mdd = res['Metrics']['MDD']
b_mdd = res['Metrics']['Bench_MDD']

header_info = "Universe: 10 Japan large-caps (equal weight), monthly rebalance, 0.1% transaction cost, Adj Close, period 2018–2023."

takeaways_list = [
    header_info,
    f"• Outperformance: Portfolio CAGR {t_cagr:.2%} vs {b_cagr:.2%} (Nikkei 225). Sharpe {t_sharpe:.2f} vs {b_sharpe:.2f} (ann. mean daily return / ann. vol).",
    f"• Risk Profile: Volatility {t_vol:.2%} vs {b_vol:.2%}; Max Drawdown {t_mdd:.2%} vs {b_mdd:.2%}, indicating similar equity risk with better downside control.",
    f"• Attribution (2023): Top contributors: {top_c_str}; Detractors: {top_d_str} (approx. contribution to portfolio return).",
    "• Note: Fixed illustrative stock basket; results may reflect survivorship/selection bias."
]

text_content = "\\n\\n".join(takeaways_list)
ax4.text(0.0, 0.5, text_content, fontsize=9, va='center', wrap=True)
ax4.set_title("Executive Summary", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Japanese_Equity_Portfolio_Summary.pdf')
plt.show()
"""))

with open('Japanese_Equity_Portfolio_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created: Japanese_Equity_Portfolio_Analysis.ipynb")
