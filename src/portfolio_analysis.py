
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from datetime import datetime

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# --- Configuration ---
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
    "1321.T"  # Nikkei 225 ETF (as part of portfolio)
]
BENCHMARK_TICKER = "^N225" # Nikkei 225 Index
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
INITIAL_CAPITAL = 10_000_000 # 10 Million JPY
TRANSACTION_COST = 0.001 # 0.1%

OUTPUT_DIR = "figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Data Fetching ---
def fetch_data(tickers, start, end):
    print(f"Fetching data for {len(tickers)} assets and benchmark...")
    # Fetch Portfolio Attributes
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex if multiple tickers. We want 'Close' (which is Adj Close due to auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
         # Handle potential different yfinance versions. v0.2+ returns (Price, Ticker) structure
        try:
            prices = data["Close"]
        except KeyError:
            # Fallback if "Close" is not top level, though with auto_adjust=True it usually is "Close"
            prices = data
    else:
        prices = data
    
    # Fetch Benchmark
    bench = yf.download(BENCHMARK_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(bench.columns, pd.MultiIndex):
         bench_prices = bench["Close"]
    else:
         bench_prices = bench["Close"] if "Close" in bench.columns else bench
    
    # Handle single column benchmark being Series or DataFrame
    if isinstance(bench_prices, pd.DataFrame):
        bench_prices = bench_prices.iloc[:, 0]

    # Clean data
    prices.dropna(how='all', inplace=True)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    
    bench_prices = bench_prices.reindex(prices.index)
    bench_prices.ffill(inplace=True)
    
    return prices, bench_prices

# --- Simulation Logic ---
def simulate_portfolio(prices):
    print("Simulating portfolio...")
    # Rebalancing dates: First trading day of each month
    # We can use resample('MS') to get start of months, then find nearest valid date (or first valid date)
    # Actually, simpler: just iterate and check if month specific logic applies?
    # Or use resample to find the index of first trading days.
    
    # Logic: Get unique Year-Months, then find the first index for each.
    monthly_starts = prices.groupby([prices.index.year, prices.index.month]).apply(lambda x: x.index[0])
    rebalance_dates = set(monthly_starts.values)
    
    cash = INITIAL_CAPITAL
    holdings = {t: 0 for t in TICKERS} # Shares held
    portfolio_history = []
    
    # Initial setup
    dates = prices.index
    
    # We need to track current weights dynamically
    # Start with all cash
    
    for date in dates:
        current_prices = prices.loc[date]
        
        # Calculate current portfolio value (pre-rebalance)
        portfolio_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
        
        is_rebalance_day = date in rebalance_dates
        # Special case: First day is always a rebalance day (Initial allocation)
        if date == dates[0]:
            is_rebalance_day = True
            
        if is_rebalance_day:
            target_per_asset = portfolio_val / len(TICKERS)
            total_valid_cost = 0
            
            # Simple rebalancing:
            # 1. Calculate diff for each asset
            # 2. Adjust partial cash logic?
            # To be precise with costs reducing buying power, typical approx is fine:
            # We pay cost from cash. if cash < 0 (unlikely with this logic unless severe drop + high cost), we have issue.
            # Let's iterate: sell overs, then buy unders.
            
            # 1. Sell Loop
            for t in TICKERS:
                current_holding_val = holdings[t] * current_prices[t]
                if current_holding_val > target_per_asset:
                    diff = current_holding_val - target_per_asset
                    # Sell 'diff' worth
                    amt_to_sell = diff
                    shares_to_sell = amt_to_sell / current_prices[t]
                    holdings[t] -= shares_to_sell
                    cash += amt_to_sell
                    cash -= amt_to_sell * TRANSACTION_COST
            
            # Re-eval cash and target? Strictly, costs reduce total equity.
            # But the 'target_per_asset' was aimed at pre-cost equity.
            # Let's simple-pass: Sell first implies we have cash. Then Buy.
            
            # 2. Buy Loop
            # Recalculate portfolio val after selling costs? 
            # Or just aim for the target calculated at start?
            # If we aim for initial target, we might run out of cash due to costs.
            # Let's update portfolio val
            portfolio_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
            target_per_asset = portfolio_val / len(TICKERS) # Approximate new target 
            
            for t in TICKERS:
                current_holding_val = holdings[t] * current_prices[t]
                if current_holding_val < target_per_asset:
                    diff = target_per_asset - current_holding_val
                    # Buy 'diff' worth? We might not have enough cash if we aren't careful?
                    # But since we just matched the others to target, and sum(target) ~ portfolio_val, it should match.
                    # Except for the cost of buying.
                    
                    cost_to_buy = diff * TRANSACTION_COST
                    amt_to_buy = diff
                    
                    if cash < (amt_to_buy + cost_to_buy):
                        # Adjust to max buyable
                        amt_to_buy = cash / (1 + TRANSACTION_COST)
                    
                    shares_to_buy = amt_to_buy / current_prices[t]
                    holdings[t] += shares_to_buy
                    cash -= amt_to_buy
                    cash -= amt_to_buy * TRANSACTION_COST
                    
        # Record End of Day status
        daily_val = cash + sum(holdings[t] * current_prices[t] for t in TICKERS)
        portfolio_history.append({'Date': date, 'PortfolioValue': daily_val})
        
    return pd.DataFrame(portfolio_history).set_index('Date')


# --- Analytics ---
def calculate_metrics(portfolio_series, benchmark_series):
    # Daily Returns
    p_ret = portfolio_series.pct_change().dropna()
    b_ret = benchmark_series.pct_change().dropna()
    
    # Align
    common_idx = p_ret.index.intersection(b_ret.index)
    p_ret = p_ret.loc[common_idx]
    b_ret = b_ret.loc[common_idx]
    
    # Cumulative
    p_cum = (1 + p_ret).cumprod()
    b_cum = (1 + b_ret).cumprod()
    
    # CAGR
    days = (p_ret.index[-1] - p_ret.index[0]).days
    years = days / 365.25
    p_cagr = (p_cum.iloc[-1])**(1/years) - 1
    b_cagr = (b_cum.iloc[-1])**(1/years) - 1
    
    # Volatility (Ann)
    p_vol = p_ret.std() * np.sqrt(252)
    b_vol = b_ret.std() * np.sqrt(252)
    
    # Sharpe (Rf=0, Arithmetic Mean)
    # User requested Standard Sharpe: Mean Annualized Return / Vol
    p_mean_ann = p_ret.mean() * 252
    b_mean_ann = b_ret.mean() * 252
    
    p_sharpe = p_mean_ann / p_vol if p_vol != 0 else 0
    b_sharpe = b_mean_ann / b_vol if b_vol != 0 else 0
    
    # Max Drawdown
    def get_dd(ts):
        peak = ts.cummax()
        dd = (ts - peak) / peak
        return dd.min(), dd
        
    p_mdd, p_dd_series = get_dd(p_cum)
    b_mdd, b_dd_series = get_dd(b_cum)
    
    return {
        'Returns': p_ret,
        'BenchmarkReturns': b_ret,
        'Cumulative': p_cum,
        'BenchmarkCumulative': b_cum,
        'Drawdown': p_dd_series,
        'Metrics': {
            'CAGR': p_cagr,
            'Vol': p_vol,
            'Sharpe': p_sharpe,
            'MDD': p_mdd,
            'Bench_CAGR': b_cagr,
            'Bench_Vol': b_vol,
            'Bench_Sharpe': b_sharpe,
            'Bench_MDD': b_mdd
        }
    }

def get_contributors_2023(prices):
    # Attribution for Year 2023
    # Get subset for 2023
    try:
        prices_2023 = prices.loc['2023']
        if prices_2023.empty:
            return {}
            
        p_start = prices_2023.iloc[0]
        p_end = prices_2023.iloc[-1]
        
        contribs = {}
        for t in TICKERS:
            # Simple return for the year
            # Approximate contribution: Weight (10%) * Asset Return
            # This is a simplified attribution.
            ret = (p_end[t] / p_start[t]) - 1
            contrib = ret * 0.10
            contribs[t] = contrib
            
        return contribs
    except KeyError:
        return {}

# --- Main Execution ---
def main():
    prices, benchmark = fetch_data(TICKERS, START_DATE, END_DATE)
    
    sim_df = simulate_portfolio(prices)
    
    res = calculate_metrics(sim_df['PortfolioValue'], benchmark)
    
    # Attribution (2023)
    contribs = get_contributors_2023(prices)
    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    top_contributors = sorted_contribs[:3]
    top_detractors = sorted_contribs[-3:] # Lowest returns (negative)
    
    # --- Visualization / One Pager ---
    fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape approx
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1])
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(res['Cumulative'].index, res['Cumulative'], label='Portfolio', color='#1f77b4', linewidth=2)
    # Updated Label
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
    
    table = ax3.table(cellText=metrics_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax3.set_title("Risk/Return Metrics", fontsize=12, fontweight='bold', pad=10)
    
    # 4. Takeaways & Attribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # Get text arrays
    top_c_str = ", ".join([f"{t[0]} ({t[1]:+.1%})" for t in top_contributors]).replace("%", "pp")
    top_d_str = ", ".join([f"{t[0]} ({t[1]:+.1%})" for t in top_detractors]).replace("%", "pp")
    
    # Updated Takeaways (PM Style)
    t_cagr = res['Metrics']['CAGR']
    b_cagr = res['Metrics']['Bench_CAGR']
    t_vol = res['Metrics']['Vol']
    b_vol = res['Metrics']['Bench_Vol']
    t_sharpe = res['Metrics']['Sharpe']
    b_sharpe = res['Metrics']['Bench_Sharpe']
    t_mdd = res['Metrics']['MDD']
    b_mdd = res['Metrics']['Bench_MDD']

    # Header Line
    header_info = "Universe: 10 Japan large-caps (equal weight), monthly rebalance, 0.1% transaction cost, Adj Close, period 2018–2023."

    # Takeaways list
    takeaways_list = [
        header_info,
        f"• Outperformance: Portfolio CAGR {t_cagr:.2%} vs {b_cagr:.2%} (Nikkei 225). Sharpe {t_sharpe:.2f} vs {b_sharpe:.2f} (ann. mean daily return / ann. vol).",
        f"• Risk Profile: Volatility {t_vol:.2%} vs {b_vol:.2%}; Max Drawdown {t_mdd:.2%} vs {b_mdd:.2%}, indicating similar equity risk with better downside control.",
        f"• Attribution (2023): Top contributors: {top_c_str}; Detractors: {top_d_str} (approx. contribution to portfolio return).",
        "• Note: Fixed illustrative stock basket; results may reflect survivorship/selection bias."
    ]
    
    # Manual wrapping for better control
    # Estimate width: A4 landscape half width approx 5 inches. 9pt font. ~50-60 chars.
    # User requested 80 chars and more spacing.
    formatted_lines = []
    
    # Header (No Indent)
    formatted_lines.append(textwrap.fill(takeaways_list[0], width=80))
    formatted_lines.append("") # Empty line gap
    
    # Bullets (Hanging Indent)
    bullet_wrapper = textwrap.TextWrapper(width=80, subsequent_indent='  ')
    for item in takeaways_list[1:]:
        formatted_lines.append(bullet_wrapper.fill(item))
        formatted_lines.append("") # Empty line gap between items
    
    text_content = "\n".join(formatted_lines)
    
    # Use top alignment, increased line spacing
    ax4.text(0.0, 1.0, text_content, fontsize=9, va='top', linespacing=1.5)
    ax4.set_title("Executive Summary", fontsize=11, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('Portfolio_Summary.pdf', dpi=300)
    print("One-page summary saved to Portfolio_Summary.pdf")

if __name__ == "__main__":
    main()
