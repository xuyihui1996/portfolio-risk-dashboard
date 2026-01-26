# Japanese Equity Portfolio Analysis Project

## Overview
This project simulates and analyzes an equal-weight Japanese equity portfolio comprising 10 major stocks and benchmarks against the Nikkei 225 index. It demonstrates portfolio construction, monthly rebalancing, performance metric calculation, and reporting using Python.

## Project Structure
```
.
├── Japanese_Equity_Portfolio_Analysis.ipynb  # Main analysis notebook
├── requirements.txt                          # Python dependencies
├── Portfolio_Summary.pdf                     # Generated one-page summary
├── data/                                     # Data storage (if cached)
├── figures/                                  # Output figures
└── README.md                                 # This file
```

## Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebook:**
    ```bash
    jupyter notebook Japanese_Equity_Portfolio_Analysis.ipynb
    ```
    - Execute all cells to fetch data, simulate the portfolio, and generate the summary PDF.

## Methodology
-   **Universe:** 10 Large-cap Japanese stocks + Nikkei 225 ETF (1321.T).
-   **Period:** Jan 2018 - Dec 2023.
-   **Strategy:** Monthly equal-weight rebalancing.
-   **Costs:** 0.1% per transaction.
-   **Benchmark:** Nikkei 225 Index (^N225).

## Key Metrics Calculated
-   CAGR (Compound Annual Growth Rate)
-   Annualized Volatility
-   Sharpe Ratio (Risk-free rate = 0)
-   Maximum Drawdown

## Results
The analysis produces a `Japanese_Equity_Portfolio_Summary.pdf` file containing:
-   Equity Curve vs Benchmark
-   Drawdown Chart
-   Metrics Table
-   Key Takeaways with Attribution Analysis
