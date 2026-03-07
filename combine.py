import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# 1. Define assets and download data
# -----------------------------

assets = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]

data = yf.download(assets, start="2018-01-01")["Close"]

returns = data.pct_change().dropna()

# -----------------------------
# 2. Portfolio statistics
# -----------------------------

def portfolio_performance(weights, mean_returns, cov_matrix):
    
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return portfolio_return, portfolio_volatility

# -----------------------------
# 3. Sharpe ratio function
# -----------------------------

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    
    sharpe = (p_return - risk_free_rate) / p_volatility
    
    return -sharpe

# -----------------------------
# 4. Optimization function
# -----------------------------

def optimize_portfolio(mean_returns, cov_matrix):

    num_assets = len(mean_returns)

    args = (mean_returns, cov_matrix)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0,1) for asset in range(num_assets))

    init_guess = num_assets * [1./num_assets]

    result = minimize(negative_sharpe,
                      init_guess,
                      args=args,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    return result

# -----------------------------
# 5. Calculate optimal portfolio
# -----------------------------

mean_returns = returns.mean()
cov_matrix = returns.cov()

optimal = optimize_portfolio(mean_returns, cov_matrix)

weights = optimal.x

# -----------------------------
# 6. Print optimal weights
# -----------------------------

print("Optimal Portfolio Allocation\n")

for i, asset in enumerate(assets):
    print(asset, ":", round(weights[i]*100,2), "%")

opt_return, opt_volatility = portfolio_performance(weights, mean_returns, cov_matrix)

print("\nExpected Annual Return:", round(opt_return*100,2), "%")
print("Expected Volatility:", round(opt_volatility*100,2), "%")

# -----------------------------
# 7. Efficient frontier simulation
# -----------------------------

num_portfolios = 5000

results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):

    w = np.random.random(len(assets))
    w /= np.sum(w)

    p_return, p_vol = portfolio_performance(w, mean_returns, cov_matrix)

    results[0,i] = p_vol
    results[1,i] = p_return
    results[2,i] = p_return / p_vol

# -----------------------------
# 8. Backtest optimized portfolio
# -----------------------------

portfolio_returns = returns.dot(weights)

cumulative_returns = (1 + portfolio_returns).cumprod()

benchmark = (1 + returns["SPY"]).cumprod()

# -----------------------------
# 9. Visualization
# -----------------------------

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.scatter(results[0,:], results[1,:],
            c=results[2,:],
            cmap="viridis",
            marker="o")

plt.colorbar(label="Sharpe Ratio")

plt.scatter(opt_volatility, opt_return,
            c="red",
            marker="*",
            s=300,
            label="Optimal Portfolio")

plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.legend()

# Backtest plot

plt.subplot(1,2,2)

plt.plot(cumulative_returns, label="Optimized Portfolio")
plt.plot(benchmark, label="SPY Benchmark")

plt.title("Portfolio Backtest")
plt.xlabel("Date")
plt.ylabel("Growth of $1 Investment")

plt.legend()

plt.tight_layout()

plt.show()
