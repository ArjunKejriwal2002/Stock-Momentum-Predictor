# Stock-Momentum-Predictor
Optimized ensemble ML model to predict a stock's short-term momentum with the goal of making trade execution algorithms more opportunistic

## Goal
Predict the following:
1) Direction of next-period stock move (Up/Down)
2) Extent of move ranked 1–5 (5 = largest expected magnitude)

## Data
1-minute aggregated stock price data over last 200 days (limitation of yfinance library) for 6 symbols of varying market cap and liquidity
-- more rigorous testing with tick-level price data over larger time horizon is done in an internal company project 

## Prediction Horizon
Predict the direction and extent of move for the following 10-minute period

## Models (Ensemble)
Note that based on research and testing, these four models offered the best combination of high accuracy and low latency:
1) Gradient Boosting (XGBoost-style via GradientBoostingRegressor)
2) Random Forest Regressor
3) Elastic Net Regressor
4) Multi-layer Perceptron Regressor
* predictions are ensembled via averaging.

## Model Output:
1) Binary classification for momentum direction (1 for Up and 0 for Down)
2) Quantile-based ranking (1–5) for extent of move (5 being the largest extent)

## Key Design Choices:
- Momentum features leveraged include returns, moving averages, volatility, and RSI (Relative Strength Index)
- 80/20 train test time-series split (with no lookahead bias)
- Trained across all symbols jointly to improve generalization
- Hyperparameter optimization to obtain best version of each model
