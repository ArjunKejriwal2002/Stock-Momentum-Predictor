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
* Predictions are ensembled via averaging.

## Model Output:
1) Binary classification for momentum direction (1 for Up and 0 for Down)
2) Quantile-based ranking (1–5) for extent of move (5 being the largest extent)

## Key Design Choices:
- Momentum features leveraged include returns, moving averages, volatility, and RSI (Relative Strength Index)
- 80/20 train test time-series split (with no lookahead bias)
- Trained across all symbols jointly to improve generalization
- Hyperparameter optimization to obtain best version of each model

## Results:
- Directional Accuracy = 67% (beats random chance of 50%)
- Extent Accuracy = 34% (beats random chance of 20%)
- Extent Discrepancy = 0.8 (beats random chance of 1.2)  -- this represents the average absolute diff in the predicted and actual quantile-based ranking of momentum-extent

Though the numbers above beat chance, we want higher accuracy to be able to build better-performing trade execution algorithms

## Limitations:
A number of reasons presented obstacles to building a highly predictive model:

1) Not a very big training/testing sample (only last 200 days) and aggregated data (1-minute)
-- fix should be to use larger sample (5+ years) and tick-level data (or second-level data)

2) Not enough models/model combinations tested and compared
-- fix should be to build a separate extensive script that automates testing combinations of different models (using forward selection or backward elimination)

3) Not enough hyperparameter combinations tested and compared
-- fix should be to optimize hyperparameters using a large set of possible values for each of the models in consideration before continuing with 2) above

4) Models not tested with the actual algorithms
-- fix should be to test the best model combinations from 2) in UAT environment and measure impact on algorithm performance

* The stock momentum prediction model built internally for the company aimed to eliminate these limitations and achieved better accuracy and performance
'''
