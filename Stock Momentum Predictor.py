#!/usr/bin/env python
# coding: utf-8


### Stock Momentum Predictor

## Overview: Uses an ensemble ML model to predict a stock's momentum (direction and extent of move) to benefit any trading strategy/algorithm 

## Goal: Predict the following:
# 1) Direction of next-period stock move (Up/Down)
# 2) Extent of move ranked 1–5 (5 = largest expected magnitude)

## Data: 1-minute aggregated stock price data over last 200 days (limitation of yfinance library) for 6 symbols of varying market cap and liquidity
# -- more rigorous testing with tick-level price data over larger time horizon is done in internal company project 

## Prediction Horizon: predict the direction and extent of move for the following 10-minute period

## Models (Ensemble):
# -- note that based on research and testing, these four models offered the best combination of high accuracy and low latency
# 1) Gradient Boosting (XGBoost-style via GradientBoostingRegressor)
# 2) Random Forest Regressor
# 3) Elastic Net Regressor
# 4) Multi-layer Perceptron Regressor
# -- predictions are ensembled via averaging.

## Model Output:
# 1) Binary classification for momentum direction (1 for Up and 0 for Down)
# 2) Quantile-based ranking (1–5) for extent of move (5 being the largest extent)

## Key Design Choices:
# - Momentum features leveraged include returns, moving averages, volatility, and RSI (Relative Strength Index)
# - 80/20 train test time-series split (with no lookahead bias)
# - Trained across all symbols jointly to improve generalization
# - Hyperparameter optimization

# ------ Code begins below ------


# install the yfinance library
get_ipython().run_line_magic('pip', 'install yfinance')

# importing numpy, pandas, and yfinance libraries
import yfinance as yf
import numpy as np
import pandas as pd

# importing sklearn modules
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


# Configuration

# choosing 6 symbols with varying market cap and liquidity
SYMBOLS = [
    "AAPL",  # Large cap, liquid
    "AVTR",  # Mid cap, moderately liquid
    "GEL",   # Small cap, illiquid
    "SPY",   # Market proxy
    "BBLU",  # Large-cap ETF 
    "IWM",   # Small-cap ETF
]

INTERVAL = "1m"  # 1-minute aggregated price data
LOOKBACK_PERIOD = "7d"  # lookback limitation of yfinance
PRED_HORIZON = 60  # 60-minute prediction horizon to capture sustained move


# Feature Engineering

def compute_features(df):
    df = df.copy()

    df["ret_1m"] = df["Close"].pct_change()  # 1-min return
    df["ret_5m"] = df["Close"].pct_change(5)  # 5-min return
    df["ret_10m"] = df["Close"].pct_change(10)  # 10-min return
    df["ret_30m"] = df["Close"].pct_change(30)  # 30-min return
    df["ret_60m"] = df["Close"].pct_change(60)  # 60-min return

    df["ma_5"] = df["Close"].rolling(5).mean()  # 5-min moving average
    df["ma_10"] = df["Close"].rolling(10).mean()  # 10-min moving average
    df["ma_30"] = df["Close"].rolling(30).mean()  # 30-min moving average
    df["ma_60"] = df["Close"].rolling(60).mean()  # 60-min moving average

    df["vol_5"] = df["ret_1m"].rolling(5).std()  # 5-min volatility using standard deviation of returns
    df["vol_10"] = df["ret_1m"].rolling(10).std()  # 10-min volatiltiy using standard deviation of returns
    df["vol_30"] = df["ret_1m"].rolling(30).std()  # 30-min volatility using standard deviation of returns
    df["vol_60"] = df["ret_1m"].rolling(60).std()  # 60-min volatiltiy using standard deviation of returns

    # Calculating RSI for a 60-minute window
    delta = df["Close"].diff()  # calculate running diff
    gain = delta.clip(lower=0)  # obtain array of gains
    loss = -delta.clip(upper=0)  # obtain array of losses
    rs = gain.rolling(60).mean() / loss.rolling(60).mean()  # calculate relative strength
    df["rsi_60"] = 100 - (100 / (1 + rs))  # plug RS into formula for RSI

    return df


# Label Construction

def create_labels(df):
    df = df.copy()

    df["future_ret_60m"] = (
        df["Close"].shift(-PRED_HORIZON) / df["Close"] - 1
    )

    df["direction"] = (df["future_ret_60m"] > 0).astype(int)

    df["extent_rank"] = pd.qcut(
        df["future_ret_60m"].abs(),
        q=5,
        labels=[1, 2, 3, 4, 5]
    )

    return df


# In[83]:


# Loading and preparing training/testing data

def load_data(symbol):
    df = yf.download(   # pulling 1-minute data for last 7 days from yfinance
            symbol,
            interval=INTERVAL,
            period=LOOKBACK_PERIOD,
            progress=False
        )[["Close"]]

    df["symbol"] = symbol
    df = compute_features(df)  # feature engineering
    df = create_labels(df)  # label construction
    df = df.dropna()
    
    return df


# Preparing the training and testing data
# Applying train/test split without breaking time-series continuity
# -- note in-built sklearn train_test_split method is not used since that shuffles the data and doesn't maintain time series continuity

def train_test_split(data, split_ratio=0.8):
    if data.empty:
        raise ValueError("No data available after feature engineering. Check data loading and NA handling.")

    data = data.sort_index()
    split_idx = int(len(data) * split_ratio)  # Obtain 80th percentile date/time

    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    features = [
        "ret_5m", "ret_10m", "ret_30m", "ret_60m",
        "ma_5", "ma_10", "ma_30", "ma_60",
        "vol_5", "vol_10", "vol_30", "vol_60", "rsi_60"
    ]  # identify inputs to the model

    X_train = train[features]
    y_train = train["future_ret_60m"]

    X_test = test[features]
    y_test = test[["direction","extent_rank"]]

    return X_train, X_test, y_train, y_test, test


# Preparing for hyperparameter optimization of ML models

tscv = TimeSeriesSplit(n_splits=5)

# possible hyperparams for Gradient Boost Regressor
gbr_param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.07, 0.1, 0.15],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.8, 0.85, 0.9, 1.0]
}

# possible hyperparams for Random Forest Regressor
rf_param_grid = {
    "n_estimators": [200, 300, 400, 500, 600],
    "max_depth": [6, 7, 8, 9, 10],
    "min_samples_leaf": [1, 3, 5, 7, 10],
    "max_features": ["sqrt", 0.5]
}

# possible hyperparams for Elastic Net Regressor
enet_param_grid = {
    "enet__alpha": [1e-4, 5e-3, 1e-3, 5e-2, 1e-2],
    "enet__l1_ratio": [0.2, 0.35, 0.5, 0.65, 0.8]
}

# possible hyperparams for Multi-layer Perceptron Regressor
mlp_param_grid = {
    "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
    "mlp__alpha": [1e-4, 5e-4, 1e-3],
    "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3]
}


# Tuning model based on MSE

def tune_model(model, param_grid, X_train, y_train):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    return search.best_estimator_


# Defining and tuning all 4 ML models

def tune_all_models(X_train, y_train):
    tuned_models = {}

    # Gradient Boost Regressor
    gbr = GradientBoostingRegressor(random_state=42)
    tuned_models["gbr"] = tune_model(gbr, gbr_param_grid, X_train, y_train)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    tuned_models["rf"] = tune_model(rf, rf_param_grid, X_train, y_train)

    # Elastic Net Regressor
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(max_iter=5000))
    ])
    tuned_models["enet"] = tune_model(enet, enet_param_grid, X_train, y_train)

    # Multi-layer Perceptron Regressor
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ])
    tuned_models["mlp"] = tune_model(mlp, mlp_param_grid, X_train, y_train)

    return tuned_models


# Training and Ensembling

def train_and_predict(X_train, y_train, X_test):
    models = tune_all_models(X_train, y_train)  # building, tuning, and training the four ML models
    print(models)
    preds = []

    for model in models.values():
        preds.append(model.predict(X_test))  # working trained model on test data

    ensemble_pred = np.mean(preds, axis=0)  # averaging predictions from the three models
    return ensemble_pred


# Defining main method for stock momentum predictor

def main():
    dir_accuracy_scores = []
    extent_accuracy_scores = []
    extent_disc_scores = []
    
    for symbol in SYMBOLS:
        data = load_data(symbol)  # obtain enriched data for symbol
        X_train, X_test, y_train, y_test, test = train_test_split(data)  # split data into train and test data

        ensemble_returns = train_and_predict(X_train, y_train, X_test)  # obtain predictions
        direction_preds = (ensemble_returns > 0).astype(int)  # convert to direction predictions
        extent_preds = pd.qcut(np.abs(ensemble_returns), q=5, labels=[1, 2, 3, 4, 5])
        
        dir_accuracy_scores.append(accuracy_score(y_test["direction"], direction_preds))  # add accuracy score to direction list
        extent_accuracy_scores.append(accuracy_score(y_test["extent_rank"], extent_preds))  # add accuracy score to extent accuracy list
        extent_disc_scores.append(np.mean(abs(y_test["extent_rank"].astype(int) - extent_preds.astype(int)))) # add accuracy score to extent disc list
        
    print("Direction Accuracy:",
          np.mean(dir_accuracy_scores))  # compute mean direction accuracy score across symbols
    
    print("Extent Accuracy:",
          np.mean(extent_accuracy_scores))  # compute mean extent accuracy score across symbols
    
    print("Extent Discrepancy:",
          np.mean(extent_disc_scores))  # compute mean extent discrepancy score across symbols
    

# Running the stock momentum predictor for the 6 symbols

if __name__ == "__main__":
    main()


## Results:

'''
Directional Accuracy = 67% (beats random chance of 50%)
Extent Accuracy = 34% (beats random chance of 20%)
Extent Discrepancy = 0.8 (beats random chance of 1.2)  -- this represents the average absolute diff in the predicted and actual quantile-based ranking of momentum-extent

* though the numbers above beat chance, we want higher accuracy to be able to build better performing trade execution algorithms
'''


## Limitations:

'''
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

