import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from enum import Enum

# 1. Trend Direciton Linear Regresion
# 2. Clarity R2
# 3. Volatility Normalized ATR/price
def get_regime_features(data, window=30):
    close = data['Close'].dropna()
    high = data['High'].dropna()
    low = data['Low'].dropna()

    y = close[-window:].values.reshape(-1, 1)
    X = np.arange(window).reshape(-1, 1)
    linreg = LinearRegression().fit(X, y)
    slope = linreg.coef_[0][0]

    r_squared = linreg.score(X, y)

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift()).abs(),
        "lc": (low - close.shift()).abs()
    }).max(axis=1)

    atr = tr.rolling(window).mean().iloc[-1]
    normalized_atr = atr / close.iloc[-1]

    return (slope, r_squared, normalized_atr)

def get_regime_curve(data, window=30, target_points=1):
    regimes = []
    stride = max(1, (len(data) - window + 1) // target_points)

    for i in range(0, len(data) - window + 1, stride):
        slice_data = data.iloc[i:i+window]
        slope, r2, atr_ratio = get_regime_features(slice_data)
        regimes.append((slope, r2, atr_ratio))

    return regimes

