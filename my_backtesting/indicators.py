import pandas as pd
import numpy as np
from regimes import regime

def average_true_range(high, low, close, period=14):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr

def exponential_moving_average(series, period):
    series = pd.Series(series)
    return series.ewm(span=period, adjust=False).mean()

def weighted_moving_average(series, period):
    weights = np.arange(1, period + 1)
    return pd.Series(series).rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def double_exponential_moving_average(series, period):
    series = pd.Series(series)
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2

def r_squared(prices, window):
    prices = np.asarray(prices)
    r2 = np.full_like(prices, fill_value=np.nan, dtype=float)

    x = np.arange(window)

    for i in range(window, len(prices)):
        y = prices[i - window:i]
        if np.any(np.isnan(y)):
            continue

        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return r2

