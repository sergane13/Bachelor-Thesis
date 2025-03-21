import pandas as pd
import numpy as np

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
