import pandas as pd
import numpy as np

def simple_moving_average(series, period):
    return pd.Series(series).rolling(window=period).mean()

def exponential_moving_average(series, period):
    series = pd.Series(series)
    return series.ewm(span=period, adjust=False).mean()

def weighted_moving_average(series, period):
    weights = np.arange(1, period + 1)
    return pd.Series(series).rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hull_moving_average(series, period):
    wma_half = weighted_moving_average(series, period // 2)
    wma_full = weighted_moving_average(series, period)
    hma = weighted_moving_average(2 * wma_half - wma_full, int(np.sqrt(period)))
    return hma

def double_exponential_moving_average(series, period):
    series = pd.Series(series)
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2
