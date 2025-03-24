from backtesting import Backtest
from . import strategies
from . import constant
from shared import var_types
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import ta

def get_regime_features(close, high, low, window=30):
    X = np.arange(len(close[-window:])).reshape(-1, 1)
    y = close[-window:].values.reshape(-1, 1)
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

def runBackTest(population, data_set, isShortOnly, emas_data_set = None):
    results_list = []
    
    for offspring in population:
        short_ma = offspring[0]
        long_ma = offspring[1]
        stop_loss_multiplier = offspring[2]
        position_size = offspring[3]
        
        if emas_data_set is not None:
            past_data_size = max(short_ma, long_ma)
            emas_data_set = emas_data_set[-past_data_size:]
            data_set = pd.concat([emas_data_set, data_set])
            data_set = data_set.sort_index()

        if isShortOnly:
            ShortOnlyCrossOverStrategy = strategies.createShortOnlyCrossOverStrategy(short_ma, long_ma, stop_loss_multiplier, position_size)
            bt = Backtest(data_set, ShortOnlyCrossOverStrategy, cash=constant.INITIAL_CAPITAL, commission=constant.COMMISSION)
        else:
            LongOnlyCrossOverStrategy = strategies.createLongOnlyCrossOverStrategy(short_ma, long_ma, stop_loss_multiplier, position_size)
            bt = Backtest(data_set, LongOnlyCrossOverStrategy, cash=constant.INITIAL_CAPITAL, commission=constant.COMMISSION)
        
        result = bt.run()
        
        duration = result[constant.DURATION]
        if isinstance(duration, pd.Timedelta):
            duration = duration / pd.Timedelta(days=1)
            
        results_list.append((
            short_ma, long_ma, stop_loss_multiplier, position_size, 
            result[constant.RETURN], 
            result[constant.MAX_DRAWDOWN],
            result[constant.BUY_AND_HOLD],
            duration,
            result[constant.TRADES_NUMBER],
            result[constant.EXPOSURE_TIME]
        ))
    
    dtype = np.dtype([
        (var_types.SHORT_MA, np.int32),
        (var_types.LONG_MA, np.int32),
        (var_types.STOP_LOSS_MULTIPLIER, np.float32),
        (var_types.POSITION_SIZE, np.float32),
        (var_types.RETURN, np.float32),
        (var_types.MAX_DRAWDOWN, np.float32),
        (var_types.BUY_AND_HOLD, np.float32),
        (var_types.DURATION, np.float32),
        (var_types.TRADES_NUMBER, np.float32),
        (var_types.EXPOSURE_TIME, np.float32),
    ])
    
    return np.array(results_list, dtype=dtype)

