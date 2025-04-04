from backtesting import Backtest
from . import strategies
from . import constants
import numpy as np
import pandas as pd

def run_backtest(population, data_set, isShortOnly, emas_data_set = None):
    results_list = []
    
    for offspring in population:
        short_ma = offspring[0]
        long_ma = offspring[1]
        stop_loss_multiplier = offspring[2]
        position_size = offspring[3]
        
        if emas_data_set is not None:
            past_data_size = max(short_ma, long_ma)
            emas_data_set_last_relevant = emas_data_set[-past_data_size:]
            combined_data = pd.concat([emas_data_set_last_relevant, data_set])
            combined_data = combined_data.sort_index()
        
        if isShortOnly:
            ShortOnlyCrossOverStrategy = strategies.createShortOnlyCrossOverStrategy(short_ma, long_ma, stop_loss_multiplier, position_size)
            bt = Backtest(combined_data, ShortOnlyCrossOverStrategy, cash=constants.INITIAL_CAPITAL, commission=constants.COMMISSION)
        else:
            LongOnlyCrossOverStrategy = strategies.createLongOnlyCrossOverStrategy(short_ma, long_ma, stop_loss_multiplier, position_size)
            bt = Backtest(combined_data, LongOnlyCrossOverStrategy, cash=constants.INITIAL_CAPITAL, commission=constants.COMMISSION)
        
        result = bt.run()
        
        duration = result[constants.DURATION]
        if isinstance(duration, pd.Timedelta):
            duration = duration / pd.Timedelta(days=1)
            
        # trades = result[constants.TRADES]
            
        results_list.append((
            short_ma, long_ma, stop_loss_multiplier, position_size, 
            result[constants.RETURN], 
            result[constants.MAX_DRAWDOWN],
            result[constants.BUY_AND_HOLD],
            duration,
            result[constants.TRADES_NUMBER],
            result[constants.EXPOSURE_TIME],
            result[constants.SHARPE_RATIO],
        ))
    
    return np.array(results_list, dtype=constants.INDIVIDUAL_METRICS)
