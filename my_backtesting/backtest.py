from backtesting import Backtest
from . import strategies
from . import constant
from shared import var_types
import numpy as np

def runBackTest(population, data_set):
    results_list = []
    
    for offspring in population:
        short_ma = offspring[0]
        long_ma = offspring[1]
        stop_loss = offspring[2]
        position_size = offspring[3]

        CrossOverStrategy = strategies.crateCrossOverStrategy(short_ma, long_ma, stop_loss, position_size)
        bt = Backtest(data_set, CrossOverStrategy, cash=constant.INITIAL_CAPITAL, commission=constant.COMMISSION)
        result = bt.run()
        results_list.append((
            short_ma, long_ma, stop_loss, position_size, 
            result[constant.RETURN], 
            result[constant.MAX_DRAWDOWN]
        ))
    
    dtype = np.dtype([
        (var_types.SHORT_MA, np.int32),
        (var_types.LONG_MA, np.int32),
        (var_types.STOP_LOSS, np.float32),
        (var_types.POSITION_SIZE, np.float32),
        (var_types.RETURN, np.float32),
        (var_types.MAX_DRAWDOWN, np.float32)
    ])
    
    return np.array(results_list, dtype=dtype)

def calculate_buy_and_hold_metrics(price_data):
    if 'Close' not in price_data.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    initial_price = price_data['Close'].iloc[0]
    final_price = price_data['Close'].iloc[-1]
    total_return = (final_price - initial_price) / initial_price

    cumulative_max = price_data['Close'].cummax()
    drawdown = (price_data['Close'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return total_return, max_drawdown