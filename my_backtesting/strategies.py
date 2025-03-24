from backtesting import Strategy
from backtesting.lib import crossover
from . import moving_average

def createCrossOverStrategy(short_period, long_period, stop_loss_multiplier = 0.2, position_size = 1):
    class CrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(moving_average.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(moving_average.double_exponential_moving_average, self.data.Close, long_period)
            self.atr = self.I(moving_average.average_true_range, self.data.High, self.data.Low, self.data.Close, 14)
            
        def next(self):
            min_bars = max(short_period, long_period, 14)
            if len(self.data) < min_bars:
                return
            
            cash_available = self.equity * position_size
            size = int(cash_available / self.data.Close[-1])

            if size < 1:
                return 
            
            atr_value = self.atr[-1]
            
            if crossover(self.ema_short, self.ema_long):
                self.position.close()
                
                stop_loss_price = self.data.Close[-1] - (atr_value * stop_loss_multiplier)
                stop_loss_price = max(0.01, stop_loss_price)
                                                
                self.buy(size=size, sl=stop_loss_price)

            elif crossover(self.ema_long, self.ema_short):
                self.position.close()
                
                stop_loss_price = self.data.Close[-1] + (atr_value * stop_loss_multiplier)
                stop_loss_price = max(0.01, stop_loss_price)
                
                self.sell(size=size, sl=stop_loss_price) 
    
    return CrossOverStrategy


def createLongOnlyCrossOverStrategy(short_period, long_period, stop_loss_multiplier=0.2, position_size=1):
    class LongOnlyCrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(moving_average.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(moving_average.double_exponential_moving_average, self.data.Close, long_period)
            self.atr = self.I(moving_average.average_true_range, self.data.High, self.data.Low, self.data.Close, 14)

        def next(self):
            min_bars = max(short_period, long_period, 14)
            if len(self.data) < min_bars:
                return
            
            atr_value = self.atr[-1]

            if self.position and crossover(self.ema_long, self.ema_short):
                self.position.close()
                return
            
            if self.position:
                return 

            cash_available = self.equity * position_size
            size = int(cash_available / self.data.Close[-1])

            if size < 1:
                return
            
            if crossover(self.ema_short, self.ema_long):
                stop_loss_price = self.data.Close[-1] - (atr_value * stop_loss_multiplier)
                stop_loss_price = max(0.01, stop_loss_price)

                self.buy(size=size, sl=stop_loss_price)

    return LongOnlyCrossOverStrategy


def createShortOnlyCrossOverStrategy(short_period, long_period, stop_loss_multiplier=0.2, position_size=1):
    class ShortOnlyCrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(moving_average.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(moving_average.double_exponential_moving_average, self.data.Close, long_period)
            self.atr = self.I(moving_average.average_true_range, self.data.High, self.data.Low, self.data.Close, 14)

        def next(self):
            min_bars = max(short_period, long_period, 14)
            if len(self.data) < min_bars:
                return
            
            atr_value = self.atr[-1]

            if self.position and crossover(self.ema_short, self.ema_long):
                self.position.close()
                return
            
            if self.position:
                return 

            cash_available = self.equity * position_size
            size = int(cash_available / self.data.Close[-1])

            if size < 1:
                return
            
            if crossover(self.ema_long, self.ema_short):
                stop_loss_price = self.data.Close[-1] + (atr_value * stop_loss_multiplier)
                stop_loss_price = max(0.01, stop_loss_price)

                self.sell(size=size, sl=stop_loss_price)

    return ShortOnlyCrossOverStrategy
