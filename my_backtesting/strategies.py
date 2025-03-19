from backtesting import Strategy
from backtesting.lib import crossover
from . import moving_average

def crateCrossOverStrategy(short_period, long_period, stop_loss = 0.2, position_size = 1):
    class CrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(moving_average.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(moving_average.double_exponential_moving_average, self.data.Close, long_period)
        
        def next(self):
            cash_available = self.equity * position_size
            size = int(cash_available / self.data.Close[-1])

            if size < 1:
                return 
            
            if crossover(self.ema_short, self.ema_long):
                self.position.close()
                
                stop_loss_price = self.data.Close[-1] * (1 - stop_loss / 100)
                stop_loss_price = max(0.01, stop_loss_price)
                                                
                self.buy(size=size, sl=stop_loss_price)

            elif crossover(self.ema_long, self.ema_short):
                self.position.close()
                
                stop_loss_price = self.data.Close[-1] * (1 + stop_loss / 100)
                stop_loss_price = max(0.01, stop_loss_price)
                
                stop_loss_price = self.data.Close[-1] * (1 + stop_loss / 100)

                self.sell(size=size, sl=stop_loss_price) 
    
    return CrossOverStrategy

def createLongOnlyCrossOverStrategy(short_period, long_period, stop_loss=0.2, position_size=1):
    class LongOnlyCrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(moving_average.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(moving_average.double_exponential_moving_average, self.data.Close, long_period)
        
        def next(self):
            if self.position:
                return 

            cash_available = self.equity * position_size
            size = int(cash_available / self.data.Close[-1])

            if size < 1:
                return
            
            if crossover(self.ema_short, self.ema_long):
                stop_loss_price = self.data.Close[-1] * (1 - stop_loss / 100)
                stop_loss_price = max(0.01, stop_loss_price)

                self.buy(size=size, sl=stop_loss_price)

    return LongOnlyCrossOverStrategy
