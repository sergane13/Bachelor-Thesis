from backtesting import Strategy
from backtesting.lib import crossover
from . import indicators

def createLongOnlyCrossOverStrategy(short_period, long_period, stop_loss_multiplier=0.2, position_size=1):
    class LongOnlyCrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(indicators.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(indicators.double_exponential_moving_average, self.data.Close, long_period)
            self.atr = self.I(indicators.average_true_range, self.data.High, self.data.Low, self.data.Close, 14)
            self.initialized = False
            self.highest_price = None 

        def next(self):
            min_bars = max(short_period, long_period, 14)
            if len(self.data) < min_bars:
                return

            close_price = self.data.Close[-1]
            atr_value = self.atr[-1]

            if self.position:
                if self.highest_price is None:
                    self.highest_price = close_price
                else:
                    self.highest_price = max(self.highest_price, close_price)

                trailing_stop = self.highest_price - (atr_value * stop_loss_multiplier)

                if close_price < trailing_stop:
                    self.position.close()
                    self.highest_price = None
                    return

                if crossover(self.ema_long, self.ema_short):
                    self.position.close()
                    self.highest_price = None
                    return

                return 

            cash_available = self.equity * position_size
            size = int(cash_available / close_price)
            if size < 1:
                return

            should_enter = False
            if not self.initialized:
                if self.ema_short[-1] > self.ema_long[-1]:
                    should_enter = True
                self.initialized = True
            else:
                if crossover(self.ema_short, self.ema_long):
                    should_enter = True

            if should_enter:
                self.buy(size=size)
                self.highest_price = close_price

    return LongOnlyCrossOverStrategy


def createShortOnlyCrossOverStrategy(short_period, long_period, stop_loss_multiplier=0.2, position_size=1):
    class ShortOnlyCrossOverStrategy(Strategy):
        def init(self):
            self.ema_short = self.I(indicators.double_exponential_moving_average, self.data.Close, short_period)
            self.ema_long = self.I(indicators.double_exponential_moving_average, self.data.Close, long_period)
            self.atr = self.I(indicators.average_true_range, self.data.High, self.data.Low, self.data.Close, 14)
            self.initialized = False
            self.lowest_price = None
            
        def next(self):
            min_bars = max(short_period, long_period, 14)
            if len(self.data) < min_bars:
                return

            close_price = self.data.Close[-1]
            atr_value = self.atr[-1]

            if self.position:
                if self.lowest_price is None:
                    self.lowest_price = close_price
                else:
                    self.lowest_price = min(self.lowest_price, close_price)

                trailing_stop = self.lowest_price + (atr_value * stop_loss_multiplier)

                if close_price > trailing_stop:
                    self.position.close()
                    self.lowest_price = None
                    return

                if crossover(self.ema_short, self.ema_long):
                    self.position.close()
                    self.lowest_price = None
                    return

                return

            cash_available = self.equity * position_size
            size = int(cash_available / close_price)
            if size < 1:
                return

            should_enter = False
            if not self.initialized:
                if self.ema_long[-1] > self.ema_short[-1]:
                    should_enter = True
                self.initialized = True
            else:
                if crossover(self.ema_long, self.ema_short):
                    should_enter = True

            if should_enter:
                self.sell(size=size)
                self.lowest_price = close_price
                
    return ShortOnlyCrossOverStrategy
