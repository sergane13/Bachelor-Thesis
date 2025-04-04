EXPOSURE = "Exposure Time [%]"
RETURN = "Return [%]"
SHARPE_RATIO = "Sharpe Ratio"
MAX_DRAWDOWN = "Max. Drawdown [%]"
TRADES_NUMBER = "# Trades"
EXPOSURE_TIME = "Exposure Time [%]"
DURATION = "Duration"
BUY_AND_HOLD = "Buy & Hold Return [%]"
TRADES = "_trades"

# ----------- Trading ------------- #

INITIAL_CAPITAL = 200_000
COMMISSION = 0.0004 # 0.04 %
# 0.0180%/0.0450% MAKER | TAKER fees ---> USDT BNB
# 0.0000%/0.0400% MAKER | TAKER fees ---> USDC
# 0.0000%/0.0360% MAKER | TAKER fees ---> USDC BNB

# ----------- Types ------------- #

import numpy as np
from src import var_types

INDIVIDUAL_METRICS = np.dtype([
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
    (var_types.SHARPE_RATIO, np.float32),
])