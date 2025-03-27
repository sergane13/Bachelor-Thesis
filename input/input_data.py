from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CHUNK_SIZE = 100

def split_into_chunks(data, chunk_size):
    num_chunks = len(data) // chunk_size
    chunks = [data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    if len(data) % chunk_size != 0:
        chunks.append(data.iloc[num_chunks * chunk_size:])

    return chunks

spx_path = './data/S&P500.csv'
spx = read_csv(spx_path)
spx['Date'] = pd.to_datetime(spx['Date'])
spx.set_index('Date', inplace=True)
spx.rename(columns={'value': 'Close'}, inplace=True)
spx['Open'] = spx['Close']
spx['High'] = spx['Close']
spx['Low'] = spx['Close']
spx['Volume'] = 0
SPX = spx[['Open', 'High', 'Low', 'Close', 'Volume']]
SPX_CHUNKS = split_into_chunks(SPX, CHUNK_SIZE)

btc = './data/BTC_2018_2025_1h.csv'
btc = read_csv(btc)
btc = btc.iloc[::-1]
btc['date'] = pd.to_datetime(btc['date'])
btc.set_index('date', inplace=True)
btc.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'Volume USD': 'Volume'
}, inplace=True)
BTC = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
# BTC = BTC[10000:20000]
BTC_CHUNKS = split_into_chunks(BTC, CHUNK_SIZE)


hot = './data/HOTUSDT_2019_2025_1h.csv'
hot = read_csv(hot)
hot = hot.iloc[::-1]
hot['date'] = pd.to_datetime(hot['datetime'])
hot.set_index('datetime', inplace=True)
hot.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
HOT = hot[['Open', 'High', 'Low', 'Close', 'Volume']]
HOT_CHUNKS = split_into_chunks(HOT, CHUNK_SIZE)


btc_5m = './data/BTC_2011_2021_5m.csv'
btc_5m = read_csv(btc_5m)
btc_5m['Timestamp'] = pd.to_datetime(btc_5m['Timestamp'], unit='ms')
btc_5m.set_index('Timestamp', inplace=True)
btc_5m.rename(columns={
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume'
}, inplace=True)
BTC_5M = btc_5m[['Open', 'High', 'Low', 'Close', 'Volume']]
BTC_5M = BTC_5M[400_000: 600_000]
BTC_5M_CHUNKS = split_into_chunks(BTC_5M, CHUNK_SIZE)


tqqq = './data/TQQQ_10Years.csv'
tqqq = read_csv(tqqq)
tqqq = tqqq.iloc[::-1]
tqqq['date'] = pd.to_datetime(tqqq['Date'])
tqqq.set_index('date', inplace=True)
tqqq['Open'] = tqqq['Close']
tqqq['High'] = tqqq['Close']
tqqq['Low'] = tqqq['Close']
tqqq['Volume'] = 0
TQQQ = tqqq[['Open', 'High', 'Low', 'Close', 'Volume']]
TQQQ_CHUNKS = split_into_chunks(TQQQ, CHUNK_SIZE)

# plt.figure(figsize=(14, 6))

# for i, chunk in enumerate(BTC_CHUNKS):
#     plt.plot(chunk.index, chunk['Close'], label=f'Chunk {i+1}')

# plt.title('BTC Close Price - All Chunks')
# plt.xlabel('Date')
# plt.ylabel('Price [$]')
# plt.legend(loc='upper left', fontsize='small', ncol=2)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# === Trend Slope & R² calculation ===
# def compute_trend_features(series, window):
#     slopes, r2s = [], []
#     x = np.arange(window).reshape(-1, 1)

#     for i in range(len(series)):
#         if i < window:
#             slopes.append(np.nan)
#             r2s.append(np.nan)
#             continue
#         y = series[i-window:i].values.reshape(-1, 1)
#         model = LinearRegression().fit(x, y)
#         slopes.append(model.coef_[0][0])
#         r2s.append(model.score(x, y))

#     return np.array(slopes), np.array(r2s)

# df['Trend_Slope'], df['R2'] = compute_trend_features(df['Close'], WINDOW)

# # === Normalized ATR ===
# def compute_normalized_atr(df, window):
#     high_low = df['High'] - df['Low']
#     high_close = np.abs(df['High'] - df['Close'].shift())
#     low_close = np.abs(df['Low'] - df['Close'].shift())
#     tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#     atr = tr.rolling(window=window).mean()
#     normalized_atr = atr / df['Close']
#     return normalized_atr

# df['Normalized_ATR'] = compute_normalized_atr(df, WINDOW)

# # === Plotting ===
# plt.figure(figsize=(14, 10))

# plt.subplot(3, 1, 1)
# plt.plot(df.index, df['Trend_Slope'])
# plt.title('Trend Slope')

# plt.subplot(3, 1, 2)
# plt.plot(df.index, df['R2'])
# plt.title('R² of Trend Line')

# plt.subplot(3, 1, 3)
# plt.plot(df.index, df['Normalized_ATR'])
# plt.title('Normalized ATR')

# plt.tight_layout()
# plt.show()
