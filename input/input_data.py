from pandas import read_csv
import pandas as pd

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

# This means 50 chunks for SPX
SPX_CHUNKS = split_into_chunks(SPX, CHUNK_SIZE)

# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data?resource=download
# btc 1 ian 2012 - 30 dec 2024

# https://www.cryptodatadownload.com/data/bitstamp/

btc = './data/BTC2018_2025.csv'
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

# For bitcoin data that would be 116 chunks (58k / 500)
BTC_CHUNKS = split_into_chunks(BTC, CHUNK_SIZE)


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
