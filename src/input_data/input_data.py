import pandas as pd

def split_into_chunks(data, chunk_size):
    num_chunks = len(data) // chunk_size
    chunks = [data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    if len(data) % chunk_size != 0:
        chunks.append(data.iloc[num_chunks * chunk_size:])

    return chunks

def load_market_data(
    path,
    datetime_col,
    reverse=False,
    index_is_datetime=True,
    chunk_size=None
):
    df = pd.read_csv(path)

    if reverse:
        df = df.iloc[::-1]

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    if index_is_datetime:
        df.set_index(datetime_col, inplace=True)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    if chunk_size:
        chunks = split_into_chunks(df, chunk_size)
        return df, chunks

    return df


SPX, SPX_CHUNKS = load_market_data(
    path='./data/S&P500.csv',
    datetime_col='Date',
    reverse=False,
    chunk_size=100
)

BTC, BTC_CHUNKS = load_market_data(
    path='./data/BTC_2018_2025_1h.csv',
    datetime_col='date',
    reverse=True,
    chunk_size=100
)

HOT, HOT_CHUNKS = load_market_data(
    path='./data/HOTUSDT_2019_2025_1h.csv',
    datetime_col='datetime',
    reverse=True,
    chunk_size=100
)

BTC_5M, BTC_5M_CHUNKS = load_market_data(
    path='./data/BTC_2011_2021_5m.csv',
    datetime_col='Timestamp',
    reverse=False,
    chunk_size=100
)

TQQQ, TQQQ_CHUNKS = load_market_data(
    path='./data/TQQQ_10Years.csv',
    datetime_col='Date',
    reverse=True,
    chunk_size=100
)
