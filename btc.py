import pandas as pd
import matplotlib.pyplot as plt

# === Configurație ===
EMA_SHORT = 20
EMA_LONG = 50
INITIAL_CAPITAL = 10000

df = pd.read_csv("./data/BTC_2018_2025_1h.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df = df[df["Date"] >= "2018-01-01"]

df["EMA_short"] = df["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
df["EMA_long"] = df["Close"].ewm(span=EMA_LONG, adjust=False).mean()

df["Signal"] = 0
df["Signal"][EMA_LONG:] = (
    df["EMA_short"][EMA_LONG:] > df["EMA_long"][EMA_LONG:]
).astype(int)
df["Position"] = df["Signal"].diff()

df["Returns"] = df["Close"].pct_change()
df["Strategy_Returns"] = df["Returns"] * df["Signal"].shift(1)
df["Equity"] = (1 + df["Strategy_Returns"]).cumprod() * INITIAL_CAPITAL
df["BuyHold"] = (1 + df["Returns"]).cumprod() * INITIAL_CAPITAL

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Equity"], label="EMA Crossover Strategy")
plt.plot(df["Date"], df["BuyHold"], label="Buy & Hold", linestyle="--")
plt.title(f"BTC EMA Crossover vs Buy & Hold ({EMA_SHORT}/{EMA_LONG}) [2018–2025]")
plt.xlabel("Data")
plt.ylabel("Capital")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

total_return = df["Equity"].iloc[-1] / INITIAL_CAPITAL - 1
buy_hold_return = df["BuyHold"].iloc[-1] / INITIAL_CAPITAL - 1
print(f"Return strategie EMA: {total_return:.2%}")
print(f"Return Buy & Hold:    {buy_hold_return:.2%}")
