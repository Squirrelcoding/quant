import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = yf.download('NFLX', start='2022-01-01', end='2025-09-28')

assert df is not None

df['Close'].plot()
plt.show()

print(df.head())


window = 20

df["ma_20"] = df["Close"].rolling(window=window).mean()
df["diff"] = df["Close"] - df["ma_20"]
df['signal'] = np.where(df["diff"] > 0, -1, 1)

# figs=(8,4)

# df[['Close',"ma_20"]].plot(figsize=figs)
# plt.title("Mean Reversion")
# plt.show()

# df['diff'].plot(figsize=figs)
# #I multiplied the signal by 20 be able to show it clearly in the graph
# (20*df['signal']).plot(figsize=figs, linestyle='--')
# plt.title("Diff vs Signal")
# plt.legend()
# plt.show()

# (df["Close"]/df["ma_20"] ).plot(figsize=figs)
# plt.title("Ratio=Close/ma_20")
# plt.show()