import yfinance as yf
import pandas as pd

# Download S&P 500 index data (ticker: ^GSPC)
sp500 = yf.download("^GSPC", start="2010-01-01", end="2025-10-13")

# Display the first few rows
print(sp500.head())

# Save to CSV if you want
sp500.to_csv("sp500_data.csv")
