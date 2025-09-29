import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = yf.download('NFLX', start='2022-01-01', end='2025-09-28')

assert df is not None

df.set_index("")