# Minimal demo: VADER sentiment + tiny price baseline
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

# Example texts (replace with your event descriptions / tweets)
texts = [
    "Market expects event to happen — strong chance.",
    "Uncertain outcome, many people disagree.",
    "Breaking: new information increasing chances.",
    "No news; market calm.",
    "Negative update; probability falling."
]

# Mock timestamps and mock prices
rng = pd.date_range("2025-01-01", periods=len(texts), freq="H")
prices = pd.Series([0.52, 0.50, 0.56, 0.55, 0.49], index=rng)  # example market probabilities

# Sentiment
analyzer = SentimentIntensityAnalyzer()
sent_scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
df = pd.DataFrame({
    "timestamp": rng,
    "text": texts,
    "price": prices.values,
    "sentiment": sent_scores
}).set_index("timestamp")

# Create a simple label: will price increase next step?
df["price_next"] = df["price"].shift(-1)
df = df.dropna()
df["label_up"] = (df["price_next"] > df["price"]).astype(int)

# Features: price momentum + sentiment
df["price_diff_1"] = df["price"].diff().fillna(0)
X = df[["price", "price_diff_1", "sentiment"]]
y = df["label_up"]

# TimeSeries cross-validation
tscv = TimeSeriesSplit(n_splits=3)
aucs = []
accs = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    accs.append(accuracy_score(y_test, preds))
    aucs.append(roc_auc_score(y_test, probs))

print("Accuracy:", np.mean(accs), "AUC:", np.mean(aucs))
print(df[["text","price","sentiment","label_up"]])