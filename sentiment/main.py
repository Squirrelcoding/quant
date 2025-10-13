import requests
from transformers import pipeline
from datetime import datetime

# === 1. Setup ===
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# === 2. Get current news (we'll use a placeholder API) ===
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
TOPIC = "inflation"  # or "Biden", "interest rates", "Trump", etc.
url = f"https://newsapi.org/v2/everything?q={TOPIC}&language=en&pageSize=5&apiKey={NEWS_API_KEY}"

def get_headlines():
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [a["title"] for a in articles]

# === 3. Analyze sentiment ===
def analyze_sentiment(headlines):
    results = sentiment(headlines)
    avg_score = sum(r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in results) / len(results)
    return avg_score

# === 4. Simple trading logic ===
def trading_signal(sentiment_score, threshold=0.1):
    if sentiment_score > threshold:
        return "BUY 📈"
    elif sentiment_score < -threshold:
        return "SELL 📉"
    else:
        return "HOLD 🤔"

# === 5. Combine it all ===
if __name__ == "__main__":
    headlines = get_headlines()
    print(f"[{datetime.now()}] Analyzing {len(headlines)} headlines on '{TOPIC}'...")
    score = analyze_sentiment(headlines)
    signal = trading_signal(score)
    print(f"Average sentiment: {score:.3f} → Suggested action: {signal}")
    print("\nSample headlines:")
    for h in headlines:
        print(" •", h)
