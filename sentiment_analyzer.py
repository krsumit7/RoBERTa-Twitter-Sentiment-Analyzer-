import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import emoji
import matplotlib.pyplot as plt
from collections import Counter

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
labels = ['Negative', 'Neutral', 'Positive']

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment(tweet):
    tweet = emoji.demojize(tweet.lower().strip())
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    sentiment = labels[prediction.item()]
    return sentiment, confidence.item()

sample_tweets = [
    "I love the new features in the iPhone update! 😍📱",
    "I'm not sure how I feel about this policy change.",
    "Worst customer service experience ever. I'm so disappointed. 😡",
    "Looking forward to the weekend! 😊 #happy",
    "Ugh, this day couldn't get any worse. 😤",
    "Great job team! We nailed the project deadline! 🥳",
    "Traffic was horrible this morning. I'm so late!",
    "Meh. It’s just another boring day.",
    "I'm extremely grateful for the support I've received.",
    "Why does everything always go wrong?!"
]

sentiment_results = []

print("RoBERTa Twitter Sentiment Analyzer Results:\n")
for tweet in sample_tweets:
    sentiment, confidence = analyze_sentiment(tweet)
    sentiment_results.append(sentiment)
    print(f"Tweet: {tweet}\n→ Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

# Visualize sentiment distribution
sentiment_counts = Counter(sentiment_results)
plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'gray', 'green'])
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("sample_output.png")
plt.show()
