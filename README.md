# RoBERTa Twitter Sentiment Analyzer

This project uses the pre-trained `cardiffnlp/twitter-roberta-base-sentiment` transformer model to classify tweets into **Positive**, **Neutral**, or **Negative** sentiments. It leverages PyTorch and HuggingFace Transformers to analyze real-time tweet text and visualize sentiment distribution.

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Run the Analyzer

```bash
python sentiment_analyzer.py
```

It will:
- Analyze 10 sample tweets
- Print sentiment with confidence
- Display a sentiment distribution bar chart

## 🧠 Model

- Model: [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- Framework: HuggingFace Transformers + PyTorch

## 📊 Example Output

![Sentiment Distribution](sample_output.png)

## 📝 Author

Sumit Kumar
