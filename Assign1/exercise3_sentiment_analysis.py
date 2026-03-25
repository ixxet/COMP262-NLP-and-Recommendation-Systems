"""
COMP 262 - Assignment 1, Exercise 3: Sentiment Analysis
Student: Izzet Abidi (300898230)

Performs lexicon-based sentiment analysis on the COVID19_data.csv dataset:
- Cleans tweets and performs basic data exploration
- Scores each tweet using positive/negative word lexicons
- Calculates predicted sentiment and compares to original labels
- Reports Accuracy and F1 Score
"""

import pandas as pd
import re
import os
from sklearn.metrics import accuracy_score, f1_score

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "COVID19_data.csv")
POS_LEXICON = os.path.join(SCRIPT_DIR, "positive-words.txt")
NEG_LEXICON = os.path.join(SCRIPT_DIR, "negative-words.txt")


def clean_tweet(text):
    """
    Cleans a single tweet by removing:
    - RT @username: prefix
    - @mentions
    - URLs
    - Hashtag symbols (keeps the word)
    - Special characters and extra whitespace
    """
    text = re.sub(r"^RT\s+@\w+:\s*", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lexicon(filepath):
    """Loads a lexicon file (one word per line) into a set."""
    words = set()
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith(";"):
                words.add(word.lower())
    return words


def main():
    print("=" * 60)
    print("Exercise 3: Lexicon-Based Sentiment Analysis")
    print("=" * 60)

    # ----- Step 1: Load data and drop user column -----
    print("\n--- Step 1: Loading data ---")
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape}")
    df = df.drop(columns=["user"])
    print(f"Shape after dropping 'user': {df.shape}")

    # ----- Step 2: Clean tweets -----
    print("\n--- Step 2: Cleaning tweets ---")
    df["text"] = df["text"].apply(clean_tweet)
    print("Tweets cleaned successfully.")

    # ----- Step 3: Basic data exploration -----
    print("\n--- Step 3: Data Exploration ---")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")

    # ----- Step 4: Add tweet length column -----
    print("\n--- Step 4: Adding tweet_len column ---")
    df["tweet_len"] = df["text"].apply(lambda x: len(x.split()))
    print(f"Average tweet length: {df['tweet_len'].mean():.2f} words")
    print(f"Min: {df['tweet_len'].min()}, Max: {df['tweet_len'].max()}")

    # ----- Step 5: Load lexicons -----
    print("\n--- Step 5: Loading lexicons ---")
    positive_words = load_lexicon(POS_LEXICON)
    negative_words = load_lexicon(NEG_LEXICON)
    print(f"Positive words loaded: {len(positive_words)}")
    print(f"Negative words loaded: {len(negative_words)}")

    # ----- Step 6: Calculate sentiment percentages -----
    print("\n--- Step 6: Calculating sentiment scores ---")

    def calculate_positive_pct(text):
        words = text.split()
        if len(words) == 0:
            return 0.0
        pos_count = sum(1 for w in words if w in positive_words)
        return pos_count / len(words)

    def calculate_negative_pct(text):
        words = text.split()
        if len(words) == 0:
            return 0.0
        neg_count = sum(1 for w in words if w in negative_words)
        return neg_count / len(words)

    df["positive_pct"] = df["text"].apply(calculate_positive_pct)
    df["negative_pct"] = df["text"].apply(calculate_negative_pct)

    print("Positive and negative percentages calculated.")
    print(f"\nAverage positive %: {df['positive_pct'].mean():.4f}")
    print(f"Average negative %: {df['negative_pct'].mean():.4f}")

    # ----- Step 7: Predict sentiment -----
    print("\n--- Step 7: Predicting sentiment ---")

    def predict_sentiment(row):
        pos = row["positive_pct"]
        neg = row["negative_pct"]
        if (pos == 0 and neg == 0) or (pos == neg):
            return "neutral"
        elif pos > neg:
            return "positive"
        else:
            return "negative"

    df["predicted_sentiment_score"] = df.apply(predict_sentiment, axis=1)

    print(f"\nPredicted sentiment distribution:\n{df['predicted_sentiment_score'].value_counts()}")

    # ----- Step 8: Calculate accuracy and F1 score -----
    print("\n--- Step 8: Evaluation ---")

    # Strip whitespace from sentiment labels for comparison
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["predicted_sentiment_score"] = df["predicted_sentiment_score"].str.strip().str.lower()

    accuracy = accuracy_score(df["sentiment"], df["predicted_sentiment_score"])
    f1 = f1_score(df["sentiment"], df["predicted_sentiment_score"], average="weighted")

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 Score (weighted): {f1:.4f}")

    # ----- Step 9: Display comparison -----
    print("\n--- Step 9: Comparison (first 10 rows) ---")
    comparison = df[["text", "sentiment", "predicted_sentiment_score"]].head(10)
    for _, row in comparison.iterrows():
        match = "MATCH" if row["sentiment"] == row["predicted_sentiment_score"] else "MISS"
        print(f"  [{match}] Original: {row['sentiment']:>8} | "
              f"Predicted: {row['predicted_sentiment_score']:>8} | "
              f"Tweet: {row['text'][:60]}...")

    # ----- Conclusions -----
    print("\n--- Conclusions ---")
    print(f"The lexicon-based approach achieved an accuracy of {accuracy * 100:.2f}%.")
    print("Potential reasons for misclassifications:")
    print("  1. Lexicon-based methods do not capture context or sarcasm.")
    print("  2. Some tweets contain non-English text, reducing word matches.")
    print("  3. Short tweets may have too few words for reliable scoring.")
    print("  4. The lexicons may not cover domain-specific COVID-19 vocabulary.")
    print("\nSuggestions for improvement:")
    print("  1. Use machine learning models (e.g., Naive Bayes, SVM) trained on labeled data.")
    print("  2. Add domain-specific lexicons for COVID-19 terminology.")
    print("  3. Filter out non-English tweets before analysis.")
    print("  4. Use more advanced NLP techniques like BERT for contextual understanding.")

    print("\nDone.")


if __name__ == "__main__":
    main()
