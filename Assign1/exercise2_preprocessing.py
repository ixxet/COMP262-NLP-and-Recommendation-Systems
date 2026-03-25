"""
COMP 262 - Assignment 1, Exercise 2: Text Preprocessing and Data Augmentation
Student: Izzet Abidi (300898230)

Loads the COVID19_mini.csv dataset, applies preprocessing and data augmentation:
- Cleans tweets (removes RT prefixes, URLs, mentions, special characters)
- Applies Word2Vec embedding augmentation to double the dataset
- Applies random insertion augmentation using Word2Vec synonyms
Exports the augmented dataset to Izzet_df_after_random_insertion.txt
"""

import pandas as pd
import re
import os
import nltk
import nlpaug.augmenter.word as naw

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "COVID19_mini.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "GoogleNews-vectors-negative300.bin.gz")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Izzet_df_after_random_insertion.txt")

STOP_WORDS = set(stopwords.words("english"))


def clean_tweet(text):
    """
    Cleans a single tweet by removing:
    - RT @username: prefix
    - @mentions
    - URLs
    - Hashtag symbols (keeps the word)
    - Special characters and extra whitespace
    """
    # Remove RT @username: prefix
    text = re.sub(r"^RT\s+@\w+:\s*", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 60)
    print("Exercise 2: Text Preprocessing and Data Augmentation")
    print("=" * 60)

    # ----- Step 1: Load and examine data -----
    print("\n--- Step 1: Loading data ---")
    Izzet_df = pd.read_csv(INPUT_FILE)
    print(f"Original dataset shape: {Izzet_df.shape}")
    print(f"\nOriginal data:\n{Izzet_df}")

    # ----- Step 2: Drop user column -----
    print("\n--- Step 2: Dropping 'user' column ---")
    Izzet_df = Izzet_df.drop(columns=["user"])
    print(f"Shape after dropping 'user': {Izzet_df.shape}")
    print(f"\nData after drop:\n{Izzet_df}")

    # ----- Step 3: Clean tweets using regex -----
    print("\n--- Step 3: Cleaning tweets ---")
    Izzet_df["text"] = Izzet_df["text"].apply(clean_tweet)
    print("Cleaned tweets:")
    for i, row in Izzet_df.iterrows():
        print(f"  [{row['sentiment']}] {row['text']}")

    # ----- Step 4: Additional preprocessing -----
    # Remove stopwords from tweets
    print("\n--- Step 4: Removing stop words ---")
    def remove_stopwords(text):
        words = text.split()
        filtered = [w for w in words if w not in STOP_WORDS]
        return " ".join(filtered)

    Izzet_df["text"] = Izzet_df["text"].apply(remove_stopwords)
    print("Tweets after stopword removal:")
    for i, row in Izzet_df.iterrows():
        print(f"  [{row['sentiment']}] {row['text']}")

    original_size = len(Izzet_df)
    print(f"\nOriginal dataset size: {original_size}")

    # ----- Step 5: Word embedding augmentation -----
    print("\n--- Step 5: Word2Vec embedding augmentation ---")
    print(f"Loading Word2Vec model from: {MODEL_PATH}")
    print("(This may take a few minutes...)")

    # Use nlpaug Word2Vec augmenter with substitute action
    word2vec_aug = naw.WordEmbsAug(
        model_type="word2vec",
        model_path=MODEL_PATH,
        action="substitute"
    )

    # Augment each tweet to double the dataset
    augmented_rows = []
    for _, row in Izzet_df.iterrows():
        augmented_text = word2vec_aug.augment(row["text"])
        if isinstance(augmented_text, list):
            augmented_text = augmented_text[0]
        augmented_rows.append({
            "sentiment": row["sentiment"],
            "text": augmented_text
        })

    augmented_df = pd.DataFrame(augmented_rows)
    Izzet_df_after_word_augmenter = pd.concat([Izzet_df, augmented_df], ignore_index=True)

    print(f"Dataset size after word embedding augmentation: {len(Izzet_df_after_word_augmenter)}")
    print(f"  (Original: {original_size}, Augmented: {len(augmented_df)})")
    print(f"\nAugmented dataset:\n{Izzet_df_after_word_augmenter}")

    # ----- Step 6: Random insertion augmentation -----
    print("\n--- Step 6: Random insertion augmentation ---")

    # Use nlpaug with substitute action and Word2Vec for synonym-based augmentation
    random_insert_aug = naw.WordEmbsAug(
        model_type="word2vec",
        model_path=MODEL_PATH,
        action="substitute",
        aug_p=0.3  # Proportion of words to substitute
    )

    insertion_rows = []
    for _, row in Izzet_df.iterrows():
        augmented_text = random_insert_aug.augment(row["text"])
        if isinstance(augmented_text, list):
            augmented_text = augmented_text[0]
        insertion_rows.append({
            "sentiment": row["sentiment"],
            "text": augmented_text
        })

    insertion_df = pd.DataFrame(insertion_rows)
    Izzet_df_after_random_insertion = pd.concat([Izzet_df, insertion_df], ignore_index=True)

    print(f"Dataset size after random insertion: {len(Izzet_df_after_random_insertion)}")
    print(f"\nFinal augmented dataset:\n{Izzet_df_after_random_insertion}")

    # ----- Step 7: Export to file -----
    print(f"\n--- Step 7: Exporting to {OUTPUT_FILE} ---")
    Izzet_df_after_random_insertion.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Exported {len(Izzet_df_after_random_insertion)} rows to {OUTPUT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
