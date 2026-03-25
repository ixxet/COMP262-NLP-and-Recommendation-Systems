"""
COMP 262 - Assignment 3, Exercise 2: Content-Based Song Recommender
Student: Izzet Abidi (300898230)

Builds a content-based recommender that suggests the top 10 most similar
song titles based on textual features from the Amazon Digital Music metadata.

Pipeline:
1. Loads meta_Digital_Music.json.gz into a DataFrame (songs_Izzet)
2. Explores the data: shape, dtypes, null/empty counts per column
3. Selects relevant columns: title, brand (artist), description, category
4. Drops rows with missing titles, deduplicates on title
5. Combines text features into a single 'content' column
6. Preprocesses text: lowercasing, punctuation removal, stopword removal
7. Creates TF-IDF vectors for all song content
8. Computes pairwise cosine similarity matrix
9. Saves the similarity matrix and title index to disk
10. Interactive loop: accepts a song title, returns top-10 similar titles
"""

import gzip
import json
import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_FILE = os.path.join(SCRIPT_DIR, "meta_Digital_Music.json.gz")
SIMILARITY_FILE = os.path.join(SCRIPT_DIR, "similarity_data.pkl")


def load_music_data(filepath):
    """
    Loads the gzipped JSON file line by line into a DataFrame.
    Each line is a separate JSON object (JSONL format).
    Names the DataFrame songs_Izzet as required by the assignment.
    """
    records = []
    with gzip.open(filepath, "rb") as f:
        for line in f:
            records.append(json.loads(line))

    songs_Izzet = pd.DataFrame(records)
    return songs_Izzet


def explore_data(df):
    """
    Performs thorough data exploration:
    - Shape, columns, dtypes
    - Null counts AND empty string/list counts per column
    - Suggests which columns to keep and why
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")

    # Check for nulls AND empty values (empty strings, empty lists)
    exploration_data = []
    for col in df.columns:
        null_count = df[col].isnull().sum()

        # Check for empty strings and empty lists
        if df[col].dtype == object:
            empty_count = df[col].apply(
                lambda x: (isinstance(x, str) and x.strip() == "") or
                           (isinstance(x, list) and len(x) == 0)
            ).sum()
        else:
            empty_count = 0

        non_empty = len(df) - null_count - empty_count
        exploration_data.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Null": null_count,
            "Empty": empty_count,
            "Has Data": non_empty,
            "% Usable": f"{non_empty / len(df) * 100:.1f}%"
        })

    exploration_df = pd.DataFrame(exploration_data)
    print(exploration_df.to_string(index=False))

    # Column selection rationale
    print("\n" + "-" * 60)
    print("COLUMN SELECTION RATIONALE")
    print("-" * 60)
    print("""
    KEEP:
    - title:       The primary identifier. Required for user input/output matching.
    - brand:       Represents the artist/band name. Songs by the same artist
                   often share stylistic similarities (genre, themes, production).
    - description: Contains textual descriptions of the music. Rich semantic
                   content for TF-IDF vectorization.
    - category:    Genre/category tags. Directly relevant for similarity matching.

    DROP:
    - tech1, tech2, fit, similar_item, date: Mostly empty across all records.
    - also_buy, also_view:    Collaborative filtering signals, not content features.
    - main_cat:               Contains HTML img tags, not usable text content.
    - rank:                   Sales rank string — not a content feature.
    - price:                  Numeric, not a content feature for text similarity.
    - asin:                   Amazon product ID — identifier, not a feature.
    - feature:                Mostly empty lists for music items.
    - imageURL, imageURLHighRes: Image URLs — not text content.
    """)

    return exploration_df


def prepare_features(df):
    """
    Selects relevant columns, cleans the data, and builds the feature space.

    Steps:
    1. Keep only: title, brand, description, category
    2. Drop rows where title is null or empty (cannot recommend without a title)
    3. Deduplicate on title (keeps first occurrence)
    4. Convert list columns (description, category) to joined strings
    5. Fill remaining nulls with empty strings
    6. Combine all text features into a single 'content' column
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Select columns
    cols_to_keep = ["title", "brand", "description", "category"]
    df_clean = df[cols_to_keep].copy()
    print(f"\nSelected columns: {cols_to_keep}")
    print(f"Starting shape: {df_clean.shape}")

    # Drop rows without titles
    df_clean = df_clean[df_clean["title"].notnull()]
    df_clean = df_clean[df_clean["title"].str.strip() != ""]
    print(f"After dropping empty titles: {df_clean.shape}")

    # Deduplicate on title
    df_clean = df_clean.drop_duplicates(subset="title", keep="first")
    print(f"After deduplication on title: {df_clean.shape}")

    # Convert list columns to strings
    for col in ["description", "category"]:
        df_clean[col] = df_clean[col].apply(
            lambda x: " ".join(x) if isinstance(x, list) else (str(x) if pd.notnull(x) else "")
        )

    # Fill remaining nulls
    df_clean = df_clean.fillna("")

    # Combine all text features into a single content column
    df_clean["content"] = (
        df_clean["title"] + " " +
        df_clean["brand"] + " " +
        df_clean["description"] + " " +
        df_clean["category"]
    )

    print(f"Combined text features into 'content' column")
    print(f"Sample content (first record, first 200 chars):")
    print(f"  '{df_clean['content'].iloc[0][:200]}...'")

    return df_clean


def preprocess_text(text):
    """
    Preprocesses a single text string:
    1. Lowercase
    2. Remove HTML tags (some descriptions contain HTML)
    3. Remove punctuation
    4. Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf_and_similarity(df_clean):
    """
    Builds TF-IDF vectors and computes the cosine similarity matrix.

    Steps:
    1. Preprocess all content text
    2. Create TF-IDF vectors using sklearn TfidfVectorizer
       - max_features=5000 to limit vocabulary (memory/speed trade-off)
       - stop_words='english' to remove common English stopwords
    3. Compute pairwise cosine similarity between all song vectors

    Returns the similarity matrix and a mapping of title→index.
    """
    print("\n" + "=" * 60)
    print("TF-IDF VECTORIZATION AND SIMILARITY COMPUTATION")
    print("=" * 60)

    # Preprocess
    print("\nPreprocessing text content...")
    df_clean["content_clean"] = df_clean["content"].apply(preprocess_text)

    # TF-IDF
    print("Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)  # unigrams and bigrams for better phrase matching
    )
    tfidf_matrix = tfidf.fit_transform(df_clean["content_clean"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    # Cosine similarity
    print("Computing pairwise cosine similarity...")
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Build title-to-index mapping
    titles = df_clean["title"].tolist()
    title_to_idx = {title.lower(): idx for idx, title in enumerate(titles)}

    return similarity_matrix, titles, title_to_idx


def save_similarity_data(similarity_matrix, titles, title_to_idx, filepath):
    """
    Saves the precomputed similarity matrix and title mappings to disk
    so the recommender can load them without recomputing.
    """
    data = {
        "similarity_matrix": similarity_matrix,
        "titles": titles,
        "title_to_idx": title_to_idx
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSimilarity data saved to: {filepath}")


def load_similarity_data(filepath):
    """Loads precomputed similarity data from disk."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["similarity_matrix"], data["titles"], data["title_to_idx"]


def recommend_songs(song_title, similarity_matrix, titles, title_to_idx, top_n=10):
    """
    Recommender function: takes a song title and returns the top N most
    similar song titles based on cosine similarity scores.

    Steps:
    1. Look up the song's index in the title_to_idx mapping
    2. Get the similarity scores for that song against all others
    3. Sort by similarity (descending), skip the song itself (score=1.0)
    4. Return the top N titles and their scores
    """
    title_lower = song_title.lower()

    if title_lower not in title_to_idx:
        return None

    idx = title_to_idx[title_lower]
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity score descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (the song itself, similarity = 1.0)
    sim_scores = sim_scores[1:top_n + 1]

    recommendations = []
    for i, score in sim_scores:
        recommendations.append((titles[i], score))

    return recommendations


def main():
    print("=" * 60)
    print("Assignment 3, Exercise 2: Content-Based Song Recommender")
    print("=" * 60)

    # Check if precomputed data exists
    if os.path.exists(SIMILARITY_FILE):
        print(f"\nLoading precomputed similarity data from: {SIMILARITY_FILE}")
        similarity_matrix, titles, title_to_idx = load_similarity_data(SIMILARITY_FILE)
        print(f"Loaded {len(titles)} song titles")
    else:
        # Full pipeline: load, explore, prepare, vectorize, save
        print(f"\nLoading music data from: {MUSIC_FILE}")
        songs_Izzet = load_music_data(MUSIC_FILE)
        print(f"Loaded {len(songs_Izzet)} records")

        # Explore
        explore_data(songs_Izzet)

        # Feature engineering
        df_clean = prepare_features(songs_Izzet)

        # TF-IDF and similarity
        similarity_matrix, titles, title_to_idx = build_tfidf_and_similarity(df_clean)

        # Save for future runs
        save_similarity_data(similarity_matrix, titles, title_to_idx, SIMILARITY_FILE)

    # ----- Interactive recommender loop -----
    print("\n" + "=" * 60)
    print("Enter a song title to get recommendations.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    # Show a few sample titles to help the user
    print("\nSample titles in the dataset:")
    for t in titles[:5]:
        print(f"  - {t}")

    while True:
        user_input = input("\nEnter song title: ").strip()

        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        if not user_input:
            print("Please enter a song title.")
            continue

        recommendations = recommend_songs(
            user_input, similarity_matrix, titles, title_to_idx
        )

        if recommendations is None:
            print(f"We don't have recommendations for {user_input}")
            continue

        print(f"\nTop 10 recommendations for: '{user_input}'")
        print(f"{'Rank':<6} {'Song Title':<60} {'Similarity':>10}")
        print("-" * 78)
        for rank, (title, score) in enumerate(recommendations, 1):
            print(f"{rank:<6} {title:<60} {score:>10.4f}")


if __name__ == "__main__":
    main()
