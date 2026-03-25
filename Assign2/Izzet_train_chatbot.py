"""
COMP 262 - Assignment 2: Simple Chatbot Using Deep Learning (Training Script)
Student: Izzet Abidi (300898230)

Trains a deep learning model to classify user input into coffee shop intents.
- Loads intents from Izzet_intents.json
- Encodes intent labels using LabelEncoder
- Tokenizes patterns using Keras Tokenizer with padding
- Builds a Sequential model: Embedding → GlobalAveragePooling1D → Dense layers
- Trains for 500 and 1000 epochs, reporting final accuracy for each
- Saves the trained model, tokenizer, and label encoder to the 'model/' directory
"""

import json
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(SCRIPT_DIR, "Izzet_intents.json")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Model hyperparameters (as specified in the assignment)
VOCAB_SIZE = 900
EMBEDDING_DIM = 20
MAX_LENGTH = 40
DENSE_1_UNITS = 16
DENSE_2_UNITS = 10


def load_intents(filepath):
    """
    Reads the intents JSON file and extracts four lists:
    - tags: the intent category labels
    - patterns: the user utterances
    - responses: the bot responses per intent
    - pattern_tags: the tag associated with each individual pattern
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = []
    patterns = []
    responses = {}
    pattern_tags = []

    for intent in data["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        responses[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            patterns.append(pattern.lower())
            pattern_tags.append(tag)

    return tags, patterns, responses, pattern_tags


def encode_labels(pattern_tags):
    """
    Encodes intent tag strings into integer labels using sklearn LabelEncoder.
    Returns the encoder and the encoded label array.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(pattern_tags)
    return encoder, encoded_labels


def tokenize_patterns(patterns, vocab_size, max_length):
    """
    Tokenizes patterns using Keras Tokenizer:
    1. fit_on_texts - builds the word index from all patterns
    2. texts_to_sequences - converts each pattern to a sequence of integers
    3. pad_sequences - pads/truncates all sequences to max_length
    Returns the tokenizer and padded sequences.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)

    sequences = tokenizer.texts_to_sequences(patterns)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    return tokenizer, padded


def build_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    Builds the Sequential model as specified in the assignment:
    - Embedding layer: vocab_size x embedding_dim, input_length=max_length
    - GlobalAveragePooling1D: reduces sequence to single vector
    - Dense(16, relu): first hidden layer
    - Dense(10, sigmoid): second hidden layer
    - Dense(num_classes, softmax): output layer for intent classification
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(DENSE_1_UNITS, activation="relu"),
        Dense(DENSE_2_UNITS, activation="sigmoid"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def train_and_report(model, X, y, epochs):
    """Trains the model for the given number of epochs and returns the history."""
    print(f"\n{'='*60}")
    print(f"Training with {epochs} epochs")
    print(f"{'='*60}")

    history = model.fit(X, y, epochs=epochs, verbose=1)

    final_accuracy = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print(f"\n--- Results after {epochs} epochs ---")
    print(f"Final Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"Final Loss: {final_loss:.4f}")

    return history, final_accuracy


def main():
    print("=" * 60)
    print("Assignment 2: Coffee Shop Chatbot - Training Script")
    print("=" * 60)

    # ----- Step 1: Load intents -----
    print("\n--- Step 1: Loading intents ---")
    tags, patterns, responses, pattern_tags = load_intents(INTENTS_FILE)
    print(f"Total intents (classes): {len(tags)}")
    print(f"Total patterns (training samples): {len(patterns)}")
    print(f"Intent tags: {tags}")

    # ----- Step 2: Encode labels -----
    print("\n--- Step 2: Encoding intent labels ---")
    encoder, encoded_labels = encode_labels(pattern_tags)
    print(f"Classes: {list(encoder.classes_)}")
    print(f"Encoded labels shape: {encoded_labels.shape}")
    print(f"Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # ----- Step 3: Tokenize patterns -----
    print("\n--- Step 3: Tokenizing patterns ---")
    tokenizer, padded_sequences = tokenize_patterns(patterns, VOCAB_SIZE, MAX_LENGTH)
    print(f"Vocabulary size (unique words): {len(tokenizer.word_index)}")
    print(f"Padded sequences shape: {padded_sequences.shape}")
    print(f"Sample pattern: '{patterns[0]}' -> {padded_sequences[0][:10]}...")

    # ----- Step 4: Build model -----
    print("\n--- Step 4: Building model ---")
    num_classes = len(tags)
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, num_classes)
    model.summary()

    # ----- Step 5: Train with 500 epochs -----
    X = np.array(padded_sequences)
    y = np.array(encoded_labels)

    history_500, acc_500 = train_and_report(model, X, y, epochs=500)

    # ----- Step 6: Rebuild and train with 1000 epochs -----
    # Rebuild fresh model for fair comparison
    model_1000 = build_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, num_classes)
    history_1000, acc_1000 = train_and_report(model_1000, X, y, epochs=1000)

    # ----- Step 7: Comparison -----
    print("\n--- Training Comparison ---")
    print(f"500 epochs  -> Accuracy: {acc_500 * 100:.2f}%")
    print(f"1000 epochs -> Accuracy: {acc_1000 * 100:.2f}%")

    if acc_1000 > acc_500:
        print("Conclusion: The 1000-epoch model achieves higher training accuracy.")
        print("The additional epochs allow the optimizer to converge further.")
    elif acc_1000 == acc_500:
        print("Conclusion: Both models reach the same accuracy, suggesting convergence before 500 epochs.")
    else:
        print("Conclusion: The 500-epoch model slightly outperforms, likely due to random initialization differences.")

    print("Note: Higher training accuracy does not guarantee better generalization.")
    print("With a small dataset, overfitting is likely at higher epoch counts.")

    # ----- Step 8: Save model, tokenizer, and encoder -----
    print(f"\n--- Step 8: Saving to {MODEL_DIR}/ ---")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the 1000-epoch model (more trained)
    model_path = os.path.join(MODEL_DIR, "chatbot_model")
    model_1000.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save tokenizer with pickle
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {tokenizer_path}")

    # Save label encoder with pickle
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    print(f"Label encoder saved to: {encoder_path}")

    print("\nTraining complete. Run Izzet_test_chatbot.py to interact with the bot.")


if __name__ == "__main__":
    main()
