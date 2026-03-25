"""
COMP 262 - Assignment 2: Simple Chatbot Using Deep Learning (Testing Script)
Student: Izzet Abidi (300898230)

Loads the trained chatbot model, tokenizer, and label encoder from the 'model/'
directory. Reads the intents JSON to map predicted tags to stored responses.
Runs an interactive loop:
1. Receives user input via input()
2. Tokenizes and pads the input using the saved tokenizer
3. Passes the padded sequence through the model to predict the intent
4. Decodes the predicted class back to an intent tag using the label encoder
5. Randomly selects a response from the matching intent's response list
6. Continues until the user types "bye"
"""

import json
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(SCRIPT_DIR, "Izzet_intents.json")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MAX_LENGTH = 40


def load_resources():
    """
    Loads all required resources for inference:
    - The trained TensorFlow model
    - The Keras tokenizer (pickled)
    - The sklearn LabelEncoder (pickled)
    - The intents JSON (for response lookup)
    """
    # Load model
    model_path = os.path.join(MODEL_DIR, "chatbot_model")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    # Load tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from: {tokenizer_path}")

    # Load label encoder
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    print(f"Label encoder loaded from: {encoder_path}")

    # Load intents for response lookup
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    # Build a tag-to-responses mapping for quick lookup
    responses = {}
    for intent in intents_data["intents"]:
        responses[intent["tag"]] = intent["responses"]

    return model, tokenizer, encoder, responses


def predict_intent(user_input, model, tokenizer, encoder):
    """
    Processes user input through the inference pipeline:
    1. Lowercases the input
    2. Converts to a token sequence using the saved tokenizer
    3. Pads the sequence to MAX_LENGTH
    4. Runs the model to get class probabilities
    5. Takes the argmax as the predicted class
    6. Decodes back to the intent tag string
    Returns the predicted tag and the confidence score.
    """
    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([user_input.lower()])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")

    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    # Decode the predicted class to tag
    predicted_tag = encoder.inverse_transform([predicted_class])[0]

    return predicted_tag, confidence


def main():
    print("=" * 60)
    print("Coffee Shop Chatbot - Interactive Mode")
    print("=" * 60)
    print("Loading model and resources...")

    model, tokenizer, encoder, responses = load_resources()

    print("\n" + "-" * 60)
    print("Chatbot is ready! Type your message below.")
    print("Type 'bye' to exit.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        # Exit condition
        if user_input.lower() == "bye":
            print("Bot: Goodbye! Thanks for visiting our coffee shop!")
            break

        # Skip empty input
        if not user_input:
            print("Bot: I didn't catch that. Could you say something?")
            continue

        # Predict intent and respond
        predicted_tag, confidence = predict_intent(user_input, model, tokenizer, encoder)

        # Check if the predicted tag exists in our responses
        if predicted_tag in responses:
            response = random.choice(responses[predicted_tag])
            print(f"Bot: {response}")
        else:
            print("Bot: I'm not sure I understand. Could you rephrase that?")

        # Debug info (can be removed in production)
        print(f"     [Debug: intent='{predicted_tag}', confidence={confidence:.4f}]")


if __name__ == "__main__":
    main()
