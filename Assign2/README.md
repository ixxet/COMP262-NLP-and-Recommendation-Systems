# Assignment 2 – Chatbots (Deep Learning Intent Classification)

**Course:** COMP 262 – NLP and Recommendation Systems
**Weight:** Due Week 9
**Student:** Izzet Abidi (300898230)

---

## Table of Contents

1. [Overview](#overview)
2. [Exercise Breakdown](#exercise-breakdown)
   - [Part 1: Data Preparation (30%)](#part-1-data-preparation-30)
   - [Part 2: Preprocessing (20%)](#part-2-preprocessing-20)
   - [Part 3: Deep Learning Training (20%)](#part-3-deep-learning-training-20)
   - [Part 4: Testing the Bot (30%)](#part-4-testing-the-bot-30)
3. [Runbook](#runbook)
4. [Expected Results](#expected-results)
5. [Topics Learned](#topics-learned)
6. [Definitions and Key Concepts](#definitions-and-key-concepts)
7. [Potential Improvements and Industry Considerations](#potential-improvements-and-industry-considerations)

---

## Overview

This assignment transitions from rule-based NLP (Assignment 1's lexicon matching) to **learned representations**. Instead of manually curating word lists and writing if/else rules, a neural network learns to map user utterances to intent categories from labeled examples. This is the fundamental shift that powers modern conversational AI — the system generalizes from patterns it has seen to patterns it has not.

The task is a **coffee shop order-process chatbot**: 17 intents covering greetings, menu inquiries, drink orders, payment, store info, and order management. The model architecture uses an embedding layer followed by dense layers — a lightweight intent classifier suitable for small-vocabulary, closed-domain bots.

---

## Exercise Breakdown

### Part 1: Data Preparation (30%)

**Objective:** Build a domain-specific intents dataset for a coffee shop chatbot.

**What was done:**
1. Started from `sample_intents.json` (2 intents: greeting, thanks) and expanded it to `Izzet_intents.json` with **17 total intents**.
2. Each intent includes:
   - A `tag` (the intent label the model classifies into)
   - 4+ `patterns` (user utterances that express this intent)
   - 3–4 `responses` (bot replies to randomly select from)
3. The 15 new intents cover the full coffee shop order process:

| Intent Tag | Purpose | Example Pattern |
|-----------|---------|----------------|
| `goodbye` | End conversation | "See you later" |
| `menu` | Ask what's available | "What drinks do you serve?" |
| `order_espresso` | Order a specific drink | "I want an espresso" |
| `order_latte` | Order a specific drink | "Can I get a latte?" |
| `order_cappuccino` | Order a specific drink | "One cappuccino please" |
| `order_cold_brew` | Order a cold drink | "I'd like an iced coffee" |
| `order_mocha` | Order a specific drink | "Can I get a mocha?" |
| `sizes` | Ask about drink sizes | "What sizes do you have?" |
| `prices` | Ask about pricing | "How much is a coffee?" |
| `customization` | Modify a drink | "Do you have oat milk?" |
| `payment` | Ask about payment methods | "Do you accept credit cards?" |
| `hours` | Ask about store hours | "What time do you close?" |
| `location` | Ask about store location | "Where are you located?" |
| `order_status` | Check order progress | "Is my order ready?" |
| `cancel_order` | Cancel an existing order | "I want to cancel my order" |

**Key design decisions:**
- **Diverse phrasing per intent** — each intent has 4 patterns with varied wording (formal, casual, direct, indirect) so the model learns semantic similarity rather than memorizing exact strings.
- **Realistic responses** — responses are written as a real barista would speak, not generic chatbot filler. This makes demo conversations look natural.
- **Granular drink intents over a single "order" intent** — separate intents per drink type allows the bot to acknowledge the specific item ordered. A production system would use entity extraction instead, but the assignment scope uses intent-per-drink.

**File manifest:**
| File | Purpose |
|------|---------|
| `sample_intents.json` | Original starter file (2 intents) |
| `Izzet_intents.json` | Expanded intents file (17 intents, 68+ patterns) |

---

### Part 2: Preprocessing (20%)

**Objective:** Encode intents and tokenize patterns into a format suitable for neural network training.

**What the script does:**
1. **Label encoding** — converts the 17 string tags (e.g., `"order_latte"`, `"payment"`) into integer labels (0–16) using sklearn's `LabelEncoder`. Neural networks require numeric targets.
2. **Tokenization** — uses Keras `Tokenizer` with:
   - `fit_on_texts()` — scans all patterns to build a word→integer vocabulary (word_index)
   - `texts_to_sequences()` — converts each pattern string into a list of integers
   - `pad_sequences()` — pads or truncates all sequences to exactly 40 tokens (post-padding with zeros)
3. The `<OOV>` (out-of-vocabulary) token is registered so unknown words at test time get mapped to index 1 instead of being silently dropped.

**Key design decisions:**
- **Post-padding over pre-padding** — for short sequences processed by GlobalAveragePooling, padding position has minimal impact. Post-padding is the Keras default and keeps the actual content left-aligned, which is more intuitive for debugging.
- **Vocabulary cap at 900** — the assignment specifies this limit. With only ~68 patterns containing ~150 unique words, 900 is more than sufficient. The cap prevents the embedding matrix from growing unnecessarily large.
- **Max length 40** — also assignment-specified. Most coffee shop utterances are under 10 words, so the model learns with mostly zero-padded vectors.

---

### Part 3: Deep Learning Training (20%)

**Objective:** Train a Sequential model to classify padded token sequences into one of 17 intent classes.

**Model architecture** (as specified in the assignment):

```
Layer                        Output Shape    Parameters
─────────────────────────────────────────────────────────
Embedding(900, 20, len=40)   (None, 40, 20)  18,000
GlobalAveragePooling1D       (None, 20)      0
Dense(16, relu)              (None, 16)      336
Dense(10, sigmoid)           (None, 10)      170
Dense(17, softmax)           (None, 17)      187
─────────────────────────────────────────────────────────
Total params: 18,693
```

**Training procedure:**
1. **500 epochs** — trains a fresh model, reports final accuracy.
2. **1000 epochs** — trains a second fresh model from scratch for fair comparison.
3. Both runs use `sparse_categorical_crossentropy` loss (integer labels, not one-hot) and the `adam` optimizer.

**Key design decisions:**
- **Fresh model per epoch count** — rather than continuing training from 500 to 1000, a new model is initialized each time. This ensures the comparison is between "500 total epochs" vs. "1000 total epochs," not "500" vs. "500 + 500 more."
- **GlobalAveragePooling1D over Flatten** — averages across the sequence dimension, producing a fixed-size vector regardless of input length. This is computationally cheaper than Flatten and acts as a regularizer by smoothing over positional noise.
- **Sigmoid in the second dense layer** — unusual for intermediate layers (ReLU is standard), but this is per the assignment specification. Sigmoid squashes values to [0, 1], which can limit gradient flow but works acceptably on this small dataset.

---

### Part 4: Testing the Bot (30%)

**Objective:** Load the saved model and run interactive conversations.

**What the script does:**
1. Loads the TensorFlow SavedModel from `model/chatbot_model/`.
2. Loads the pickled Keras Tokenizer and sklearn LabelEncoder from `model/`.
3. Reads `Izzet_intents.json` to build a `{tag: [responses]}` lookup dictionary.
4. Enters an interactive loop:
   - Accepts user input via `input()`
   - Lowercases and tokenizes the input using the same tokenizer from training
   - Pads to length 40 (matching training dimensions)
   - Runs `model.predict()` to get a probability distribution over 17 classes
   - Takes `argmax` as the predicted class, decodes it back to a tag string
   - Looks up the tag in the responses dictionary and randomly selects one
   - Prints the response and a debug line showing predicted intent + confidence
5. Exits when the user types `"bye"`.

**Key design decisions:**
- **Same tokenizer for train and test** — this is critical. A new tokenizer would assign different integer mappings to the same words, producing garbage predictions. Persisting the tokenizer with pickle ensures consistent encoding.
- **Confidence score in debug output** — shows the softmax probability of the predicted class. Useful during demo to show when the model is confident (>0.9) vs. uncertain (<0.5).
- **Random response selection** — prevents the bot from being repetitive when the same intent is triggered multiple times in a conversation.

**File manifest:**
| File | Purpose |
|------|---------|
| `Izzet_train_chatbot.py` | Training script — run this first |
| `Izzet_test_chatbot.py` | Interactive testing script — run after training |
| `Izzet_intents.json` | Intent definitions (17 intents) |
| `model/chatbot_model/` | Saved TensorFlow model (created by training script) |
| `model/tokenizer.pkl` | Saved Keras Tokenizer (created by training script) |
| `model/label_encoder.pkl` | Saved sklearn LabelEncoder (created by training script) |

---

## Runbook

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install tensorflow numpy scikit-learn

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Training the Model

```bash
cd Assign2/

# Step 1: Train the model (creates model/ directory with saved artifacts)
python Izzet_train_chatbot.py
```

This takes 2–5 minutes depending on hardware. The script trains twice (500 and 1000 epochs) and saves the 1000-epoch model.

### Testing the Bot

```bash
# Step 2: Run the interactive chatbot
python Izzet_test_chatbot.py
```

**Sample conversation:**
```
You: Hi
Bot: Hello! Welcome to our coffee shop.

You: What do you have?
Bot: We serve espresso, latte, cappuccino, americano, mocha, cold brew, and tea!

You: I want a latte
Bot: One latte on the way!

You: How much is that?
Bot: Our drinks range from $3.50 to $6.50 depending on size and type.

You: bye
Bot: Goodbye! Thanks for visiting our coffee shop!
```

### Troubleshooting

| Issue | Resolution |
|-------|------------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Install with `pip install tensorflow`. On Apple Silicon Macs, use `pip install tensorflow-macos tensorflow-metal` for GPU acceleration. |
| `FileNotFoundError` on model directory | Run `Izzet_train_chatbot.py` first — it creates the `model/` directory. |
| Low accuracy after training | Expected on small datasets. 17 classes with only 4 samples each is very little data. The model relies heavily on the embedding layer to generalize. |
| `OMP: Error` or threading warnings | TensorFlow threading warnings are cosmetic. Set `export OMP_NUM_THREADS=1` to suppress them. |
| Incorrect predictions at test time | The model works best with inputs similar to training patterns. Novel phrasing or typos may confuse it. |

---

## Expected Results

### Training Output
- **500 epochs:** Training accuracy converges to approximately **95–100%** on this small dataset. The model memorizes the 68 training patterns quickly.
- **1000 epochs:** Training accuracy reaches **~100%** as the model fully memorizes the patterns.
- **Key observation:** Near-perfect training accuracy on 68 samples does not indicate generalization ability. The model has likely overfit, which is expected given the data size. The purpose is to demonstrate the training pipeline, not to build a production-grade classifier.

### Model Summary
- Total parameters: ~18,693
- The embedding layer accounts for 96% of parameters (18,000 out of 18,693), which is typical for small NLP models.

### Interactive Testing
- The bot correctly responds to inputs that closely match training patterns.
- Novel phrasings (e.g., "gimme a coffee" when trained on "I want a coffee") may produce incorrect predictions due to limited vocabulary coverage.
- Confidence scores above 0.8 generally correspond to correct predictions.

---

## Topics Learned

### Deep Learning for NLP
- **Intent classification as a supervised learning problem** — mapping variable-length text inputs to fixed categorical labels using a neural network, as opposed to the rule-based lexicon matching in Assignment 1.
- **Embedding layers** — learning dense vector representations of words during training, as opposed to using pre-trained static embeddings (Word2Vec in Assignment 1). Here the embeddings are task-specific: words that appear in similar intents get similar vectors.
- **Sequence modeling with pooling** — using GlobalAveragePooling1D to collapse a sequence of word embeddings into a single fixed-size vector, enabling classification with standard dense layers.

### Text-to-Tensor Pipeline
- **Tokenizer persistence** — the critical requirement that the same tokenizer used during training must be used during inference. Different tokenizers produce different word→integer mappings, breaking the model.
- **Sequence padding** — ensuring all inputs have identical dimensions (length 40) so they can be batched into tensors. Padding with zeros and using post-padding to keep content left-aligned.
- **OOV handling** — registering an `<OOV>` token so words not seen during training still get a valid integer representation instead of being dropped.

### Model Architecture and Training
- **Sequential model construction** — stacking layers linearly in Keras, understanding that each layer's output shape becomes the next layer's input shape.
- **Loss function selection** — `sparse_categorical_crossentropy` for integer-labeled multi-class classification (vs. `categorical_crossentropy` for one-hot labels).
- **Overfitting on small data** — observing that 68 training samples across 17 classes leads to near-perfect training accuracy but limited generalization, illustrating the need for more data or regularization.

### Model Serialization
- **Pickle for Python objects** — serializing the Keras Tokenizer and sklearn LabelEncoder, which are standard Python objects with internal state (vocabulary, class mappings).
- **TensorFlow SavedModel** — saving the full model graph and weights in TensorFlow's native format, enabling loading without needing the original model-building code.
- **Separation of training and inference** — two independent scripts that communicate only through saved files, mimicking the train→deploy pattern in production ML.

### Chatbot Design
- **Intent-response architecture** — the standard pattern for task-oriented bots: classify the user's intent, then select a response from a predefined set. This is distinct from generative models that produce novel text.
- **Closed-domain vs. open-domain** — this bot handles a fixed set of 17 intents. It cannot handle topics outside its training data. Understanding this boundary is essential for setting user expectations.

---

## Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Intent** | The purpose or goal behind a user's utterance. In a chatbot context, each intent represents a distinct action or information request (e.g., "order_latte", "check_hours"). |
| **Pattern (Utterance)** | A sample phrase that a user might say to express a particular intent. Multiple patterns per intent teach the model that different phrasings map to the same action. |
| **Response** | A predefined reply associated with an intent. The bot selects one randomly to avoid repetition. |
| **Embedding Layer** | A trainable lookup table that maps integer token IDs to dense vectors. Unlike pre-trained embeddings (Word2Vec), these are learned from scratch during training and become task-specific. |
| **GlobalAveragePooling1D** | A layer that averages all vectors in a sequence into a single vector. Converts a (batch, sequence_length, embedding_dim) tensor to (batch, embedding_dim), enabling classification with dense layers. |
| **Dense Layer** | A fully connected neural network layer where every input neuron connects to every output neuron. The core building block for classification heads. |
| **ReLU (Rectified Linear Unit)** | An activation function: f(x) = max(0, x). Standard for hidden layers because it avoids the vanishing gradient problem and is computationally cheap. |
| **Sigmoid** | An activation function: f(x) = 1 / (1 + e^(-x)). Squashes output to [0, 1]. Typically used for binary classification or as a gate mechanism; used here per assignment specification. |
| **Softmax** | An activation function that converts a vector of raw scores into a probability distribution (all values sum to 1). Used on the output layer for multi-class classification. |
| **Sparse Categorical Crossentropy** | A loss function for multi-class classification where labels are integers (not one-hot encoded). Measures the difference between predicted probabilities and the true class. |
| **Adam Optimizer** | An adaptive learning rate optimizer (Kingma & Ba, 2014) that combines momentum and RMSProp. The default choice for most deep learning tasks due to its robustness across hyperparameter ranges. |
| **Epoch** | One complete pass through the entire training dataset. More epochs allow the model to learn more, but risk overfitting on small datasets. |
| **Overfitting** | When a model memorizes training data rather than learning generalizable patterns. Indicated by high training accuracy but poor performance on unseen data. Especially likely with small datasets. |
| **Tokenizer** | A tool that converts raw text into sequences of integer tokens. The Keras Tokenizer builds a word→index vocabulary and uses it to encode new text consistently. |
| **Padding** | Adding zeros to sequences so all inputs have the same length, required because neural networks operate on fixed-size tensors. |
| **OOV (Out-of-Vocabulary)** | Words encountered at inference time that were not present in the training vocabulary. Handled by mapping them to a special `<OOV>` token index. |
| **LabelEncoder** | A sklearn utility that maps categorical string labels to consecutive integers (0, 1, 2, ...) and back. Required because neural networks output numeric predictions. |
| **Pickle** | Python's built-in serialization protocol for saving and loading Python objects to/from binary files. Used here to persist the Tokenizer and LabelEncoder. |
| **SavedModel** | TensorFlow's native format for exporting trained models, including the computation graph, weights, and metadata. Enables loading without the original training code. |
| **Inference** | The process of running a trained model on new input to produce predictions, as opposed to training (updating weights). |
| **Closed-Domain Chatbot** | A bot designed to handle a fixed, predefined set of topics. Cannot respond meaningfully to queries outside its training scope. |
| **Confidence Score** | The softmax probability of the predicted class. Higher values indicate the model is more certain about its prediction. |

---

## Potential Improvements and Industry Considerations

### Data and Training

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **68 hand-written patterns** across 17 intents | **Data augmentation** (paraphrasing, back-translation) or **few-shot learning** with large language models to generate hundreds of synthetic training examples per intent. | Hand-written patterns are limited in diversity. Augmented data improves generalization significantly, but generated patterns need quality review. |
| **Training from scratch** on small data | **Transfer learning** — fine-tune a pre-trained language model (BERT, DistilBERT) that already understands English grammar and semantics. Requires far fewer examples to reach high accuracy. | Pre-trained models are larger (66M+ parameters vs. 18K) and slower to fine-tune, but achieve dramatically better accuracy with small training sets. |
| **Single train/test split** (no validation set) | **K-fold cross-validation** or a held-out validation set to measure generalization during training and detect overfitting early. | With only 4 patterns per intent, splitting further leaves almost no training data. The current approach acknowledges overfitting as a known limitation. |

### Model Architecture

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Embedding + GlobalAvgPool + Dense** | **Transformer-based models** (BERT, DistilBERT) — contextual representations that understand word order, negation, and long-range dependencies. Industry standard for intent classification since 2019. | Transformers are 1000x larger and require GPU for reasonable training times. The current architecture trains in seconds on CPU, which is appropriate for a learning exercise. |
| **Custom Embedding(900, 20)** learned from scratch | **Pre-trained word embeddings** (GloVe, FastText) as the embedding layer initialization, then fine-tuned during training. | Pre-trained embeddings give the model a head start on word semantics. With only 68 samples, the custom embedding barely sees each word and cannot learn meaningful relationships. |
| **Dense(10, sigmoid)** as second hidden layer | **Dense(10, relu)** or **Dropout layers** — ReLU is standard for hidden layers (avoids vanishing gradients); Dropout randomly zeroes neurons during training to reduce overfitting. | Sigmoid in hidden layers can cause slow training due to gradient saturation. It is used here per assignment specification but would not be chosen in practice. |

### Chatbot Architecture

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Intent-only classification** (no entity extraction) | **Intent + Entity extraction** (e.g., Rasa NLU, Dialogflow) — extracts both the intent ("order_drink") and entities ("latte", "large", "oat milk") from a single utterance. | The current approach needs a separate intent per drink. Entity extraction allows one "order" intent with slots, scaling to hundreds of menu items without adding intents. |
| **Stateless responses** (no conversation memory) | **Dialogue state tracking** — maintains context across turns (e.g., remembering the user ordered a latte when they ask about price). Frameworks: Rasa, Botpress, Amazon Lex. | Stateless bots cannot handle multi-turn flows like "I want a latte" → "What size?" → "Large." Adding state management is the next evolution of chatbot complexity. |
| **Random response selection** | **Template-based responses** with slot filling (e.g., "Your {drink} will be ready in {time}!") or **generative responses** using a language model. | Random selection from a fixed list is simple but cannot personalize. Slot filling adds personalization; generative models add fluency but risk hallucination. |
| **Predefined JSON responses** | **Retrieval-augmented generation (RAG)** — retrieve relevant documents (menu, FAQ) and generate context-aware responses using an LLM. | RAG handles dynamic information (changing menus, seasonal items) without retraining. Overkill for 17 fixed intents but essential for production bots. |

### Deployment

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Command-line `input()` loop** | **REST API** (Flask/FastAPI) serving the model, consumed by a web UI, Slack bot, or messaging platform integration. | The `input()` loop is the simplest possible interface. A REST API decouples the model from the frontend and enables scaling, logging, and A/B testing. |
| **TensorFlow SavedModel on disk** | **TensorFlow Serving**, **TorchServe**, or **ONNX Runtime** for optimized model inference in production. | SavedModel is the right serialization format but loading it in a Python script adds startup latency. Serving frameworks keep the model warm in memory. |
| **No monitoring or logging** | **MLflow**, **Weights & Biases**, or custom logging for tracking predictions, confidence distributions, and conversation quality metrics. | Without logging, there is no way to detect model degradation or understand failure modes in production. |

### Where the Baseline Tech Still Holds Up

Despite the availability of LLM-powered bots (ChatGPT, Claude, Gemini), the lightweight intent classifier approach used in this assignment remains the right choice in several real-world scenarios:

- **Edge/embedded devices** — a model with 18,693 parameters runs on microcontrollers and IoT devices where large models cannot. Think kiosk ordering systems, in-car assistants, or factory floor terminals with no internet connection.
- **Latency-critical systems** — inference takes <1ms on CPU. LLM-based bots take 200ms–2s per response. For real-time voice assistants or high-frequency trading support bots, sub-millisecond classification matters.
- **Deterministic, auditable responses** — in regulated industries (healthcare, finance, legal), the bot must give predefined, approved responses. An intent classifier with fixed response templates is fully auditable. Generative models are not.
- **Cost at scale** — classifying millions of messages per day with a small TensorFlow model costs virtually nothing. LLM API calls at that volume cost thousands of dollars monthly.
- **Keras Tokenizer** — while Hugging Face tokenizers (BPE, WordPiece) are the modern standard for transformer models, Keras Tokenizer remains the simplest option for word-level tokenization in small-vocabulary, closed-domain tasks where subword tokenization adds unnecessary complexity.
