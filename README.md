# COMP 262 – NLP and Recommendation Systems

**Student:** Izzet Abidi
**Student ID:** 300898230
**Program:** Artificial Intelligence – Software Engineering Technology
**Institution:** Centennial College

---

## Repository Structure

```
├── Assign1/     Data Preprocessing & Sentiment Analysis
├── Assign2/     Deep Learning Intent Classification (Chatbot)
├── Assign3/     Recommender Systems (Association Rules + Content-Based)
└── README.md    ← You are here (course-level overview)
```

Each assignment directory contains its own `README.md` with granular exercise breakdowns, a full runbook, expected results, definitions, and industry context. This page provides the macro view: how the three assignments connect and build on each other.

---

## Assignments

| # | Topic | Core Technique | Directory |
|---|-------|---------------|-----------|
| 1 | Data Preprocessing & Sentiment Analysis | Web scraping, regex, Word2Vec augmentation, lexicon-based scoring | [Assign1](Assign1/) |
| 2 | Deep Learning Intent Classification | Keras Tokenizer, embedding layers, Sequential model, train/serve split | [Assign2](Assign2/) |
| 3 | Recommender Systems | Apriori association rules, TF-IDF vectorization, cosine similarity | [Assign3](Assign3/) |

---

## Course Progression: Scaffolded Knowledge Map

The three assignments form a deliberate progression through the NLP and recommendation systems landscape. Each assignment introduces new concepts while reinforcing and extending earlier ones.

### Phase 1 → Phase 2 → Phase 3

```
Assignment 1                    Assignment 2                    Assignment 3
─────────────                   ─────────────                   ─────────────
RULE-BASED NLP                  LEARNED REPRESENTATIONS         RECOMMENDATION SYSTEMS

Web scraping          ───→      (data comes pre-structured)     (data comes pre-structured)
Regex cleaning        ───→      Tokenizer handles cleanup  ──→  TF-IDF handles cleanup
Manual word lists     ───→      Learned embeddings         ──→  TF-IDF term weighting
If/else sentiment     ───→      Neural classification      ──→  Similarity computation
Static dictionaries   ───→      Trained model weights      ──→  Precomputed similarity matrix
One-off script        ───→      Train/serve separation     ──→  Precompute/serve separation
```

### How Each Assignment Builds on the Last

#### Assignment 1 → Assignment 2: From Rules to Learning

| What Assignment 1 Does | What Assignment 2 Changes |
|------------------------|--------------------------|
| Matches words against a hand-curated positive/negative word list | Learns which words are associated with which intents from labeled examples |
| Preprocessing is the entire pipeline (clean text → count matches → score) | Preprocessing is just the first step; the model does the heavy lifting |
| Word2Vec is used only for data augmentation (synonym-based insertion) | An Embedding layer learns task-specific word vectors from scratch during training |
| Output is a numeric score (sentiment percentage) | Output is a probability distribution across 17 classes |

**The key insight:** Assignment 1 requires human expertise to define what "positive" and "negative" mean. Assignment 2 requires only labeled examples — the model discovers the patterns itself. This is the fundamental shift from feature engineering to representation learning.

#### Assignment 2 → Assignment 3: From Classification to Recommendation

| What Assignment 2 Does | What Assignment 3 Changes |
|------------------------|--------------------------|
| Maps one input to one label (intent classification) | Maps one input to a ranked list of similar items (recommendation) |
| Keras Tokenizer converts words to integer sequences | TF-IDF Vectorizer converts words to weighted float vectors |
| Embedding layer learns dense vectors for classification | TF-IDF produces sparse vectors for similarity computation |
| Model outputs a single prediction per input | System outputs a ranked list of N recommendations per query |
| Trained model saved for inference | Precomputed similarity matrix saved for lookups |

**The key insight:** Classification asks "what category does this belong to?" Recommendation asks "what other items are most similar to this?" Both require converting text to numeric vectors, but they use those vectors differently — classification feeds them through a neural network; recommendation computes pairwise distances between them.

#### Assignment 1 → Assignment 3: The Full Circle

| Concept from Assignment 1 | How It Reappears in Assignment 3 |
|---------------------------|----------------------------------|
| Text preprocessing (lowercasing, punctuation removal, stopwords) | Applied to song metadata before TF-IDF vectorization |
| Word frequency analysis | TF-IDF is a sophisticated version of word counting that accounts for corpus-level term rarity |
| Data exploration (shape, nulls, distributions) | Extended to handle empty-vs-null distinction and column usability assessment |
| Working with JSON data structures | Both recipes.json and meta_Digital_Music.json.gz require JSON parsing |
| Interactive `input()` loops | Both recommender scripts use the same pattern with validation and exit commands |

---

## Techniques Across All Three Assignments

### Text Representation Evolution

| Assignment | Technique | Representation | Dimensionality |
|-----------|-----------|---------------|----------------|
| 1 | Lexicon matching | Binary (word present or not in positive/negative list) | 2 (positive count, negative count) |
| 1 | Word2Vec (for augmentation) | Dense pre-trained vectors | 300 (Google News vectors) |
| 2 | Keras Embedding | Dense learned vectors | 20 (trained from scratch) |
| 3 | TF-IDF | Sparse weighted vectors | 5,000 (capped vocabulary) |

Each representation captures different information. Lexicon matching captures sentiment polarity. Word2Vec captures semantic similarity between words. Learned embeddings capture task-specific word relationships. TF-IDF captures term importance relative to a corpus. Understanding when to use which representation is the central skill of applied NLP.

### Model Complexity Progression

| Assignment | Approach | Parameters | Training Required |
|-----------|----------|-----------|-------------------|
| 1 | Lexicon lookup (no model) | 0 | No |
| 2 | Embedding + Dense layers | ~18,700 | Yes (500–1000 epochs) |
| 3 | TF-IDF + Cosine similarity (no model) | 0 | No (computed, not trained) |

Assignments 1 and 3 are model-free — they use algorithms (counting, matrix operations) rather than learned parameters. Assignment 2 introduces a trainable model. This illustrates that not every NLP problem requires deep learning; the right approach depends on the task, data size, and explainability requirements.

### The Preprocessing Thread

Every assignment starts with the same fundamental challenge: raw text must become numbers before computation can happen.

```
Assignment 1:  raw HTML  → BeautifulSoup → regex clean → word lists → counts
Assignment 2:  raw text  → lowercase     → Tokenizer   → pad_sequences → integers
Assignment 3:  raw JSON  → column select → text clean   → TfidfVectorizer → float vectors
```

The tooling changes, but the pattern is constant: **acquire → clean → encode → compute**. This four-step pipeline appears in virtually every NLP system in production.

---

## Technology Stack

| Library | Used In | Purpose |
|---------|---------|---------|
| `requests` + `BeautifulSoup` | Assign1 | Web scraping and HTML parsing |
| `pandas` | All | DataFrame operations, data exploration |
| `numpy` | All | Numeric array operations |
| `nltk` | Assign1 | Tokenization, stopword removal |
| `nlpaug` + `gensim` | Assign1 | Data augmentation via Word2Vec synonyms |
| `scikit-learn` | Assign1, 2, 3 | LabelEncoder, TfidfVectorizer, cosine_similarity, F1 score |
| `tensorflow` / `keras` | Assign2 | Neural network construction, training, and inference |
| `apyori` | Assign3 | Apriori algorithm for association rule mining |
| `pickle` | Assign2, 3 | Serialization of tokenizers, encoders, and similarity matrices |

---

## Running the Full Repository

```bash
# Clone
git clone https://github.com/ixxet/COMP262-NLP-and-Recommendation-Systems.git
cd COMP262-NLP-and-Recommendation-Systems

# Install all dependencies
pip install requests beautifulsoup4 pandas numpy nltk nlpaug gensim scikit-learn tensorflow apyori

# Assignment 1
cd Assign1
python exercise1_web_scraping.py
python exercise2_preprocessing.py      # Requires GoogleNews vectors (1.5GB, not in repo)
python exercise3_sentiment_analysis.py
cd ..

# Assignment 2
cd Assign2
python Izzet_train_chatbot.py          # Trains model, saves to model/
python Izzet_test_chatbot.py           # Interactive chatbot
cd ..

# Assignment 3
cd Assign3
python Izzet_cuisine_recommender.py    # Association rules recommender
python Izzet_songs_recommender.py      # Content-based recommender (first run: ~3 min)
cd ..
```

See each assignment's `README.md` for detailed prerequisites, expected output, and troubleshooting.

---

## Industry Context: Where This All Leads

The three assignments map directly to production NLP and recommendation system components:

| Assignment Concept | Industry Application |
|-------------------|---------------------|
| Web scraping + preprocessing | Data ingestion pipelines (Scrapy, Apache Airflow, dbt) |
| Lexicon-based sentiment | Brand monitoring, customer feedback triage (before ML adoption) |
| Word2Vec augmentation | Training data expansion for low-resource NLP tasks |
| Intent classification | Customer service chatbots (Rasa, Dialogflow, Amazon Lex) |
| Train/serve separation | MLOps model deployment (MLflow, SageMaker, Vertex AI) |
| Association rule mining | Market basket analysis, cross-sell/upsell engines (retail, e-commerce) |
| TF-IDF + cosine similarity | Content-based search and recommendation (Elasticsearch, Solr) |
| Precomputed similarity matrices | Offline recommendation serving (Redis, Memcached, feature stores) |

The progression from rule-based to learned to similarity-based systems mirrors how most companies evolve their NLP capabilities: start with rules (fast, interpretable), add ML models as data grows, and eventually build recommendation engines that synthesize user and content signals at scale.
