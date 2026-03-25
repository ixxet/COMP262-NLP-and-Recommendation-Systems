# Assignment 1 – Data Preprocessing and Sentiment Analysis

**Course:** COMP 262 – NLP and Recommendation Systems
**Weight:** Due Week 5
**Student:** Izzet Abidi (300898230)

---

## Table of Contents

1. [Overview](#overview)
2. [Exercise Breakdown](#exercise-breakdown)
   - [Exercise 1: Web Scraping (30%)](#exercise-1-web-scraping-30)
   - [Exercise 2: Text Preprocessing & Data Augmentation (35%)](#exercise-2-text-preprocessing--data-augmentation-35)
   - [Exercise 3: Lexicon-Based Sentiment Analysis (35%)](#exercise-3-lexicon-based-sentiment-analysis-35)
3. [Runbook](#runbook)
4. [Expected Results](#expected-results)
5. [Topics Learned](#topics-learned)
6. [Definitions and Key Concepts](#definitions-and-key-concepts)
7. [Potential Improvements and Industry Considerations](#potential-improvements-and-industry-considerations)

---

## Overview

This assignment establishes the foundational NLP pipeline: **acquire → preprocess → analyze**. It covers three core competencies that every NLP system depends on—data acquisition through web scraping, text normalization and augmentation to expand limited datasets, and lexicon-based sentiment classification as a baseline approach before moving to learned models in later assignments.

---

## Exercise Breakdown

### Exercise 1: Web Scraping (30%)

**Objective:** Scrape the Centennial College AI program page and extract structured information.

**What the script does:**
1. Launches a headless Chrome browser via Selenium (the site is React-based, so static `requests` would return an empty shell).
2. Waits for dynamic content to render, then clicks the "Career Options" tab to expose hidden sections.
3. Parses the rendered HTML with BeautifulSoup to extract:
   - The page title
   - Program highlights
   - Companies offering jobs
   - Career outlook entries
4. Exports all extracted data to `Izzet_my_future.csv` with `[Category, Information]` columns.

**Key design decisions:**
- **Selenium over requests/BeautifulSoup alone** — the target page renders content client-side via JavaScript. A static HTTP GET returns skeleton HTML with no data. Selenium renders the full DOM.
- **Text-based parsing over CSS selectors** — the site's class names are auto-generated (React build hashes), making CSS selectors fragile. Parsing by section heading text is more stable across deployments.
- **Explicit waits with fallback** — the Career Options tab click is wrapped in a try/except so the script degrades gracefully if the page layout changes.

**File manifest:**
| File | Purpose |
|------|---------|
| `exercise1_web_scraping.py` | Main scraping script |
| `Izzet_my_future.csv` | Output: scraped data in CSV format |

---

### Exercise 2: Text Preprocessing & Data Augmentation (35%)

**Objective:** Clean a small tweet dataset and expand it using Word2Vec-based augmentation techniques.

**What the script does:**
1. Loads `COVID19_mini.csv` (4 tweets with sentiment labels) into a DataFrame.
2. Drops the `user` column (not needed for text analysis).
3. Applies regex-based cleaning:
   - Strips `RT @username:` prefixes
   - Removes `@mentions` and URLs
   - Removes hashtag symbols (retains the word)
   - Strips special characters, lowercases everything
4. Removes English stopwords using NLTK's stopword corpus.
5. **Word embedding augmentation:** Uses `nlpaug`'s `WordEmbsAug` with Google's pre-trained Word2Vec model (`GoogleNews-vectors-negative300.bin.gz`) to substitute words with semantically similar alternatives. This doubles the dataset from 4 → 8 rows.
6. **Random insertion augmentation:** Applies a second pass of Word2Vec-based substitution (`aug_p=0.3`) to generate additional augmented rows, again doubling from the original 4 → 8, for a combined dataset.
7. Exports the final augmented dataset to `Izzet_df_after_random_insertion.txt` (tab-separated).

**Key design decisions:**
- **`nlpaug` with `action="substitute"`** — the assignment hint explicitly recommends this over manually implementing tokenization → synonym lookup → insertion logic. It abstracts steps iii–v of the random insertion requirement into a single augmenter call.
- **Word2Vec over contextual models** — the assignment mandates Google's pre-trained Word2Vec. In production, contextual augmentation (e.g., with BERT) would produce more natural paraphrases, but Word2Vec is the prescribed tool here.
- **Stopword removal before augmentation** — augmenting stopwords adds noise, not signal. Removing them first ensures substitutions target meaningful content words.

**File manifest:**
| File | Purpose |
|------|---------|
| `exercise2_preprocessing.py` | Preprocessing and augmentation script |
| `COVID19_mini.csv` | Input: 4 labeled tweets |
| `GoogleNews-vectors-negative300.bin.gz` | Pre-trained Word2Vec model (not tracked in git — 1.6 GB) |
| `Izzet_df_after_random_insertion.txt` | Output: augmented dataset (tab-separated) |

---

### Exercise 3: Lexicon-Based Sentiment Analysis (35%)

**Objective:** Classify 100 tweets by sentiment using positive/negative word lexicons and evaluate against ground truth.

**What the script does:**
1. Loads `COVID19_data.csv` (100 tweets with sentiment labels), drops the `user` column.
2. Applies the same regex cleaning pipeline as Exercise 2.
3. Performs basic data exploration: shape, dtypes, sentiment distribution, descriptive statistics.
4. Adds a `tweet_len` column (word count per tweet).
5. Loads Hu & Liu's opinion lexicons (`positive-words.txt`, `negative-words.txt`) — approximately 6,800 words total.
6. For each tweet, calculates:
   - `positive_pct` = (count of positive words) / (total words)
   - `negative_pct` = (count of negative words) / (total words)
7. Assigns a `predicted_sentiment_score` using decision rules:
   - Both zero or equal → `neutral`
   - positive_pct > negative_pct → `positive`
   - negative_pct > positive_pct → `negative`
8. Evaluates predictions against ground truth using scikit-learn's `accuracy_score` and `f1_score` (weighted).
9. Prints a side-by-side comparison of the first 10 predictions with MATCH/MISS indicators.
10. Outputs conclusions about lexicon-based limitations and improvement suggestions.

**Key design decisions:**
- **Normalized percentages over raw counts** — longer tweets naturally contain more lexicon hits. Dividing by word count normalizes for length, preventing bias toward verbose tweets.
- **Weighted F1 over macro F1** — the dataset has imbalanced sentiment classes. Weighted F1 accounts for class frequency, giving a more representative evaluation than macro averaging.
- **Lexicon-based as a baseline** — this approach requires no training data and is fully interpretable. Its limitations (no context, no sarcasm detection, vocabulary gaps) motivate the transition to learned models in Assignment 2.

**File manifest:**
| File | Purpose |
|------|---------|
| `exercise3_sentiment_analysis.py` | Sentiment analysis script |
| `COVID19_data.csv` | Input: 100 labeled tweets |
| `positive-words.txt` | Hu & Liu positive lexicon |
| `negative-words.txt` | Hu & Liu negative lexicon |

---

## Runbook

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install requests beautifulsoup4 selenium pandas numpy nltk nlpaug gensim scikit-learn

# Download NLTK data (runs automatically in scripts, but can be done manually)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Selenium setup:** ChromeDriver must be installed and match the local Chrome version. On macOS:
```bash
brew install chromedriver
```

**Word2Vec model:** Download `GoogleNews-vectors-negative300.bin.gz` (~1.6 GB) and place it in the `Assign1/` directory. This file is not tracked in git due to its size.

### Running the Exercises

```bash
cd Assign1/

# Exercise 1 – Web Scraping
# Outputs: Izzet_my_future.csv
python exercise1_web_scraping.py

# Exercise 2 – Preprocessing & Augmentation
# Requires: GoogleNews-vectors-negative300.bin.gz in this directory
# Outputs: Izzet_df_after_random_insertion.txt
python exercise2_preprocessing.py

# Exercise 3 – Sentiment Analysis
# Outputs: terminal report with accuracy and F1 score
python exercise3_sentiment_analysis.py
```

### Troubleshooting

| Issue | Resolution |
|-------|------------|
| Selenium `WebDriverException` | Ensure ChromeDriver version matches Chrome browser version. Run `chromedriver --version` and compare with `chrome://version`. |
| `FileNotFoundError` on Word2Vec model | Download `GoogleNews-vectors-negative300.bin.gz` into `Assign1/`. The file is ~1.6 GB compressed. |
| NLTK `LookupError` | Run `python -c "import nltk; nltk.download('all')"` to fetch all NLTK datasets. |
| Empty scraping results | The Centennial College site may have changed its DOM structure. Check that the URL still resolves and inspect the page source for updated section headings. |

---

## Expected Results

### Exercise 1
- A CSV file (`Izzet_my_future.csv`) containing rows categorized as `Title`, `Program Highlight`, `Company Offering Jobs`, and `Career Outlook`.
- Terminal output showing the count of items scraped per category.
- Typical output: 1 title, 3–5 program highlights, 5–8 companies, 4–6 career outlook entries (exact counts depend on current page content).

### Exercise 2
- The original 4-tweet dataset is doubled to 8 rows after word embedding augmentation.
- After random insertion augmentation, a second set of 4 augmented rows is appended, resulting in 8 total rows in the final export.
- The exported file `Izzet_df_after_random_insertion.txt` contains tab-separated columns: `sentiment` and `text`.
- Augmented tweets should read as plausible paraphrases—substituted words are semantically related to the originals (e.g., "virus" → "pathogen").

### Exercise 3
- A terminal report showing:
  - Dataset shape (100 rows × 2 columns after dropping `user`)
  - Sentiment distribution across positive, negative, and neutral classes
  - Average positive and negative percentages
  - Predicted sentiment distribution
  - **Accuracy:** typically in the range of 40–60% (lexicon-based approaches on social media text are inherently limited)
  - **F1 Score (weighted):** typically in a similar range
  - A 10-row comparison table showing MATCH/MISS for each prediction

---

## Topics Learned

### Data Acquisition
- **Web scraping with dynamic content** — using Selenium to render JavaScript-heavy pages before parsing with BeautifulSoup, as opposed to static scraping with `requests` which only works on server-rendered HTML.
- **Headless browser automation** — running Chrome without a GUI for automated data extraction in scripted pipelines.
- **DOM traversal and text extraction** — navigating parsed HTML trees to isolate specific content sections.

### Text Preprocessing
- **Regex-based text cleaning** — systematic removal of noise (URLs, mentions, special characters) from social media text using regular expressions.
- **Stopword removal** — filtering high-frequency, low-information words (e.g., "the", "is", "at") to reduce dimensionality and focus on content-bearing tokens.
- **Text normalization** — lowercasing, whitespace normalization, and special character removal to create a uniform representation.

### Data Augmentation
- **Word embedding-based augmentation** — leveraging pre-trained word vectors to generate semantically similar text variants, expanding small datasets without manual labeling.
- **Word2Vec architecture** — understanding that Word2Vec maps words to dense vector spaces where semantic similarity corresponds to vector proximity (cosine similarity).
- **Augmentation strategies** — substitution vs. insertion vs. deletion as different approaches to generating synthetic training data, each with trade-offs in naturalness and diversity.

### Sentiment Analysis
- **Lexicon-based sentiment scoring** — using curated word lists (Hu & Liu opinion lexicons) to assign polarity scores without any model training.
- **Normalization for text length** — dividing hit counts by total word count to prevent length bias in scoring.
- **Evaluation metrics** — using Accuracy and F1 Score to measure classification performance, understanding why F1 (which balances precision and recall) is more informative than accuracy alone on imbalanced datasets.
- **Baseline establishment** — understanding that simple rule-based methods serve as benchmarks against which learned models are compared.

---

## Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Web Scraping** | The automated extraction of data from websites by programmatically fetching and parsing HTML content. |
| **DOM (Document Object Model)** | A tree-structured representation of an HTML document that browsers construct from raw markup; used by JavaScript and scraping tools to traverse and manipulate page elements. |
| **Headless Browser** | A browser instance that runs without a visible UI, used for automated testing and scraping of JavaScript-rendered pages. |
| **Selenium** | A browser automation framework that controls real browsers programmatically, necessary for scraping sites that render content client-side. |
| **BeautifulSoup** | A Python library for parsing HTML and XML documents, providing Pythonic methods to navigate and search parsed document trees. |
| **Regular Expression (Regex)** | A sequence of characters defining a search pattern, used for pattern matching and text substitution in string processing. |
| **Stopwords** | Common words (e.g., "the", "is", "and") that carry minimal semantic meaning and are typically removed during text preprocessing. |
| **Tokenization** | The process of splitting text into individual units (tokens), typically words or subwords, as the first step in most NLP pipelines. |
| **Word2Vec** | A neural network-based model (Mikolov et al., 2013) that learns dense vector representations of words from large corpora, capturing semantic relationships in vector space. |
| **Word Embeddings** | Dense, low-dimensional vector representations of words where geometric proximity encodes semantic similarity. |
| **Data Augmentation** | Techniques for artificially expanding a dataset by generating modified copies of existing samples, used to improve model robustness and reduce overfitting. |
| **Cosine Similarity** | A metric measuring the cosine of the angle between two vectors, used to quantify semantic similarity in embedding spaces. Ranges from -1 (opposite) to 1 (identical). |
| **Sentiment Analysis** | The computational identification and categorization of opinions expressed in text, typically classifying as positive, negative, or neutral. |
| **Lexicon** | A curated dictionary of words annotated with metadata (e.g., polarity), used as a knowledge base for rule-based NLP systems. |
| **Opinion Lexicon (Hu & Liu)** | A widely used sentiment lexicon containing ~2,000 positive and ~4,800 negative English words, developed by Minqing Hu and Bing Liu (2004). |
| **Accuracy** | The proportion of correct predictions out of total predictions. Simple but potentially misleading on imbalanced datasets. |
| **F1 Score** | The harmonic mean of precision and recall, providing a single metric that balances both. Weighted F1 accounts for class imbalance. |
| **Precision** | The proportion of predicted positives that are actually positive (i.e., how many of the model's positive calls were correct). |
| **Recall** | The proportion of actual positives that were correctly identified (i.e., how many true positives the model found). |

---

## Potential Improvements and Industry Considerations

### Exercise 1: Web Scraping

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Selenium** for dynamic rendering | **Playwright** (Microsoft) — faster, more reliable, native async support, multi-browser. Considered the modern successor to Selenium for web automation. | Playwright has a smaller community but is actively developed and handles modern SPAs better. Selenium remains the standard for legacy enterprise test suites. |
| **BeautifulSoup** for parsing | **Parsel** (Scrapy's parser) or **lxml** — significantly faster on large documents. lxml uses C-based parsing. | BeautifulSoup is more beginner-friendly. For production pipelines scraping at scale, lxml or Scrapy's built-in parsing is preferred. |
| **Hardcoded section headings** for navigation | **Structured data extraction** via JSON-LD or Schema.org markup when available, or **LLM-based extraction** (e.g., passing raw text to a language model with a structured output schema). | JSON-LD is the cleanest source when sites expose it. LLM-based extraction is more robust to layout changes but introduces latency and cost. |
| **Single-page script** | **Scrapy framework** — provides request scheduling, middleware, item pipelines, retry logic, rate limiting, and built-in export formats. | Overkill for a single page, but essential for multi-page crawling in production. |

### Exercise 2: Text Preprocessing & Augmentation

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Word2Vec** (2013) for embeddings | **FastText** (2016) — handles out-of-vocabulary words via subword information. Falls back to character n-grams for unknown words. | Word2Vec fails on OOV words entirely. FastText is a direct evolution and is preferred in modern production systems for embedding-based tasks. |
| **nlpaug with Word2Vec** for augmentation | **Back-translation** (translate to another language and back), or **contextual augmentation** with BERT/GPT models via `nlpaug`'s `ContextualWordEmbsAug`. | Contextual augmentation produces more grammatically natural text. Back-translation preserves meaning better but requires API calls or large translation models. |
| **Static embeddings** (same vector regardless of context) | **Contextual embeddings** (ELMo, BERT, GPT) — different vector for each word occurrence based on surrounding context. | Static embeddings cannot distinguish "bank" (financial) from "bank" (river). This is the fundamental limitation that motivated the Transformer revolution. |
| **Manual regex cleaning** | **spaCy pipelines** — provides tokenization, lemmatization, NER, and POS tagging in a single pass with pre-trained models. | spaCy is heavier to install but dramatically reduces boilerplate preprocessing code and handles edge cases better. |

### Exercise 3: Sentiment Analysis

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Hu & Liu lexicon** (2004) | **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — specifically designed for social media text, handles slang, emoticons, capitalization, and degree modifiers. | VADER is the direct upgrade for social media sentiment. It understands that "GREAT!!!" is more positive than "great." The Hu & Liu lexicon has no intensity modeling. |
| **Bag-of-words lexicon matching** | **Transformer-based models** (BERT, RoBERTa fine-tuned for sentiment, or Hugging Face `sentiment-analysis` pipeline) — capture context, negation, sarcasm to a degree. | Transformers require GPU for training and are orders of magnitude more expensive to run. But accuracy on sentiment tasks improves from ~50% to 85%+ on standard benchmarks. |
| **Simple positive/negative/neutral rules** | **Multi-class or aspect-based sentiment** — more granular classification (e.g., 1–5 stars) or per-aspect sentiment (e.g., "food was great but service was slow"). | Rule-based ternary classification is the coarsest possible sentiment system. Production systems typically need finer granularity. |
| **No handling of negation** | **Negation scope detection** — "not good" should flip the sentiment of "good." Libraries like VADER and NLP models handle this natively. | The current lexicon approach counts "good" as positive even in "not good." This is a primary source of misclassification. |
| **scikit-learn evaluation only** | **Confusion matrix visualization**, **per-class precision/recall**, **error analysis** — standard practice for understanding where a classifier fails. | Simple accuracy/F1 gives one number. A confusion matrix reveals whether the model confuses positive→neutral, negative→positive, etc., which guides targeted improvements. |

### Where the Baseline Tech Still Holds Up

Despite the existence of newer alternatives, several technologies used in this assignment remain the right choice in specific contexts:

- **Regex for text cleaning** — remains the fastest, most efficient tool for pattern-based string operations. Even production NLP pipelines use regex as a preprocessing step before feeding text to models. There is no replacement for regex; it is foundational.
- **Lexicon-based sentiment** — remains viable for real-time, low-latency systems where model inference cost is prohibitive (e.g., processing millions of tweets per second). It is also fully explainable, which matters in regulated industries.
- **Word2Vec** — while largely superseded by contextual models for downstream NLP tasks, Word2Vec-style static embeddings are still used in recommendation systems, search engines, and any system where pre-computing fixed vectors is necessary (e.g., approximate nearest neighbor search at scale).
