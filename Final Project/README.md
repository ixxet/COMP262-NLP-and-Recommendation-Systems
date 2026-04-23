# Final Project - Amazon Gift Cards Sentiment and Recommendation

**Course:** COMP 262 - NLP and Recommendation Systems  
**Team:** Group 5  
**Assigned Dataset:** Amazon Gift Cards  
**Primary Deliverable:** `project262_gr5_ph2_code.ipynb`<br>
**Product Demo:** FastAPI + SvelteKit app scaffold in `backend/` and `frontend/`

---

## Table of Contents

1. [Overview](#overview)
2. [Project Breakdown](#project-breakdown)
3. [Runbook](#runbook)
4. [Expected Results](#expected-results)
5. [Topics Learned](#topics-learned)
6. [Definitions and Key Concepts](#definitions-and-key-concepts)
7. [Potential Improvements and Industry Considerations](#potential-improvements-and-industry-considerations)

---

## Overview

This final project combines the course's two major tracks: sentiment analysis and recommendation systems. The assigned Group 5 dataset is **Amazon Gift Cards**, taken from the Amazon Review Data 2018 collection published by Jianmo Ni and collaborators.

The project uses customer review text to solve four connected problems:

1. Build lexicon-based sentiment classifiers with **VADER** and **SentiWordNet**.
2. Build machine learning sentiment classifiers with **TF-IDF**, **Logistic Regression**, and **Multinomial Naive Bayes**.
3. Compare lexicon and machine learning models on the same held-out test set.
4. Use review sentiment to enhance rating-based recommendation scores, then test whether the enhancement improves prediction error.

The implementation is intentionally notebook-first because the course deliverable requires working code, visible charts, model outputs, and presentation-ready results. The upgraded app layer wraps the notebook results in a presentation-grade FastAPI/SvelteKit demo without retraining models on every request.

---

## Project Breakdown

### Phase 1: Lexicon Sentiment Analysis

**Objective:** Use the Amazon Gift Cards 5-core review subset to build and evaluate two lexicon-based sentiment classifiers.

**What the notebook does:**

1. Downloads `Gift_Cards_5.json.gz` if it is missing locally.
2. Loads gzipped JSON-line reviews with schema checks.
3. Cleans missing text, missing ratings, empty rows, and duplicate reviews.
4. Labels star ratings into sentiment classes:
   - 4-5 stars: `positive`
   - 3 stars: `neutral`
   - 1-2 stars: `negative`
5. Randomly samples exactly 1,000 cleaned reviews using a fixed seed.
6. Runs exploratory analysis:
   - rating distribution
   - sentiment label distribution
   - review length distribution
   - reviews per user
   - reviews per product
   - date distribution
   - duplicate and missing-value checks
7. Builds two lexicon models:
   - **VADER**, using raw-ish text because punctuation, capitalization, negation, and intensifiers matter.
   - **SentiWordNet**, using tokenization, lowercasing, stopword removal, POS tagging, and lemmatization.
8. Evaluates both models with accuracy, precision, recall, F1, and confusion matrices.

**Key design decision:** The two lexicons do not use identical preprocessing. VADER is designed for surface cues such as punctuation and capitalization, while SentiWordNet depends on WordNet-style lexical normalization. Treating both models the same would be simpler but technically weaker.

---

### Phase 2: Machine Learning Sentiment Analysis

**Objective:** Use the larger Gift Cards dataset to train and compare supervised machine learning classifiers.

**What the notebook does:**

1. Downloads `Gift_Cards.json.gz` if it is missing locally.
2. Repeats cleaning and EDA on the larger dataset.
3. Uses a stratified modeling subset larger than the 2,000-review project minimum.
4. Splits data 70/30 with stratification by sentiment label.
5. Represents review text with **TF-IDF**.
6. Tunes and trains:
   - Logistic Regression
   - Multinomial Naive Bayes
7. Compares both ML models to VADER and SentiWordNet on the exact same test set.

**Key design decision:** The dataset is heavily skewed toward 5-star reviews. For that reason, the notebook reports macro-F1 and confusion matrices instead of relying only on accuracy. Accuracy rewards majority-class prediction too heavily on this dataset.

---

### Recommender Enhancement

**Objective:** Implement one review-based rating enhancement strategy inspired by the supplied recommender systems paper.

**Implemented approach:** Overall review opinion is converted into a virtual rating and blended with the original star rating.

```text
sentiment_virtual_rating = clip(3 + 2 * VADER_compound_score, 1, 5)
enhanced_rating = 0.7 * actual_rating + 0.3 * sentiment_virtual_rating
```

The notebook compares:

- baseline item-average rating prediction
- sentiment-enhanced item-average rating prediction

using MAE and RMSE.

**Result interpretation:** The sentiment-enhanced rating did not improve MAE/RMSE on this dataset. That is a useful finding, not a failed implementation. Gift Cards reviews are extremely positive, so sentiment often repeats information already captured by star ratings and can add noise instead of useful independent signal.

---

### Local Hugging Face LLM Tasks

**Objective:** Satisfy the project requirement to host Hugging Face models locally.

The notebook uses:

| Task | Local model |
|------|-------------|
| Review summarization | `sshleifer/distilbart-cnn-12-6` |
| Service-response drafting | `google/flan-t5-small` |

The LLM section:

1. Selects 10 reviews longer than 100 words.
2. Summarizes each review to roughly 50 words.
3. Selects one question-style review.
4. Generates a customer-service-style response.
5. Applies a guardrail when the small local response model produces an unusably short draft.

---

## Runbook

### Prerequisites

```bash
python --version
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch jupyter
```

The notebook downloads required NLTK resources automatically:

- `vader_lexicon`
- `sentiwordnet`
- `wordnet`
- `omw-1.4`
- POS tagger resources

### Run the Notebook

```bash
cd "Final Project"
jupyter notebook project262_gr5_ph2_code.ipynb
```

To execute from the command line:

```bash
cd "Final Project"
RUN_LLM=1 jupyter nbconvert --to notebook --execute --inplace project262_gr5_ph2_code.ipynb --ExecutePreprocessor.timeout=3600
```

If the machine cannot run or download Hugging Face models:

```bash
RUN_LLM=0 jupyter nbconvert --to notebook --execute --inplace project262_gr5_ph2_code.ipynb --ExecutePreprocessor.timeout=3600
```

### Data Handling

The raw dataset files are intentionally ignored by git:

```text
Final Project/data/raw/Gift_Cards_5.json.gz
Final Project/data/raw/Gift_Cards.json.gz
```

The notebook downloads them from the UCSD Amazon review data host when missing.

### Run the Product Demo API

```bash
cd "Final Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn giftcard_sentiment.api.main:app --app-dir backend --reload
```

Useful endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | API health check |
| `GET /readyz` | artifact readiness check |
| `GET /v1/summary` | notebook-derived dataset/model/recommender metrics |
| `POST /v1/sentiment/predict` | live demo sentiment scoring |
| `POST /v1/recommend/enhance` | rating + sentiment blend calculator |
| `GET /v1/examples/summaries` | local Hugging Face example outputs |
| `POST /v1/assistant/ask` | evidence retrieval over project findings |
| `GET /metrics` | Prometheus metrics |

### Run the SvelteKit UI

```bash
cd "Final Project/frontend"
npm install
npm run dev
```

The UI proxies `/api/...` requests to `http://localhost:8000` by default. Override with:

```bash
API_BASE_URL=http://localhost:8000 npm run dev
```

### Container and Kubernetes Shape

The app mirrors the Deep Learning final project's deployment style:

- Python package with `pyproject.toml`
- FastAPI backend Dockerfile
- SvelteKit frontend with adapter-node and Dockerfile
- Kustomize base and Talos overlay
- Prometheus scrape path through `/metrics`
- GitHub Actions workflow that tests backend, frontend, Kustomize, and builds GHCR images

Local image build:

```bash
cd "Final Project"
docker build -f backend/Dockerfile -t giftcard-sentiment-api:local .
docker build -f frontend/Dockerfile -t giftcard-sentiment-ui:local .
```

Kustomize validation:

```bash
cd "Final Project"
kustomize build k8s/overlays/talos
```

---

## Expected Results

The exact values can change if model parameters or sample size change, but the current executed notebook shows:

| Model | Main Observation |
|------|------------------|
| VADER | Stronger than SentiWordNet as a lightweight lexicon baseline |
| SentiWordNet | More academic, but weaker on short informal product reviews |
| Logistic Regression | Best macro-F1 among the tested models |
| Naive Bayes | Highest accuracy, but less balanced across minority classes |
| Sentiment-enhanced recommender | Did not beat item-average baseline on MAE/RMSE |

The main dataset caveat is class imbalance: Amazon Gift Cards reviews are overwhelmingly positive.

---

## Topics Learned

- JSONL and gzip data loading
- Dataset cleaning and schema validation
- Rating-to-sentiment label creation
- Class imbalance analysis
- Lexicon-based sentiment classification
- WordNet-based sentiment scoring
- TF-IDF feature representation
- Stratified train/test splitting
- Classical text classification with Logistic Regression and Naive Bayes
- Apples-to-apples model comparison
- Review-enhanced recommendation scoring
- Local Hugging Face model inference
- Reproducible notebook verification

---

## Definitions and Key Concepts

| Concept | Meaning |
|---------|---------|
| 5-core dataset | A filtered subset where users/items meet minimum interaction counts |
| Lexicon sentiment | Sentiment classification using predefined word or phrase sentiment scores |
| VADER | A rule-based sentiment model tuned for social and review-style text |
| SentiWordNet | A lexical resource that assigns positive and negative scores to WordNet synsets |
| TF-IDF | Text representation that weights words by frequency and rarity across documents |
| Stratified split | Train/test split that preserves label proportions |
| Macro-F1 | F1 averaged equally across classes, useful for imbalanced data |
| Weighted-F1 | F1 weighted by class frequency, useful but majority-class sensitive |
| Virtual rating | A rating inferred from review sentiment rather than directly provided by the user |
| MAE/RMSE | Regression metrics used to measure rating prediction error |

---

## Potential Improvements and Industry Considerations

The current notebook is a correct academic deliverable. A stronger productized version would add:

1. Persist exact trained scikit-learn model artifacts from the notebook with `joblib`.
2. Add a reverse proxy or route so the SvelteKit UI and API share one public origin in Kubernetes.
3. Replace the lightweight retrieval endpoint with a Qdrant-backed evidence index.
4. Add LangGraph once the project assistant needs a real workflow, such as retrieve -> draft -> verify citations -> revise.
5. Add Grafana dashboard JSON for request counts, latency, sentiment labels, and project-assistant questions.
6. Add report and slide PDF generation from committed markdown sources.

The best next upgrade is a small app wrapper around the completed notebook results, not a large agent system. LangGraph becomes worthwhile only if it retrieves and explains project evidence, rather than acting as a generic chatbot.
