# Assignment 3 – Recommender Systems

**Course:** COMP 262 – NLP and Recommendation Systems
**Weight:** Due Week 12
**Student:** Izzet Abidi (300898230)

---

## Table of Contents

1. [Overview](#overview)
2. [Exercise Breakdown](#exercise-breakdown)
   - [Exercise 1: Association Rules Cuisine Recommender (50%)](#exercise-1-association-rules-cuisine-recommender-50)
   - [Exercise 2: Content-Based Song Recommender (50%)](#exercise-2-content-based-song-recommender-50)
3. [Runbook](#runbook)
4. [Expected Results](#expected-results)
5. [Topics Learned](#topics-learned)
6. [Definitions and Key Concepts](#definitions-and-key-concepts)
7. [Potential Improvements and Industry Considerations](#potential-improvements-and-industry-considerations)

---

## Overview

This assignment bridges two major recommender system paradigms, completing the course progression from **rule-based text processing** (Assignment 1) through **learned intent classification** (Assignment 2) to **recommendation engines** that surface relevant items from large catalogs.

**Exercise 1** builds a **non-personalized recommender** using association rule mining — the same mathematical foundation behind "customers who bought X also bought Y" systems. It requires no user history; it mines co-occurrence patterns from ingredient lists across cuisines.

**Exercise 2** builds a **content-based recommender** using TF-IDF vectorization and cosine similarity — the same approach used by early Netflix, Spotify, and Amazon to recommend items based on feature similarity rather than user behavior. It directly applies the text preprocessing and vectorization techniques from Assignments 1 and 2 to a recommendation context.

Together, these exercises cover the two foundational recommender approaches that do not require user interaction data, making them deployable from day one without a cold-start problem.

---

## Exercise Breakdown

### Exercise 1: Association Rules Cuisine Recommender (50%)

**Objective:** Build a recommender that suggests top ingredient groups for a given cuisine type using the Apriori algorithm on recipe transaction data.

**What the script does:**

1. **Data loading** — reads `recipes.json` (39,774 recipes, 20 cuisine types) into a pandas DataFrame. Each recipe has an `id`, `cuisine` label, and an `ingredients` list.

2. **Data exploration** — prints:
   - Total number of recipe instances (39,774)
   - Number of unique cuisines (20)
   - A formatted table showing each cuisine type and its recipe count

3. **User input loop** — accepts a cuisine type string. Validates against known cuisines. Prints `"We don't have recommendations for XXX"` for invalid inputs.

4. **Apriori execution** — for a valid cuisine:
   - Filters the DataFrame to only recipes of that cuisine
   - Extracts ingredient lists as transaction sets (each recipe = one transaction)
   - Calculates minimum support as `100 / num_recipes_for_cuisine` (assignment-specified formula)
   - Sets minimum confidence to 0.5
   - Runs the `apyori` implementation of the Apriori algorithm
   - Collects all RelationRecord objects

5. **Results display** — presents:
   - The **most frequent ingredient group** (first RelationRecord, highest support) — this is the set of ingredients that co-occur most often in that cuisine
   - **All association rules with lift > 2** — showing left-hand side items, right-hand side items, confidence, and lift values in a formatted table

6. **Loop continuation** — prompts for another cuisine type until the user types `"exit"`.

**Key design decisions:**

- **Support formula `100/N`** — this is a relative threshold that adapts to cuisine size. Italian cuisine has ~7,838 recipes, so support = 0.0128 (items must appear in 1.28% of Italian recipes). Greek has 1,175, so support = 0.0851 (8.51%). This prevents over-filtering small cuisines and under-filtering large ones.
- **Confidence at 0.5** — requires that when the left-hand side ingredients appear, the right-hand side must appear in at least 50% of those cases. This is a moderate threshold that balances rule quality with quantity.
- **Lift > 2 as the display filter** — lift measures how much more likely the RHS is given the LHS, compared to chance. Lift > 1 means positive association; filtering at > 2 ensures the rules are meaningfully stronger than random co-occurrence.
- **`min_length=2`** — ensures that only rules with at least 2 items are returned (no trivial single-item "rules").

**File manifest:**

| File | Purpose |
|------|---------|
| `Izzet_cuisine_recommender.py` | Exercise 1 script — association rules recommender |
| `recipes.json` | Recipe dataset (39,774 recipes, 20 cuisines) |

---

### Exercise 2: Content-Based Song Recommender (50%)

**Objective:** Build a content-based recommender that suggests the 10 most similar song titles using TF-IDF vectors and cosine similarity on Amazon Digital Music metadata.

**What the script does:**

1. **Data loading** — reads `meta_Digital_Music.json.gz` line by line (JSONL format inside gzip) into a DataFrame named `songs_Izzet`. Contains 74,347 records with 18 columns.

2. **Data exploration** — for each of the 18 columns, reports:
   - Data type
   - Null count
   - Empty count (empty strings `""` and empty lists `[]` — critical distinction since many columns are "non-null" but contain empty values)
   - Usable record count and percentage
   - A written rationale for which columns to keep and which to drop

3. **Column selection** — keeps 4 columns with justification:

   | Column | Why Keep |
   |--------|---------|
   | `title` | Primary identifier; required for input/output matching |
   | `brand` | Artist name; songs by the same artist share stylistic traits |
   | `description` | Textual content about the music; richest semantic feature |
   | `category` | Genre/category tags; directly relevant for similarity |

   Drops 14 columns: `tech1`, `tech2`, `fit`, `similar_item`, `date` (mostly empty); `also_buy`, `also_view` (collaborative signals, not content); `main_cat` (HTML markup); `rank`, `price` (non-content); `asin` (identifier); `feature`, `imageURL`, `imageURLHighRes` (empty or non-textual).

4. **Data cleaning:**
   - Drops rows with null or empty titles (cannot recommend what has no name)
   - Deduplicates on title (keeps first occurrence; avoids redundant similarity scores)
   - Converts list-type columns (`description`, `category`) to joined strings
   - Fills remaining nulls with empty strings

5. **Feature combination** — concatenates `title + brand + description + category` into a single `content` column per song. This is the text that gets vectorized.

6. **Text preprocessing** — applies to each content string:
   - Lowercasing
   - HTML tag removal (some descriptions contain `<br>`, `<b>` tags)
   - Punctuation removal
   - Whitespace normalization

7. **TF-IDF vectorization:**
   - Uses sklearn `TfidfVectorizer` with:
     - `max_features=5000` — caps vocabulary to the 5,000 most informative terms (memory/speed trade-off)
     - `stop_words='english'` — removes common English stopwords
     - `ngram_range=(1, 2)` — captures both single words and two-word phrases (e.g., "classic rock", "holiday music")
   - Produces a sparse matrix of shape `(num_songs, 5000)`

8. **Cosine similarity** — computes the pairwise cosine similarity between all TF-IDF vectors, producing an `(N × N)` dense matrix where entry `[i][j]` is the similarity between song `i` and song `j`.

9. **Persistence** — saves the similarity matrix, title list, and title→index mapping to `similarity_data.pkl`. Subsequent runs load this file directly, skipping the 2-3 minute computation.

10. **Recommender function** — `recommend_songs(title)`:
    - Looks up the title in the index (case-insensitive)
    - Retrieves the similarity row for that song
    - Sorts by similarity descending, excludes the song itself
    - Returns the top 10 titles with their similarity scores

11. **Interactive loop** — accepts song titles from the user, displays top-10 recommendations in a formatted table, and exits on `"exit"`.

**Key design decisions:**

- **Content combination over separate vectorization** — concatenating all text features into one string before TF-IDF is simpler than building separate vectors and weighting them. For this dataset size and use case, the combined approach produces good results with minimal complexity.
- **`max_features=5000`** — with ~74K records, the full vocabulary could be 50K+ terms. Capping at 5,000 keeps the similarity matrix computation feasible in RAM while retaining the most discriminative terms (TF-IDF naturally prioritizes them).
- **Bigram range `(1, 2)`** — captures phrases like "holiday music" or "classic rock" that unigrams alone would miss. Trigrams were excluded to avoid vocabulary explosion with diminishing returns.
- **Precomputed similarity file** — the similarity matrix computation takes 2-3 minutes and produces a large file. Caching it to disk means the interactive recommender starts instantly on repeat runs.
- **Case-insensitive title matching** — users should not need to match exact capitalization to get recommendations.

**File manifest:**

| File | Purpose |
|------|---------|
| `Izzet_songs_recommender.py` | Exercise 2 script — content-based recommender |
| `meta_Digital_Music.json.gz` | Amazon Digital Music metadata (74,347 records) |
| `similarity_data.pkl` | Precomputed similarity matrix (generated on first run) |

---

## Runbook

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies for both exercises
pip install pandas numpy scikit-learn apyori
```

### Exercise 1: Cuisine Recommender

```bash
cd Assign3/

# Run the cuisine recommender
python Izzet_cuisine_recommender.py
```

**Sample interaction:**
```
Enter cuisine type: greek

Apriori parameters:
  Recipes for 'greek': 1175
  Min support: 100 / 1175 = 0.0851
  Min confidence: 0.5

--- Most Frequent Ingredient Group ---
Ingredients: feta cheese crumbles, garlic, olive oil
Support: 0.0936

--- Association Rules with Lift > 2 ---
Items (LHS)                           → Items (RHS)                           Conf   Lift
olive oil, oregano                    → garlic                                0.812  2.341
feta cheese crumbles, lemon juice     → olive oil                             0.714  2.105
...

Enter cuisine type: exit
Exiting. Goodbye!
```

### Exercise 2: Song Recommender

```bash
# First run: builds TF-IDF vectors and similarity matrix (~2-3 minutes)
python Izzet_songs_recommender.py

# Subsequent runs: loads precomputed data instantly
python Izzet_songs_recommender.py
```

**Sample interaction:**
```
Enter song title: The Glory of Love

Top 10 recommendations for: 'The Glory of Love'
Rank   Song Title                                                   Similarity
------------------------------------------------------------------------------
1      The Power of Love                                                0.4521
2      Greatest Love Songs                                              0.3987
3      Love Songs Collection                                            0.3845
...

Enter song title: exit
Exiting. Goodbye!
```

### Troubleshooting

| Issue | Resolution |
|-------|------------|
| `ModuleNotFoundError: No module named 'apyori'` | Install with `pip install apyori`. Not available in conda default channels — use pip. |
| `FileNotFoundError: recipes.json` | Ensure `recipes.json` is in the `Assign3/` directory. The dataset is the "What's Cooking" Kaggle dataset (39,774 recipes). |
| `MemoryError` during cosine similarity | The similarity matrix for ~60K+ songs requires ~28GB RAM as a dense matrix. If memory is limited, reduce the dataset by filtering to records with non-empty descriptions, or use batch computation. |
| Songs recommender takes too long on first run | Expected: 2-3 minutes for TF-IDF + cosine similarity on ~60K records. Subsequent runs load from `similarity_data.pkl` and start instantly. |
| `"We don't have recommendations for XXX"` | Verify exact spelling. The cuisine recommender is case-insensitive but requires exact cuisine names (e.g., `"cajun_creole"` not `"cajun"`). The songs recommender matches against the dataset titles. |
| `similarity_data.pkl` file is very large | Expected: the precomputed similarity matrix for ~60K songs produces a large pickle file. This is the trade-off for instant recommendations at query time. |

---

## Expected Results

### Exercise 1: Cuisine Recommender

**Data exploration output:**
- 39,774 total recipes across 20 cuisine types
- Italian has the most recipes (~7,838), Brazilian the fewest (~467)
- Each recipe contains a variable-length list of ingredient strings

**Apriori output (varies by cuisine):**
- **Greek:** Top group typically includes olive oil, garlic, feta cheese — the foundational Mediterranean trio
- **Italian:** Expect garlic, olive oil, parmesan cheese — the base of most Italian cooking
- **Indian:** Expect cumin, turmeric, garam masala — the core Indian spice palette
- Rules with lift > 2 indicate ingredient pairs that co-occur significantly more than chance would predict. Higher lift = stronger association.
- The number of rules varies by cuisine size: larger cuisines produce more rules, smaller cuisines may produce fewer or none if support thresholds filter aggressively.

### Exercise 2: Song Recommender

**Data exploration output:**
- 74,347 total records, 18 columns
- Many columns are predominantly empty: `tech1`, `tech2`, `fit`, `similar_item`, `date` are >99% empty
- After cleaning: ~60,000-65,000 unique titled songs remain

**Similarity output:**
- TF-IDF matrix shape: approximately (60,000 × 5,000)
- Similarity scores range from 0.0 (completely dissimilar) to 1.0 (identical content)
- Top recommendations typically share artist names, genre keywords, or descriptive phrases
- Songs with sparse metadata (title only, no description) tend to match poorly — their TF-IDF vectors are thin

---

## Topics Learned

### Association Rule Mining
- **Market basket analysis applied to recipes** — treating each recipe as a transaction and each ingredient as an item, exactly as retail stores analyze purchase baskets to find product associations.
- **Support, confidence, and lift as rule quality metrics** — support measures frequency, confidence measures predictability, lift measures the strength of association above random chance. All three must be considered together; a rule with high confidence but low lift may be trivially true.
- **The Apriori algorithm** — an efficient method for finding frequent itemsets by exploiting the downward closure property: if an itemset is infrequent, all its supersets must also be infrequent. This prunes the search space exponentially.

### Content-Based Filtering
- **TF-IDF as a document representation** — converting unstructured text into fixed-dimensional numeric vectors where each dimension represents a term's importance to a document relative to the corpus. This is the bridge between raw text and mathematical similarity computation.
- **Cosine similarity as a distance metric** — measuring the angle between two TF-IDF vectors rather than their magnitude. Two documents about the same topic will point in similar directions regardless of length. This makes it ideal for comparing documents of varying sizes.
- **Feature engineering for recommenders** — the critical decision of which raw data columns to include, how to combine them, and how to preprocess them. The quality of recommendations is bounded by the quality of the feature space.

### Data Engineering
- **Handling messy real-world data** — the Amazon metadata contains HTML in text fields, empty lists masquerading as non-null values, and inconsistent formatting. Distinguishing between null and empty is essential; `isnull()` alone misses empty strings and empty lists.
- **Precomputation and caching** — computing pairwise similarity for 60K+ items is expensive (O(n²)). Persisting the result to disk converts a 3-minute computation into a sub-second file load. This is the fundamental trade-off between compute time and storage in production recommenders.
- **JSONL vs. standard JSON** — the music dataset uses one JSON object per line (JSONL), which allows streaming/line-by-line parsing. The recipes dataset uses a standard JSON array. Recognizing and handling both formats is a practical data engineering skill.

### Recommender System Design
- **Non-personalized vs. personalized recommendations** — Exercise 1 produces the same recommendations for every user asking about Greek cuisine. Exercise 2 produces the same recommendations for every user asking about a specific song. Neither uses user history — they are content-driven and user-agnostic.
- **Cold-start advantage** — both approaches work from day one with zero user interaction data. This is their primary advantage over collaborative filtering, which requires a critical mass of user-item interactions before producing meaningful recommendations.
- **The recommendation presentation layer** — how results are formatted and displayed matters. Showing similarity scores alongside titles gives the user transparency into why each item was recommended.

---

## Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Association Rule** | A relationship of the form {A, B} → {C}, meaning items A and B appearing together implies item C is also likely to appear. Quantified by support, confidence, and lift. |
| **Apriori Algorithm** | An algorithm for mining frequent itemsets from transaction data. Uses the downward closure property (if {A, B} is infrequent, {A, B, C} must also be infrequent) to efficiently prune the search space. |
| **Support** | The fraction of transactions that contain a given itemset. `support({garlic, olive oil}) = count(recipes with both) / total_recipes`. Higher support means the pattern is more common. |
| **Confidence** | The conditional probability that a rule's consequent appears given its antecedent. `confidence(A → B) = support(A ∪ B) / support(A)`. High confidence means B almost always appears when A does. |
| **Lift** | The ratio of observed co-occurrence to expected co-occurrence under independence. `lift(A → B) = confidence(A → B) / support(B)`. Lift > 1 = positive association, lift = 1 = independence, lift < 1 = negative association. |
| **Transaction** | A single record in association rule mining. Here, one recipe = one transaction, and the ingredient list = the items in that transaction. |
| **Frequent Itemset** | A set of items whose support exceeds the minimum support threshold. Only frequent itemsets are candidates for association rule generation. |
| **Downward Closure Property** | The principle that all subsets of a frequent itemset must also be frequent. Enables the Apriori algorithm to prune candidate itemsets without counting them explicitly. |
| **Content-Based Filtering** | A recommendation approach that matches items based on their features (text, metadata, tags) rather than user behavior. Recommends items similar to what the user has shown interest in. |
| **Collaborative Filtering** | A recommendation approach based on user-item interaction patterns ("users who liked X also liked Y"). Not used in this assignment but is the complementary paradigm to content-based filtering. |
| **TF-IDF (Term Frequency–Inverse Document Frequency)** | A weighting scheme that scores a term's importance to a document within a corpus. TF measures how often a term appears in a document; IDF penalizes terms that appear in many documents. The product TF × IDF highlights terms that are frequent locally but rare globally. |
| **Term Frequency (TF)** | The number of times a term appears in a document, often normalized by document length. Captures local importance. |
| **Inverse Document Frequency (IDF)** | `log(N / df)` where N is the total number of documents and df is the number of documents containing the term. Penalizes common terms (e.g., "the", "and") and boosts rare, discriminative terms. |
| **Cosine Similarity** | A measure of similarity between two vectors based on the cosine of the angle between them. `cos(A, B) = (A · B) / (‖A‖ × ‖B‖)`. Ranges from 0 (orthogonal / no similarity) to 1 (identical direction). |
| **Sparse Matrix** | A matrix where most entries are zero. TF-IDF matrices are sparse because each document uses only a small fraction of the total vocabulary. Storing them in sparse format (CSR/CSC) saves memory. |
| **Feature Engineering** | The process of selecting, combining, and transforming raw data columns into features suitable for a model. In this assignment: choosing which metadata columns to include, combining them into text, and preprocessing that text. |
| **Cold-Start Problem** | The challenge of making recommendations for new users (no interaction history) or new items (no ratings). Content-based and association rule approaches avoid this because they do not require user history. |
| **N-gram** | A contiguous sequence of N items from text. Unigrams = single words; bigrams = two-word phrases ("classic rock"); trigrams = three-word phrases. Larger N captures more context but exponentially increases vocabulary size. |
| **Stopwords** | Common words (e.g., "the", "is", "and") that carry little semantic meaning and are removed during preprocessing to reduce noise in the TF-IDF vectors. |
| **Pairwise Similarity Matrix** | An N×N matrix where entry [i][j] stores the similarity between items i and j. Precomputing this matrix enables O(1) lookups at query time but requires O(N²) storage. |
| **JSONL (JSON Lines)** | A file format where each line is a valid JSON object. Unlike standard JSON arrays, JSONL can be parsed line by line without loading the entire file into memory, making it efficient for large datasets. |
| **Pickle** | Python's serialization protocol for converting in-memory objects (matrices, dictionaries, trained models) to byte streams and back. Used here to cache the precomputed similarity matrix to disk. |
| **Non-Personalized Recommender** | A system that generates the same recommendations for all users given the same query. Association rule mining produces non-personalized recommendations: "for Greek cuisine, the top ingredients are X, Y, Z" regardless of who is asking. |

---

## Potential Improvements and Industry Considerations

### Exercise 1: Association Rules

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Apriori algorithm** | **FP-Growth** (Frequent Pattern Growth) — avoids the expensive candidate generation step of Apriori by compressing the dataset into a prefix tree (FP-tree) and mining patterns directly from it. 2-10x faster on large datasets. | Apriori is easier to understand and implement, making it appropriate for learning. FP-Growth is the production standard for datasets beyond 100K transactions. |
| **Single support threshold for all cuisines** | **Adaptive thresholds per cuisine** — cuisines with fewer recipes (Brazilian: 467) get very different Apriori behavior than large cuisines (Italian: 7,838). An adaptive threshold could normalize rule quality across cuisines. | The assignment formula `100/N` already adapts somewhat, but a more sophisticated approach would use statistical significance testing (e.g., chi-squared) rather than fixed support. |
| **Ingredient-level rules only** | **Hierarchical ingredient taxonomy** — mapping specific ingredients to categories (e.g., "parmesan" → "cheese" → "dairy") enables rules at multiple levels of abstraction. Libraries like FoodOn provide standardized food ontologies. | Ingredient-level rules are concrete and actionable. Hierarchical rules reveal broader patterns but require a curated taxonomy, which is a significant data engineering effort. |
| **Static dataset analysis** | **Real-time association mining on streaming data** — as new recipes are added, update the frequent itemsets incrementally rather than rerunning Apriori from scratch. Algorithms: FUP (Fast Update), ZIGZAG. | Batch Apriori is appropriate for this dataset size. Incremental mining matters when the transaction database grows continuously (e.g., e-commerce purchase streams). |

### Exercise 2: Content-Based Filtering

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **TF-IDF vectors** | **Sentence-BERT (SBERT) embeddings** — transformer-based embeddings that capture semantic meaning, not just word overlap. "Happy tunes" and "upbeat songs" would have high similarity with SBERT but low similarity with TF-IDF. | TF-IDF is fast, interpretable, and requires no GPU. SBERT produces superior semantic similarity but needs a pre-trained model (400MB+) and is 100x slower to vectorize. |
| **Cosine similarity on full matrix** | **Approximate Nearest Neighbors (ANN)** — libraries like FAISS, Annoy, or ScaNN find the top-K most similar items in sub-linear time using indexing structures (LSH, HNSW graphs). Essential for catalogs beyond 100K items. | Exact cosine similarity is O(N²) to precompute and O(N) per query. ANN trades a small accuracy loss for 100-1000x speedup. At 60K items the exact approach is manageable; at 10M items it is not. |
| **Dense similarity matrix in memory** | **Sparse retrieval + reranking** — use an inverted index (like Elasticsearch) for initial candidate retrieval, then rerank the top-100 candidates with a more expensive similarity model. This is the standard two-stage architecture at Netflix, Spotify, and YouTube. | The dense matrix approach is simple and works for small catalogs. Two-stage retrieval scales to billions of items with constant query latency. |
| **Text features only** | **Multimodal features** — combine text metadata with audio features (tempo, key, energy, danceability) from APIs like Spotify's Audio Features endpoint. Music similarity is fundamentally about sound, not just text descriptions. | Text-only recommendations work when metadata is rich but fail for songs with sparse descriptions. Audio features capture what text cannot, but require access to the actual audio or a feature API. |
| **Single content column** | **Weighted feature fusion** — assign different weights to title, artist, description, and category. Artist name might deserve 2x weight (same-artist songs are strong recommendations) while category gets 1x. | Equal-weight concatenation is the simplest approach. Weighted fusion requires tuning weights via evaluation metrics, which requires ground-truth relevance labels that this dataset does not have. |

### Deployment and Infrastructure

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| **Pickle file for similarity cache** | **Redis or Memcached** — in-memory key-value stores that serve precomputed recommendations with sub-millisecond latency and support TTL-based cache invalidation. | Pickle is a file-based cache that works for single-machine, single-process scenarios. Redis enables distributed caching, multiple consumers, and automatic expiration for freshness. |
| **Command-line `input()` loop** | **REST API** (FastAPI/Flask) → **Web UI** or **Messaging integration** — decouples the recommendation engine from the presentation layer, enables horizontal scaling. | The `input()` loop is the simplest possible interface. An API enables A/B testing, logging, multiple frontends, and load balancing. |
| **No evaluation metrics** | **Offline evaluation with precision@K, recall@K, NDCG, MAP** — hold out known-relevant items, measure how often the recommender surfaces them in the top-K. **Online evaluation with A/B tests** measuring click-through rate and engagement. | Without evaluation metrics, there is no way to quantify recommendation quality or compare approaches. Implementing offline evaluation requires ground-truth relevance labels. |

### Where the Baseline Tech Still Holds Up

Despite the availability of deep learning recommenders (neural collaborative filtering, autoencoders, transformers), the approaches used in this assignment remain the right choice in several real-world scenarios:

- **Apriori for interpretable rules** — in regulated industries (food safety, pharmaceutical interactions), explainability is non-negotiable. Association rules are fully transparent: "these ingredients co-occur with 85% confidence and 3.2x lift." A neural network cannot explain its recommendations at this level of precision. Retailers, pharmacies, and food service companies still rely on Apriori-style analysis for compliance-sensitive recommendations.

- **TF-IDF for cold-start domains** — when launching a new music platform, book store, or content service, there are zero user interactions. TF-IDF + cosine similarity produces reasonable recommendations from metadata alone, with no training required. Spotify's original recommendation engine was content-based before they had enough user data for collaborative filtering.

- **Precomputed similarity matrices for low-latency serving** — for catalogs under 100K items, a precomputed dense matrix enables O(1) lookups with zero inference-time computation. This is simpler, cheaper, and faster than running a model at query time. Many small-to-medium e-commerce sites use this exact pattern.

- **Cosine similarity as a universal baseline** — before investing in complex recommendation architectures, TF-IDF + cosine similarity establishes a performance baseline. If the baseline already achieves acceptable recommendation quality, the added complexity of embeddings and neural models is not justified. In practice, many content-based systems in production are still TF-IDF at their core, with domain-specific enhancements layered on top.

- **Association rules for physical retail** — brick-and-mortar stores use association rules for shelf placement, cross-promotion, and inventory co-location. The rules map directly to physical actions: "place olive oil near garlic and feta" is immediately actionable. Neural recommendations require a digital interface to present; association rules do not.
