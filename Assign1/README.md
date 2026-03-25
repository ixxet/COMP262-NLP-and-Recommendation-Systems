# Assignment 1 – Data Preprocessing and Sentiment Analysis

## Exercises

### Exercise 1: Web Scraping (30%)
Scrapes the Centennial College AI program page to extract the website title, program highlights, companies offering jobs, and career outlook. Results are exported to `Izzet_my_future.csv`.

```bash
python exercise1_web_scraping.py
```

### Exercise 2: Text Preprocessing & Data Augmentation (35%)
Loads `COVID19_mini.csv`, cleans the tweet text, and applies two augmentation techniques using the Word2Vec model:
1. Word embedding augmentation (doubles the dataset)
2. Random insertion augmentation using Word2Vec synonyms

Output: `Izzet_df_after_random_insertion.txt`

```bash
python exercise2_preprocessing.py
```

**Note:** Requires `GoogleNews-vectors-negative300.bin.gz` in this directory (not tracked in git due to size).

### Exercise 3: Sentiment Analysis (35%)
Performs lexicon-based sentiment analysis on `COVID19_data.csv` using positive and negative word lists. Calculates sentiment percentages, predicts sentiment labels, and evaluates with Accuracy and F1 Score.

```bash
python exercise3_sentiment_analysis.py
```

## Dependencies
- requests, beautifulsoup4
- pandas, numpy
- nltk
- nlpaug
- gensim
- scikit-learn
