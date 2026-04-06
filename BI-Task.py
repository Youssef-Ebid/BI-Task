import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# ─────────────────────────────────────────
# MEMBER 1 — Data Exploration
# ─────────────────────────────────────────

# Load dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "Fake"
true_df["label"] = "Real"

# Combine into one dataframe
df = pd.concat([fake_df, true_df], ignore_index=True)

# Summarize the dataset structure
df.info()

# First 5 rows
df.head()

# Last 3 rows
df.tail(3)

# Check missing values in all columns
df.isna().sum()

# Check for duplicate rows
df.duplicated()

# Drop duplicate rows
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Fill missing text values with empty string
df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")

# Plot Fake vs Real distribution
df["label"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Fake vs Real News Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# ─────────────────────────────────────────
# MEMBER 2 — Text Preprocessing
# ─────────────────────────────────────────

# Combine title and text into one column for preprocessing
df["content"] = df["title"] + " " + df["text"]

# 1. Punctuation removal
df["content"] = df["content"].apply(
    lambda text: text.translate(str.maketrans("", "", string.punctuation))
)

# 2. Case Folding - convert all text to lowercase
df["content"] = df["content"].str.lower()

# 3. Tokenization - split text into individual words
df["tokens"] = df["content"].apply(word_tokenize)

# 4. Stop word removal - remove common words that add no value
stop_words = set(stopwords.words("english"))
df["tokens"] = df["tokens"].apply(
    lambda tokens: [w for w in tokens if w not in stop_words]
)

# 5. Lemmatization - reduce words to their base dictionary form
lemmatizer = WordNetLemmatizer()
df["tokens"] = df["tokens"].apply(
    lambda tokens: [lemmatizer.lemmatize(w) for w in tokens]
)

# Rejoin tokens into a single clean string
df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

df["clean_text"].head()

# Save preprocessed data for Members 3 & 4
df[["clean_text", "label"]].to_csv("preprocessed_news.csv", index=False)