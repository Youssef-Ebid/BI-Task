import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# — Data Exploration

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

# — Text Preprocessing

# Combine title and text into one column for preprocessing
df["content"] = df["title"] + " " + df["text"]

# 1. Punctuation removal
df["content"] = df["content"].apply(lambda text: text.translate(str.maketrans("", "", string.punctuation)))

# 2. Case Folding - convert all text to lowercase
df["content"] = df["content"].str.lower()

# 3. Tokenization - split text into individual words
df["tokens"] = df["content"].apply(word_tokenize)

# 4. Stop word removal - remove common words that add no value
stop_words = set(stopwords.words("english"))
df["tokens"] = df["tokens"].apply(lambda tokens: [w for w in tokens if w not in stop_words])

# 5. Lemmatization - reduce words to their base dictionary form
lemmatizer = WordNetLemmatizer()
df["tokens"] = df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])

# Rejoin tokens into a single clean string
df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

# Encode Labels
encoder = LabelEncoder()
df["label_num"] = encoder.fit_transform(df["label"])

# Train/Test Split
X = df["clean_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Bag of Words (BoW)
bow_vectorizer = CountVectorizer(max_features=5000)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("BoW shape:", X_train_bow.shape)
print("TF-IDF shape:", X_train_tfidf.shape)

df["clean_text"].head()

# Save preprocessed data
df[["clean_text", "label"]].to_csv("preprocessed_news.csv", index=False)

