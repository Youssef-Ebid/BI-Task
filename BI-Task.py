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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


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

# Initialize and learn the model LogisticRegression
print("\n--- Training Logistic Regression Model ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred_lr = lr_model.predict(X_test_tfidf)

# Evaluate the model performance
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Logistic Regression F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print("\nDetailed Classification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))

# Initialize and train Random Forest Classifier
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)

# Evaluate the model performance
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Random Forest F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))

# Final Model Comparison and Selection
print("\n--- Model Comparison ---")

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"Logistic Regression: Acc={lr_acc:.4f}, F1={lr_f1:.4f}")
print(f"Random Forest: Acc={rf_acc:.4f}, F1={rf_f1:.4f}")

if (rf_acc + rf_f1) > (lr_acc + lr_f1):
    print("Best Model: Random Forest (based on Accuracy and F1-Score)")
else:
    print("Best Model: Logistic Regression (based on Accuracy and F1-Score)")