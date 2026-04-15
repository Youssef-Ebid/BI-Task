import pandas as pd
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = "Fake"
true_df["label"] = "Real"
df = pd.concat([fake_df, true_df], ignore_index=True)

df.info()

df = df.drop_duplicates()
df = df.reset_index(drop=True)

df.isna().sum()

df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")

df["label"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Fake vs Real News Distribution")
plt.ylabel("Count")
plt.show()

df["content"] = df["title"] + " " + df["text"]

df["content"] = df["content"].apply(lambda text: text.translate(str.maketrans("", "", string.punctuation)))

df["content"] = df["content"].str.lower()

df["tokens"] = df["content"].apply(word_tokenize)

stop_words = set(stopwords.words("english"))
df["tokens"] = df["tokens"].apply(lambda tokens: [w for w in tokens if w not in stop_words])

lemmatizer = WordNetLemmatizer()
df["tokens"] = df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])

df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

encoder = LabelEncoder()
df["label_num"] = encoder.fit_transform(df["label"])

X = df["clean_text"]
y = df["label_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# BoW
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
print("BoW shape:", X_train_bow.shape)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TF-IDF shape:", X_train_tfidf.shape)

df["clean_text"].head()

df[["clean_text", "label"]].to_csv("preprocessed_news.csv", index=False)

# ------Logistic Regression------ #
print("\n--- Logistic Regression Model ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print(f"Accuracy: {lr_acc:.4f}")
print(f"F1-Score: {lr_f1:.4f}")

# ------Random Forest------ #
print("\n--- Random Forest Model ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"Accuracy: {rf_acc:.4f}")
print(f"F1-Score: {rf_f1:.4f}")

print("\n---- Model Comparison ----")
print(f"Logistic Regression: Accuracy={lr_acc:.4f}, f1_score={lr_f1:.4f}")
print(f"Random Forest: Accuracy={rf_acc:.4f}, f1_score={rf_f1:.4f}")

if (rf_acc + rf_f1) > (lr_acc + lr_f1):
    print("Best Model: Random Forest")
else:
    print("Best Model: Logistic Regression")