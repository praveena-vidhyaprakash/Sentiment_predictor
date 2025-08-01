import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load real data if exists, else synthesize toy examples
if os.path.exists("reviews.csv"):
    df = pd.read_csv("reviews.csv")  # columns: review_text, star_rating (optional), sentiment (0/1)
else:
    positive = [
        "Excellent product, loved it!",
        "Very satisfied with the service.",
        "Amazing quality and fast delivery.",
        "Best purchase ever, highly recommend!",
        "Works perfectly, I am happy."
    ]
    negative = [
        "Terrible experience, broke immediately.",
        "Not worth the money.",
        "Very disappointed, poor quality.",
        "Worst service, will not buy again.",
        "It stopped working after one day."
    ]
    rows = []
    for t in positive * 20:
        rows.append({"review_text": t, "star_rating": 5, "sentiment": 1})
    for t in negative * 20:
        rows.append({"review_text": t, "star_rating": 1, "sentiment": 0})
    df = pd.DataFrame(rows)

# Ensure star_rating exists
if 'star_rating' not in df.columns:
    df['star_rating'] = 3

# Basic feature: review length (word count)
df['review_len'] = df['review_text'].str.split().apply(len)

X = df[['review_text', 'star_rating', 'review_len']]
y = df['sentiment']

# Preprocessing: TF-IDF on text, scale numeric
text_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500))
])
preprocessor = ColumnTransformer([
    ("text", text_pipe, "review_text"),
    ("num", StandardScaler(), ["star_rating", "review_len"])
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("Classification report:")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.3f}")
except Exception:
    pass

# Save
joblib.dump(pipeline, "sentiment_pipeline.pkl")
print("Saved pipeline to sentiment_pipeline.pkl")
