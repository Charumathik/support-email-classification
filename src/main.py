"""
Support Emails Classification
- Loads data from data/Sample_Support_Emails_Dataset.csv
- Creates rule-based labels (Billing/Login/Integration/General)
- Trains TF-IDF + Logistic Regression
- Prints a classification report
- Saves predictions to data/predicted_support_emails.csv
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = "data/Sample_Support_Emails_Dataset.csv"
OUTPUT_PATH = "data/predicted_support_emails.csv"
RANDOM_STATE = 42

def categorize(text: str) -> str:
    """Simple rule-based categorizer using keywords."""
    t = text.lower()
    if re.search(r"\b(billing|pricing|invoice)\b", t):
        return "Billing Issue"
    if re.search(r"\b(login|password|account|access)\b", t):
        return "Login Issue"
    if re.search(r"\b(integration|api|crm)\b", t):
        return "Integration Query"
    return "General Query"

def main():
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # Basic column expectations
    required_cols = {"subject", "body"}
    if not required_cols.issubset({c.lower() for c in df.columns}):
        raise ValueError(
            f"CSV must include columns: {sorted(list(required_cols))}. "
            f"Found: {list(df.columns)}"
        )

    # Normalize column names (in case of capitalization)
    # Access with lower-case names consistently
    cols = {c.lower(): c for c in df.columns}
    subject_col = cols["subject"]
    body_col = cols["body"]

    # 2) Combine subject + body
    df["text"] = df[subject_col].fillna("") + " " + df[body_col].fillna("")

    # 3) Create rule-based categories (acts like labels for this small sample)
    df["category"] = df["text"].apply(categorize)

    print("Category distribution:")
    print(df["category"].value_counts())

    # 4) Vectorize
    X = df["text"]
    y = df["category"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X_vec = vectorizer.fit_transform(X)

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # 6) Train model
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # 7) Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 8) Save predictions on full data
    df["predicted_category"] = model.predict(X_vec)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
