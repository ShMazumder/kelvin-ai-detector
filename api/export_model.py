"""
Export ML Model for API Deployment

Runs the ML training pipeline from the notebook and saves pickled model
files to api/model/ for the API to load on startup.

Requirements:
  - scikit-learn, numpy, scipy, pandas
  - Either a labeled dataset (CSV with 'text' + 'generated' columns)
    or falls back to synthetic demo data

Usage:
  python api/export_model.py                          # synthetic data (demo)
  python api/export_model.py /path/to/train.csv       # real labeled data
  python api/export_model.py /path/to/train.csv text generated  # custom column names
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# Add api/ to path so we can import detector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import (
    PATTERN_WEIGHTS,
    heuristic_score,
    extract_all_patterns,
    HeuristicFeatureTransformer,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# ── Synthetic fallback dataset ────────────────────────────────────────────────

def build_synthetic_dataset(n_per_class=120, seed=42):
    rng = np.random.default_rng(seed)

    ai_templates = [
        "It is important to note that {topic} plays a pivotal role in fostering {benefit}.",
        "Our {topic} boasts a robust, seamless, and intuitive {benefit}, serving as a testament to excellence.",
        "From {a} to {b}, {topic} continues to underscore its crucial significance.",
        "In summary, {topic} represents a transformative and holistic approach to {benefit}.",
        "It's not just {a}, it's {b} -- a true testament to innovation and progress.",
    ]
    human_templates = [
        "I tried {topic} yesterday and honestly it was kind of a mess at first.",
        "So {topic} broke again last week, took me forever to figure out why.",
        "Not gonna lie, {topic} is way harder than people make it sound.",
        "We talked about {topic} for like twenty minutes and never actually decided anything.",
        "{topic} works fine most days but the {benefit} thing is still buggy.",
    ]
    topics = [
        "the platform", "our workflow", "the new feature", "this process",
        "the dataset", "the app", "the system", "our pipeline",
    ]
    benefits = ["engagement", "growth", "collaboration", "efficiency", "reliability"]

    def fill(t):
        return t.format(
            topic=rng.choice(topics), benefit=rng.choice(benefits),
            a=rng.choice(topics), b=rng.choice(benefits),
        )

    ai_texts = [
        fill(rng.choice(ai_templates)) + " " + fill(rng.choice(ai_templates))
        for _ in range(n_per_class)
    ]
    human_texts = [
        fill(rng.choice(human_templates)) + " " + fill(rng.choice(human_templates))
        for _ in range(n_per_class)
    ]

    df = pd.DataFrame({
        "text": ai_texts + human_texts,
        "label": [1] * len(ai_texts) + [0] * len(human_texts),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse args
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    text_col = sys.argv[2] if len(sys.argv) > 2 else "text"
    label_col = sys.argv[3] if len(sys.argv) > 3 else "generated"

    # Load data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df = df.dropna()
        print(f"Loaded REAL dataset: {len(df)} rows from {data_path}")
        using_real = True
    else:
        if data_path:
            print(f"File not found: {data_path}")
        print("Using SYNTHETIC demo dataset (not for production!)")
        df = build_synthetic_dataset()
        using_real = False

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"],
    )

    # TF-IDF
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Heuristic features
    print("Extracting heuristic features...")
    heuristic_tf = HeuristicFeatureTransformer()
    X_train_heur = heuristic_tf.transform(X_train)
    X_test_heur = heuristic_tf.transform(X_test)

    # Combine
    X_train_combined = hstack([X_train_tfidf, X_train_heur])
    X_test_combined = hstack([X_test_tfidf, X_test_heur])

    # Train
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_combined, y_train)

    # Evaluate
    preds = clf.predict(X_test_combined)
    probs = clf.predict_proba(X_test_combined)[:, 1]

    print(f"\nTrained on {'REAL' if using_real else 'SYNTHETIC (demo only)'} data")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    try:
        print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")
    except ValueError:
        pass
    print()
    print(classification_report(y_test, preds, target_names=["human", "ai"]))

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)

    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    heur_path = os.path.join(MODEL_DIR, "heuristic_transformer.pkl")

    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf, f)
    with open(heur_path, "wb") as f:
        pickle.dump(heuristic_tf, f)

    print(f"\nModel files saved to {MODEL_DIR}/")
    print(f"  - {clf_path}")
    print(f"  - {vec_path}")
    print(f"  - {heur_path}")
    print("\nRestart the API server to load the ML model.")


if __name__ == "__main__":
    main()
