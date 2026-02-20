from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Find the AG News CSV file
    csv_candidates = list(data_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No CSV file found in data/. Please check your dataset download.")
    csv_path = csv_candidates[0]

    # Load the dataset
    df = pd.read_csv(csv_path)
    # Try to find the text columns (title, description, etc.)
    text_cols = [col for col in df.columns if col.lower() in ("title", "description", "text")]
    if not text_cols:
        raise ValueError("No suitable text columns found in the dataset.")
    # Combine text columns for similarity
    df["full_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(df["full_text"].values)

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn.fit(X)

    # Save artifacts
    df.to_parquet(artifacts_dir / "articles.parquet", index=False)
    joblib.dump(vectorizer, artifacts_dir / "tfidf.joblib")
    joblib.dump(knn, artifacts_dir / "knn.joblib")

    # Print a sample recommendation
    sample_idx = 0
    dists, inds = knn.kneighbors(X[sample_idx], n_neighbors=4)
    print("Sample input:", df.iloc[sample_idx][text_cols[0]])
    print("Top 3 similar articles:")
    for i in inds[0][1:]:
        print("-", df.iloc[i][text_cols[0]])

    print("\nSaved artifacts:")
    print(artifacts_dir / "articles.parquet")
    print(artifacts_dir / "tfidf.joblib")
    print(artifacts_dir / "knn.joblib")


if __name__ == "__main__":
    main()
