from pathlib import Path

import joblib
import pandas as pd


class ModelNotReadyError(RuntimeError):
    pass


class ArticleNotFoundError(ValueError):
    pass


class RecommenderArtifacts:
    def __init__(self, articles, vectorizer, knn):
        self.articles = articles
        self.vectorizer = vectorizer
        self.knn = knn


def load_artifacts(artifacts_dir: Path) -> RecommenderArtifacts:
    articles_path = artifacts_dir / "articles.parquet"
    vec_path = artifacts_dir / "tfidf.joblib"
    knn_path = artifacts_dir / "knn.joblib"
    if not (articles_path.exists() and vec_path.exists() and knn_path.exists()):
        raise ModelNotReadyError("Model artifacts not found. Run: scripts/train_tfidf.py")
    articles = pd.read_parquet(articles_path)
    vectorizer = joblib.load(vec_path)
    knn = joblib.load(knn_path)
    return RecommenderArtifacts(articles, vectorizer, knn)


def recommend_by_index(art: RecommenderArtifacts, idx: int, k: int) -> list[dict]:
    articles = art.articles
    if idx < 0 or idx >= len(articles):
        raise ArticleNotFoundError("Article index out of range.")
    X = art.vectorizer.transform(articles["full_text"].values)
    dists, inds = art.knn.kneighbors(X[idx], n_neighbors=min(k + 1, len(articles)))
    out = []
    for i in inds[0]:
        if i == idx:
            continue
        rec_title = str(articles.iloc[i]["Title"])
        out.append({"idx": int(i), "title": rec_title, "reason": "Textually similar article"})
        if len(out) >= k:
            break
    return out
