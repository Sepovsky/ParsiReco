from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from text_utils import preprocess_text, load_stopwords


@dataclass
class ContentRecommendation:
    product_id: int
    product_title: str
    score: float


class ContentBasedRecommender:
    def __init__(self, stop_words_path: str):
        self.stopwords = load_stopwords(stop_words_path)
        self.vectorizer = TfidfVectorizer(
            preprocessor=lambda text: preprocess_text(text, self.stopwords),
            ngram_range=(1, 2),
            min_df=2,
            max_features=40000,
        )
        self.df: pd.DataFrame | None = None
        self.tfidf = None
        self.product_to_index: dict[int, int] = {}

    def build_corpus(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["comment"].fillna("") + " " +
            df["advantages"].fillna("") + " " +
            df["disadvantages"].fillna("") + " " +
            df["title"].fillna("") + " " +
            df["product_title"].fillna("")
        )

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        dedup = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True).copy()
        dedup["content_text"] = self.build_corpus(dedup)
        self.tfidf = self.vectorizer.fit_transform(dedup["content_text"])
        self.df = dedup
        self.product_to_index = {int(pid): idx for idx, pid in enumerate(dedup["product_id"])}
        return self

    def recommend(self, product_id: int, top_k: int = 10) -> list[ContentRecommendation]:
        if self.df is None or self.tfidf is None:
            raise RuntimeError("Model has not been fit yet")
        if product_id not in self.product_to_index:
            return []
        idx = self.product_to_index[product_id]
        cosine_scores = linear_kernel(self.tfidf[idx], self.tfidf).flatten()
        top_idx = cosine_scores.argsort()[::-1]
        recs: list[ContentRecommendation] = []
        for i in top_idx:
            if i == idx:
                continue
            row = self.df.iloc[i]
            recs.append(ContentRecommendation(int(row["product_id"]), row["product_title"], float(cosine_scores[i])))
            if len(recs) >= top_k:
                break
        return recs
