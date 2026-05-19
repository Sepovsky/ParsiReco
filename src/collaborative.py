from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


RECOMMEND_MAP = {
    "not_recommended": 0.0,
    "no_idea": 1.0,
    "recommended": 2.0,
    "\\N": 1.0,
}


@dataclass
class CollaborativeRecommendation:
    product_id: int
    score: float


class CollaborativeRecommender:
    def __init__(self):
        self.sparse_matrix: csr_matrix | None = None
        self.user_ids: np.ndarray | None = None
        self.product_ids: np.ndarray | None = None
        self.user_to_idx: dict[int, int] = {}
        self.product_to_idx: dict[int, int] = {}

    def fit(self, df: pd.DataFrame, use_sentiment: bool = True) -> "CollaborativeRecommender":
        working = df[["user_id", "product_id", "recommend", *( ["comment_sentiment"] if "comment_sentiment" in df.columns else [] )]].copy()
        working["recommend_numeric"] = working["recommend"].map(RECOMMEND_MAP).fillna(1.0)
        if use_sentiment and "comment_sentiment" in working.columns:
            working["final_rating"] = 0.7 * working["recommend_numeric"] + 0.3 * working["comment_sentiment"].fillna(0.5)
        else:
            working["final_rating"] = working["recommend_numeric"]

        self.user_ids = np.sort(working["user_id"].unique())
        self.product_ids = np.sort(working["product_id"].unique())
        self.user_to_idx = {int(uid): i for i, uid in enumerate(self.user_ids)}
        self.product_to_idx = {int(pid): i for i, pid in enumerate(self.product_ids)}

        row = working["user_id"].map(self.user_to_idx).to_numpy()
        col = working["product_id"].map(self.product_to_idx).to_numpy()
        data = working["final_rating"].astype(float).to_numpy()
        self.sparse_matrix = csr_matrix((data, (row, col)), shape=(len(self.user_ids), len(self.product_ids)))
        return self

    def recommend_for_user(self, user_id: int, top_k: int = 10, neighbor_k: int = 50) -> list[CollaborativeRecommendation]:
        if self.sparse_matrix is None or user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[int(user_id)]
        user_vector = self.sparse_matrix[user_idx]
        similarities = cosine_similarity(user_vector, self.sparse_matrix).flatten()
        similarities[user_idx] = 0.0
        if neighbor_k < len(similarities):
            top_neighbor_idx = np.argpartition(similarities, -neighbor_k)[-neighbor_k:]
        else:
            top_neighbor_idx = np.arange(len(similarities))

        neighbor_scores = similarities[top_neighbor_idx]
        neighbor_matrix = self.sparse_matrix[top_neighbor_idx]
        weighted_item_scores = neighbor_scores @ neighbor_matrix
        weighted_item_scores = np.asarray(weighted_item_scores).ravel()

        seen = set(user_vector.indices.tolist())
        candidates = []
        for item_idx, score in enumerate(weighted_item_scores):
            if item_idx in seen or score <= 0:
                continue
            candidates.append((int(self.product_ids[item_idx]), float(score)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [CollaborativeRecommendation(pid, score) for pid, score in candidates[:top_k]]
