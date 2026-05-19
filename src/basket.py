from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import pandas as pd


@dataclass
class BasketRecommendation:
    product_id: int
    score: int


class BasketRecommender:
    def __init__(self):
        self.cooccurrence: dict[int, Counter] = defaultdict(Counter)

    def fit(self, df: pd.DataFrame, cart_col: str = "cart_id", item_col: str = "product_id") -> "BasketRecommender":
        grouped = df.groupby(cart_col)[item_col].apply(lambda s: list(dict.fromkeys(s.tolist())))
        for items in grouped:
            for base in items:
                for other in items:
                    if base != other:
                        self.cooccurrence[int(base)][int(other)] += 1
        return self

    def recommend(self, product_id: int, top_k: int = 10) -> list[BasketRecommendation]:
        return [BasketRecommendation(pid, score) for pid, score in self.cooccurrence.get(int(product_id), Counter()).most_common(top_k)]
