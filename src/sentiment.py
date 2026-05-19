from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_utils import load_stopwords, preprocess_text


@dataclass
class SentimentArtifacts:
    model: Pipeline
    train_f1: float
    test_f1: float
    report: str


class PersianSentimentClassifier:
    def __init__(self, stop_words_path: str | Path):
        self.stop_words = load_stopwords(stop_words_path)
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=lambda text: preprocess_text(text, self.stop_words),
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1200)),
        ])

    def fit(self, df: pd.DataFrame, text_col: str = "Text", score_col: str = "Score") -> SentimentArtifacts:
        working = df[[text_col, score_col]].dropna().copy()
        working[score_col] = (working[score_col] / 10).astype(int)
        working = working[(working[score_col] < 6) | (working[score_col] > 7)].copy()
        working["label"] = (working[score_col] > 7).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(
            working[text_col],
            working["label"],
            test_size=0.2,
            random_state=42,
            stratify=working["label"],
        )
        self.pipeline.fit(x_train, y_train)

        train_pred = self.pipeline.predict(x_train)
        test_pred = self.pipeline.predict(x_test)
        return SentimentArtifacts(
            model=self.pipeline,
            train_f1=f1_score(y_train, train_pred),
            test_f1=f1_score(y_test, test_pred),
            report=classification_report(y_test, test_pred),
        )

    def predict(self, texts: pd.Series) -> pd.Series:
        preds = self.pipeline.predict(texts.fillna(""))
        return pd.Series(preds, index=texts.index, name="comment_sentiment")
