from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class DataPaths:
    raw_reviews = DATA_DIR / "data.csv"
    products = DATA_DIR / "PreprocessedDataset2.csv"
    sentiment_products = DATA_DIR / "SentimentDataset.csv"
    stop_words = DATA_DIR / "fa_stop_words.txt"


REQUIRED_PRODUCT_COLUMNS = [
    "product_id", "product_title", "user_id", "recommend", "title", "comment",
    "advantages", "disadvantages", "cart_id"
]


def load_raw_sentiment_reviews() -> pd.DataFrame:
    return pd.read_csv(DataPaths.raw_reviews)



def load_products() -> pd.DataFrame:
    df = pd.read_csv(DataPaths.products)
    missing = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df



def load_sentiment_products() -> pd.DataFrame:
    return pd.read_csv(DataPaths.sentiment_products)
