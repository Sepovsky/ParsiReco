from __future__ import annotations

from data_utils import DataPaths, load_products, load_raw_sentiment_reviews, load_sentiment_products
from sentiment import PersianSentimentClassifier
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from basket import BasketRecommender


def main() -> None:
    sentiment_reviews = load_raw_sentiment_reviews()
    products = load_products()
    sentiment_products = load_sentiment_products()

    sentiment_model = PersianSentimentClassifier(DataPaths.stop_words)
    artifacts = sentiment_model.fit(sentiment_reviews)
    print("Sentiment model")
    print(f"Train F1: {artifacts.train_f1:.3f}")
    print(f"Test F1:  {artifacts.test_f1:.3f}\n")

    content_model = ContentBasedRecommender(DataPaths.stop_words).fit(products)
    sample_product_id = int(products.iloc[0]["product_id"])
    print("Content-based recommendations")
    for rec in content_model.recommend(sample_product_id, top_k=5):
        print(f"- {rec.product_id} | {rec.product_title} | score={rec.score:.3f}")

    collab_model = CollaborativeRecommender().fit(sentiment_products)
    sample_user_id = int(sentiment_products.iloc[0]["user_id"])
    print("\nCollaborative recommendations")
    for rec in collab_model.recommend_for_user(sample_user_id, top_k=5):
        print(f"- product={rec.product_id} | score={rec.score:.3f}")

    basket_model = BasketRecommender().fit(products)
    print("\nBasket recommendations")
    for rec in basket_model.recommend(sample_product_id, top_k=5):
        print(f"- product={rec.product_id} | cooccurrence={rec.score}")


if __name__ == "__main__":
    main()
