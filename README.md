# ParsiReco

ParsiReco is a **Persian-language product recommender system** built from product reviews, comments, and interaction data collected from an **Amazon-like e-commerce platform**. The project combines **sentiment analysis**, **content-based recommendation**, **collaborative filtering**, and **basket-based recommendation** to generate relevant product suggestions for users.

The goal of the project is to explore how multiple recommendation strategies can be applied to **Persian e-commerce data**, where both **textual product information** and **user behavior** play an important role in recommendation quality.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Features](#features)
- [System Components](#system-components)
  - [1. Sentiment Analysis](#1-sentiment-analysis)
  - [2. Content-Based Recommendation](#2-content-based-recommendation)
  - [3. Collaborative Filtering](#3-collaborative-filtering)
  - [4. Basket-Based Recommendation](#4-basket-based-recommendation)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Example Use Cases](#example-use-cases)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [Summary](#summary)

---

## Overview

Online shopping platforms rely heavily on recommendation systems to improve user experience and help customers discover relevant products. In Persian-language e-commerce environments, recommendation becomes more challenging because the system must handle **Persian text**, **user-generated reviews**, and **interaction patterns** at the same time.

ParsiReco addresses this by combining several recommendation approaches:

- **Sentiment analysis** to understand user opinion from Persian comments
- **Content-based filtering** to recommend similar products using textual features
- **Collaborative filtering** to generate personalized suggestions from user behavior
- **Basket-based recommendation** to identify products that commonly appear together

By integrating these components, the project demonstrates how recommendation methods can be adapted to a **Persian-language product platform**.

---

## Objectives

The main objectives of ParsiReco are:

- to analyze **Persian product reviews and comments**
- to build recommendation models using both **text-based** and **behavior-based** signals
- to explore different recommendation strategies within one project
- to improve product suggestion quality by combining complementary recommendation methods
- to provide a foundation for future development of Persian e-commerce recommender systems

---

## Features

- Persian-language review and comment processing
- Sentiment analysis on customer feedback
- Content-based recommendation using product-related text
- Collaborative filtering based on user-item interactions
- Basket-based recommendation using co-occurrence patterns
- Persian text preprocessing and stop-word handling
- Modular project structure for experimentation and extension

---

## System Components

### 1. Sentiment Analysis

The sentiment analysis module processes **Persian review text** to identify whether customer feedback expresses positive, negative, or neutral opinion. This helps extract useful information from textual reviews and can be used as an additional recommendation signal.

This component is useful because product ratings or recommendation labels alone may not fully capture how users actually feel about a product. Persian comments often contain richer opinions, complaints, and preferences that can improve the recommendation process.

Typical tasks in this stage include:

- cleaning Persian review text
- tokenization and normalization
- stop-word removal
- text vectorization
- training a sentiment classifier
- predicting sentiment labels for unseen comments

---

### 2. Content-Based Recommendation

The content-based recommender suggests products that are **textually similar** to a target product. It relies on product-related fields such as:

- product title
- review title
- review comment
- advantages
- disadvantages

These textual fields are transformed into numerical representations, and similarity scores are computed to find related products. This approach is useful when recommending items with similar descriptions, features, or customer opinions.

For example, if two products have similar review patterns or similar advantages and disadvantages, the content-based module can identify that similarity and suggest them together.

---

### 3. Collaborative Filtering

The collaborative filtering module generates **personalized recommendations** based on **user-item interactions**. Instead of focusing only on product text, it uses behavioral patterns to infer what a user may like based on past interactions and similarities across users or items.

This method is helpful because users with similar preferences often interact with similar products. Collaborative filtering captures those patterns and can recommend products even when textual similarity alone is not enough.

Typical signals used in this stage may include:

- recommendation labels
- user feedback
- implicit or explicit interaction history
- optional sentiment-enhanced weighting

---

### 4. Basket-Based Recommendation

The basket-based recommender identifies products that frequently appear **together** in the same user cart, order, or interaction context. This approach helps recommend **complementary items** rather than just similar items.

For example, if users often buy or view certain products together, the system can learn these associations and suggest one item when another is selected.

This method is especially useful for:

- cross-selling
- complementary item suggestion
- transaction pattern discovery
- association-based recommendation

---

## Technology Stack

The project is implemented using the following technologies:

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning / Recommendation
- Scikit-learn

### NLP / Text Processing
- TF-IDF vectorization
- Persian text preprocessing
- stop-word removal

### Similarity / Modeling
- Cosine similarity
- classical ML models for text classification and recommendation

### Development Environment
- Jupyter Notebook
- Python scripts

---

## Dataset

The dataset is based on **Persian-language e-commerce data** collected from an online shopping platform similar to Amazon. It contains both **product-related information** and **user-generated content**, which are used across the different modules of the system.

The dataset may include fields such as:

- product title
- review title
- review comment
- advantages
- disadvantages
- recommendation labels
- user-item interactions
- cart or transaction-level co-occurrence information

These fields support different parts of the project:

- **review text** supports sentiment analysis
- **product text fields** support content-based recommendation
- **interaction labels** support collaborative filtering
- **co-occurrence data** supports basket-based recommendation

Because the data is in **Persian**, preprocessing plays an important role in cleaning text, normalizing characters, and removing noise before modeling.

---

## Project Structure

```text
ParsiReco/
├── data/
│   ├── data.csv
│   ├── PreprocessedDataset2.csv
│   ├── SentimentDataset.csv
│   └── fa_stop_words.txt
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── sentimentalComments.ipynb
│   ├── content-based.ipynb
│   ├── collaborative.ipynb
│   └── apriori.ipynb
├── src/
│   ├── data_utils.py
│   ├── text_utils.py
│   ├── sentiment.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── basket.py
│   └── run_demo.py
└── requirements.txt