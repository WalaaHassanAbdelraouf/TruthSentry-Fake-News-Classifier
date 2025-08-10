# TruthSentry: Fake News Classifier

## Overview
TruthSentry is a machine learning project designed to classify news articles as fake or true. It uses a logistic regression model trained on text data processed with TF-IDF vectorization. A Flask-based web application is included, allowing users to input text and receive predictions on whether the news is fake or true.

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), containing 44,919 news articles (23,502 fake and 21,417 true).

## Dataset
The dataset consists of two CSV files:
- **Fake.csv**: 23,502 fake news articles.
- **True.csv**: 21,417 true news articles.

Each article includes:
- **Title**: The title of the news article.
- **Text**: The body text of the news article.
- **Subject**: The subject or category of the news article.
- **Date**: The publication date of the news article.

**Important**: Due to their large size, `Fake.csv` and `True.csv` are not included in this repository. Download them from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in the `data/` folder. The `data/` directory contains a `.gitkeep` file to maintain its structure in the repository.

## Methodology
1. **Data Preprocessing**:
   - **Text Cleaning**: The `clean_text` function processes text by:
     - Converting to lowercase.
     - Removing special characters, punctuation, and numbers.
     - Eliminating extra whitespace.
     - Removing stopwords.
     - Applying lemmatization.
   - **Label Encoding**: The `Subject` column is encoded numerically.
   
2. **Feature Extraction**:
   - Text is transformed into numerical features using TF-IDF vectorization.

3. **Model Training**:
   - A logistic regression model is trained on the TF-IDF features.

4. **Evaluation**:
   - Model performance:
     - **Training Accuracy**: 96.91%
     - **Validation Accuracy**: 96.80%
     - **Test Accuracy**: 96.73%

5. **Web Application**:
   - A Flask app (`app.py`) allows users to input text and receive predictions using the trained model.

