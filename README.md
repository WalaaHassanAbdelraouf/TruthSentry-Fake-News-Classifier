# TruthSentry-Fake-News-Classifier

## Overview
This project aims to classify news articles as fake or true using machine learning techniques. The dataset used is the "Fake and Real News Dataset" from Kaggle, containing a total of 44,919 news articles (23,502 fake and 21,417 true). The project employs text preprocessing, feature extraction, and a logistic regression model to achieve high classification accuracy.

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and consists of two CSV files:
- **Fake.csv**: Contains 23,502 fake news articles.
- **True.csv**: Contains 21,417 true news articles.

Each article includes the following columns:
- **Title**: The title of the news article.
- **Text**: The body text of the news article.
- **Subject**: The subject or category of the news article.
- **Date**: The publication date of the news article.

## Methodology
The project follows these key steps:

1. **Data Preprocessing**:
   - **Text Cleaning**: The text is processed using the `clean_text` function, which:
     - Converts text to lowercase.
     - Removes special characters and punctuation using regex.
     - Removes numerical digits.
     - Eliminates extra whitespace.
     - Removes stopwords (common words like "the", "is") to reduce noise.
     - Applies lemmatization to normalize words to their base form.
   - **Label Encoding**: The `Subject` column is encoded numerically to enable model processing.
   
2. **Feature Extraction**:
   - **TF-IDF Vectorization**: The cleaned text is transformed into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF), capturing the importance of words in the dataset.

3. **Model Training**:
   - A **Logistic Regression** model is trained on the TF-IDF features to classify articles as fake or true.

4. **Evaluation**:
   - The model is evaluated on training, validation, and test sets, yielding the following accuracies:
     - **Training Accuracy**: 96.91%
     - **Validation Accuracy**: 96.80%
     - **Test Accuracy**: 96.73%

## Results
The logistic regression model demonstrates strong performance, achieving over 96% accuracy across training, validation, and test sets. These results suggest the model effectively distinguishes between fake and true news articles based on the processed text features.



You can install them using:
```bash
pip install pandas numpy scikit-learn nltk
