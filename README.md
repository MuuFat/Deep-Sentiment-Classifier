# Sentiment Analysis on Text Datasets

This project performs sentiment analysis using various machine learning and deep learning models such as Naive Bayes, SVM, Word2Vec+LSTM, FastText+LSTM, and BERT.

## Supported Datasets

- IMDb Reviews
- Yelp Reviews
- Twitter Sentiment140

## Installation

Before running the code, make sure you have installed all required Python packages. Use the following command:

pip install -r requirements.txt

## How to Run

Navigate to the project root folder and run the main script:

python main.py

You will be prompted to choose:
The dataset you want to use (IMDb, Yelp, or Twitter)
The model you want to test (Naive Bayes, SVM, Word2Vec+LSTM, FastText+LSTM, or BERT)

The script will automatically load the dataset, preprocess the data, train the selected model, and display evaluation results.

## Environment Variables (Optional)

You can set the dataset from the environment directly:

export DATASET_CHOICE=2  # 1 for IMDb, 2 for Yelp, 3 for Twitter

On Command Prompt:

set DATASET_CHOICE=2

## Project Structure

SentimentAnalysisProject/
│
├── data/                      # Datasets here
├── models/                    # Model implementation files
├── main.py                    # Entry point for running the models
├── requirements.txt           # Python dependencies
└── README.md                  # This file

## Outputs

Each model prints its evaluation results including:

Accuracy
Precision
Recall
F1-Score