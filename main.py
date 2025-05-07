"""
Main Interface for Running Sentiment Analysis Models
Supports dataset selection and model selection
Author: Muhammed Fatih Kalkan
"""

import os

def run_program():
    # Dataset Selection
    print("\n===============================")
    print("   Sentiment Analysis Project")
    print("===============================")
    print("\nAvailable datasets:")
    print("1. IMDb Reviews")
    print("2. Yelp Reviews")
    print("3. Twitter Sentiment140")

    dataset_choice = input("\nEnter the dataset number (1-3): ").strip()

    if dataset_choice not in ["1", "2", "3"]:
        print("Invalid choice. Defaulting to IMDb dataset.")
        dataset_choice = "1"

    # Model Selection
    print("\nAvailable models:")
    print("1. Naive Bayes (TF-IDF)")
    print("2. SVM (TF-IDF)")
    print("3. Word2Vec + LSTM")
    print("4. FastText + LSTM")
    print("5. BERT (Transformers)")

    model_choice = input("\nEnter the model number (1-5): ").strip()

    # Define model command mapping
    model_commands = {
        "1": f"python models/naive_bayes.py",
        "2": f"python models/svm_model.py",
        "3": f"python models/word2vec_lstm.py",
        "4": f"python models/fasttext_lstm.py",
        "5": f"python models/bert_model.py",
    }

    if model_choice not in model_commands:
        print("Invalid model choice.")
        return

    # Set environment variable to pass dataset choice
    os.environ["DATASET_CHOICE"] = dataset_choice

    # Run selected model
    os.system(model_commands[model_choice])

if __name__ == "__main__":
    run_program()
