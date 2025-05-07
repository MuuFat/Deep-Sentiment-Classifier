import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Create required directories if they don't exist
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Dataset selection
dataset_choice = os.getenv("DATASET_CHOICE", "1")

if dataset_choice == "1":
    dataset_path = "data/imdb_IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"
elif dataset_choice == "2":
    dataset_path = "data/yelp_review.csv"
    text_column = "text"
    label_column = "sentiment"
elif dataset_choice == "3":
    dataset_path = "data/twitter_training.1600000.processed.noemoticon.csv"
    text_column = "text"
    label_column = "target"
else:
    print("Invalid choice. Defaulting to IMDb.")
    dataset_path = "data/imdb_IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"

# Load dataset
if dataset_choice == "3":
    column_names = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(dataset_path, encoding="latin1", names=column_names, header=None)
else:
    df = pd.read_csv(dataset_path)

df.dropna(inplace=True)
df[text_column] = df[text_column].astype(str).str.lower()

print("\n Dataset preview (first 5 rows):")
print(df.head())

# Label mapping
if dataset_choice == "3":
    df[label_column] = df[label_column].map({0: 0, 4: 1})
else:
    df[label_column] = df[label_column].map({"positive": 1, "negative": 0})

df.dropna(subset=[label_column], inplace=True)

# Label distribution
print("\n Label Distribution:")
print(df[label_column].value_counts())

# Optional: Save plot
df[label_column].value_counts().plot(kind='bar', title='Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/svm_label_distribution.png')
plt.clf()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df[text_column])
y = df[label_column]

print(f"\n TF-IDF Matrix Shape: {X.shape} (samples, features)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = LinearSVC()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")