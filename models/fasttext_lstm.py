import os
import pandas as pd
import numpy as np
import nltk
import gensim
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Create required directories if they don't exist
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Configuration
SAMPLE_LIMIT = 5000

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

# Remove missing values
df.dropna(inplace=True)

# Lowercase and convert to string
df[text_column] = df[text_column].astype(str).str.lower()

# Preview of the dataset
print("\nPreview of the dataset (first 5 rows):")
print(df.head())

# Label mapping
if dataset_choice == "3":
    df[label_column] = df[label_column].map({0: 0, 4: 1})
elif dataset_choice == "1" or dataset_choice == "2":
    df[label_column] = df[label_column].map({"positive": 1, "negative": 0})

df.dropna(subset=[label_column], inplace=True)

# Label distribution plot
label_counts = df[label_column].value_counts()
print("\nLabel Distribution:")
print(label_counts)
label_counts.plot(kind='bar', title='Label Distribution (FastText)')
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/fasttext_label_distribution.png")
plt.clf()

# Text preprocessing: tokenize and remove stopwords
def preprocess(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        return [word for word in tokens if word.isalpha() and word not in stop_words]
    else:
        return []

df["tokens"] = df[text_column].apply(preprocess)

# Train FastText embeddings
ft_model = gensim.models.FastText(
    sentences=df["tokens"],
    vector_size=100,
    window=5,
    min_count=5,
    workers=4
)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["tokens"])
word_index = tokenizer.word_index
print(f"Total unique tokens: {len(word_index)}")

# Convert tokens to sequences and pad them
X = tokenizer.texts_to_sequences(df["tokens"])
X = pad_sequences(X, maxlen=100)
y = df[label_column].values

# Create embedding matrix from FastText model
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    if word in ft_model.wv:
        embedding_matrix[i] = ft_model.wv[word]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model with pre-trained embeddings
model = Sequential()
model.add(
    Embedding(
        input_dim=len(word_index) + 1,
        output_dim=100,
        weights=[embedding_matrix],
        input_length=100,
        trainable=False,
    )
)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split=0.1, verbose=1)

# Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("FastText + LSTM Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("plots/fasttext_training_loss.png")
plt.clf()

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print evaluation metrics
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")