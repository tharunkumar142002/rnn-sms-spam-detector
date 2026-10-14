import pandas as pd
import tensorflow as tf
import json
from pathlib import Path

# STEP 1: Load datasets
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

# STEP 2: Set tokenization parameters
VOCAB_SIZE = 10000
SEQ_LENGTH = 50

# STEP 3: Create text vectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH
)

# STEP 4: Learn vocabulary from training data ONLY
vectorizer.adapt(train_df["message"].astype(str).to_numpy())

# STEP 5: Convert text into numerical sequences
X_train = vectorizer(
    tf.constant(train_df["message"].astype(str).tolist())
)

X_val = vectorizer(
    tf.constant(val_df["message"].astype(str).tolist())
)

X_test = vectorizer(
    tf.constant(test_df["message"].astype(str).tolist())
)

# STEP 6: Display results
print("\nVocabulary size:", vectorizer.vocabulary_size())

print("\nOriginal message:")
print(train_df["message"].iloc[0])

print("\nToken IDs:")
print(X_train[0].numpy())

print("\nTraining input shape:", X_train.shape)
print("Validation input shape:", X_val.shape)
print("Testing input shape:", X_test.shape)

# STEP 7: Save vocabulary for future deployment
Path("models").mkdir(exist_ok=True)

with open("models/vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vectorizer.get_vocabulary(), f, ensure_ascii=False)

print("\nVocabulary saved successfully!")