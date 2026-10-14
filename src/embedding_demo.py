import json
import tensorflow as tf
import pandas as pd

# STEP 1: Load training dataset
train_df = pd.read_csv("data/train.csv")

# STEP 2: Load saved vocabulary
with open("models/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

# STEP 3: Recreate the text vectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    vocabulary=vocabulary,
    output_mode="int",
    output_sequence_length=50
)

# STEP 4: Select the first SMS message
message = str(train_df["message"].iloc[0])

print("\nOriginal message:")
print(message)

# STEP 5: Convert message into token IDs
token_ids = vectorizer(tf.constant([message]))

print("\nToken IDs:")
print(token_ids.numpy())

# STEP 6: Create Embedding layer
embedding_layer = tf.keras.layers.Embedding(
    input_dim=len(vocabulary),
    output_dim=32,
    mask_zero=True
)

# STEP 7: Convert token IDs into embedding vectors
embedded_output = embedding_layer(token_ids)

print("\nToken shape:", token_ids.shape)

print("\nEmbedding output shape:", embedded_output.shape)

print("\nFirst token embedding:")
print(embedded_output[0][0].numpy())