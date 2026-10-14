import tensorflow as tf
import json

# STEP 1: Load saved vocabulary
with open("models/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

# STEP 2: Recreate text vectorization
vectorizer = tf.keras.layers.TextVectorization(
    vocabulary=vocabulary,
    output_mode="int",
    output_sequence_length=50
)

# STEP 3: Example message
message = "Congratulations! You won a free prize!"

# STEP 4: Convert text into token IDs
token_ids = vectorizer(tf.constant([message]))

print("\nToken IDs shape:", token_ids.shape)

# STEP 5: Create embedding layer
embedding_layer = tf.keras.layers.Embedding(
    input_dim=len(vocabulary),
    output_dim=32,
    mask_zero=True
)

embedded_output = embedding_layer(token_ids)

print("Embedding output shape:", embedded_output.shape)

# STEP 6: Create SimpleRNN layer
rnn_layer = tf.keras.layers.SimpleRNN(
    units=64
)

rnn_output = rnn_layer(embedded_output)

print("RNN output shape:", rnn_output.shape)

# STEP 7: Create output layer
output_layer = tf.keras.layers.Dense(
    units=1,
    activation="sigmoid"
)

prediction = output_layer(rnn_output)

print("Prediction shape:", prediction.shape)

print("\nRaw prediction:", prediction.numpy())