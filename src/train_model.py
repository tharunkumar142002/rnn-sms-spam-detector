import tensorflow as tf
import pandas as pd
import json
from pathlib import Path

# STEP 1: Reproducibility
tf.keras.utils.set_random_seed(42)

# STEP 2: Load datasets
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")

# STEP 3: Convert labels into numbers
label_map = {
    "ham": 0,
    "spam": 1
}

y_train = train_df["label"].map(label_map).to_numpy()
y_val = val_df["label"].map(label_map).to_numpy()

# STEP 4: Prepare input SMS messages
X_train = train_df["message"].astype(str).to_numpy()
X_val = val_df["message"].astype(str).to_numpy()

# STEP 5: Load saved vocabulary
with open("models/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

# STEP 6: Recreate text vectorizer
vectorizer = tf.keras.layers.TextVectorization(
    vocabulary=vocabulary,
    output_mode="int",
    output_sequence_length=50
)

# STEP 7: Build complete RNN model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(), dtype=tf.string),

    vectorizer,

    tf.keras.layers.Embedding(
        input_dim=len(vocabulary),
        output_dim=32,
        mask_zero=True
    ),

    tf.keras.layers.SimpleRNN(
        units=64
    ),

    tf.keras.layers.Dense(
        units=1,
        activation="sigmoid"
    )
])

# STEP 8: Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# STEP 9: Display model architecture
model.summary()

# STEP 10: Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# STEP 11: Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# STEP 12: Save trained model
Path("models").mkdir(exist_ok=True)

model.save("models/sms_rnn.keras")

print("\nModel trained and saved successfully!")