import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load unseen test dataset
test_df = pd.read_csv("data/test.csv")

print("Test dataset shape:", test_df.shape)

# STEP 2: Prepare test messages
X_test = test_df["message"].astype(str).to_numpy()

# STEP 3: Convert labels into numbers
label_map = {
    "ham": 0,
    "spam": 1
}

y_test = test_df["label"].map(label_map).to_numpy()

# STEP 4: Load trained RNN model
model = tf.keras.models.load_model(
    "models/sms_rnn.keras",
    compile=False
)

print("\nModel loaded successfully!")

# STEP 5: Predict spam probabilities
probabilities = model.predict(X_test).flatten()

# STEP 6: Convert probabilities into classes
y_pred = (probabilities >= 0.5).astype(int)

# STEP 7: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# STEP 8: Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

# STEP 9: Detailed classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

# STEP 10: Display example predictions
print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(5):
    print("\nMessage:", X_test[i][:150])
    print("Actual:", "Spam" if y_test[i] == 1 else "Ham")
    print("Predicted:", "Spam" if y_pred[i] == 1 else "Ham")
    print("Spam probability:", round(float(probabilities[i]), 4))