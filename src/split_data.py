import pandas as pd
from sklearn.model_selection import train_test_split

# STEP 1: Load cleaned dataset
df = pd.read_csv("data/cleaned_sms.csv")

print("Total messages:", len(df))

# STEP 2: Split into 80% training and 20% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# STEP 3: Split temporary data into validation and testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# STEP 4: Print dataset sizes
print("\nTraining messages:", len(train_df))
print("Validation messages:", len(val_df))
print("Testing messages:", len(test_df))

# STEP 5: Save the splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\nDataset split completed successfully!")