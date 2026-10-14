import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# Step 2: Display first 5 rows
print("\nFirst five records:")
print(df.head())

# Step 3: Check dataset size
print("\nDataset shape:")
print(df.shape)

# Step 4: Count spam and ham messages
print("\nMessage counts:")
print(df["label"].value_counts())

# Step 5: Check missing values
print("\nMissing values:")
print(df.isnull().sum())
# STEP 6: Check duplicate records
print("\nDuplicate rows:")
print(df.duplicated().sum())

# STEP 7: Check duplicate messages
print("\nDuplicate messages:")
print(df["message"].duplicated().sum())

# STEP 8: Verify labels
print("\nUnique labels:")
print(df["label"].unique())
# STEP 9: Remove duplicate rows
df = df.drop_duplicates().copy()

# STEP 10: Reset row indexes
df = df.reset_index(drop=True)

# STEP 11: Check cleaned dataset
print("\nDataset after removing duplicates:")
print(df.shape)

# STEP 12: Check remaining spam and ham counts
print("\nCleaned message counts:")
print(df["label"].value_counts())

# STEP 13: Save cleaned dataset
df.to_csv(
    "data/cleaned_sms.csv",
    index=False,
    encoding="utf-8"
)

print("\nCleaned dataset saved successfully!")