import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv", sep=";")

# Explore dataset
print(df.head())
df.info()
print(df.describe())
print(df["Target"].value_counts())

# Encode target labels
df["Target"] = df["Target"].map({
    "Dropout": 0,
    "Graduate": 1,
    "Enrolled": 2
})

# Separate features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)
if __name__ == "__main__":
    main()
