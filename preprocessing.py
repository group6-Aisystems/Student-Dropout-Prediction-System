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
