import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


def preprocess_data(file_path="data/data.csv"):
    """
    Load and preprocess the student dropout dataset.
    """

    # Load dataset
    df = pd.read_csv(file_path, sep=";")
    print("\nMissing Values:")
    print(df.isnull().sum())
    # Display dataset information
    print("First 5 Rows:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nDataset Statistics:")
    print(df.describe())

    print("\nTarget Class Distribution:")
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

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nNumber of Features: {X.shape[1]}")
    print("\nTraining Set Shape:", X_train.shape)
    print("Testing Set Shape:", X_test.shape)

    # Save processed data for later use
    joblib.dump(
        (X_train, X_test, y_train, y_test, X.columns.tolist()),
        "processed_data.pkl"
    )

    print("\nPreprocessing completed successfully.")
    print("Processed data saved as 'processed_data.pkl'.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
