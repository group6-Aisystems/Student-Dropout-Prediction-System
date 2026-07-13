
# training.py

import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model():
    """
    Train a Random Forest model using preprocessed training data.
    """

 # Load processed training and testing data
    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        "processed_data.pkl"
    )



    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "dropout_model.pkl")

    print("\nModel training completed successfully.")
    print(f"Training samples : {len(X_train)}")
    print(f"Number of features : {X_train.shape[1]}")
    print("Trained model saved as dropout_model.pkl")


if __name__ == "__main__":
    train_model()
