
# training.py

import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model():
    """
    Train a Random Forest model using preprocessed training data.
    """

    # Load training data
    X_train = joblib.load("X_train.pkl")
    y_train = joblib.load("y_train.pkl")

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "dropout_model.pkl")

    print("Model training completed.")
    print("Trained model saved as dropout_model.pkl")


if __name__ == "__main__":
    train_model()
