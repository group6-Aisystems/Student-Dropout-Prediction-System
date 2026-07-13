# prediction.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model():
    """
    Load trained model and evaluate it on test data.
    """

    # Load model and test data
    model = joblib.load("dropout_model.pkl")

    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        "processed_data.pkl"
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
     print(f"\nModel Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Dropout", "Graduate", "Enrolled"]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Dropout", "Graduate", "Enrolled"]
    )

    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["Feature"][:10],
        feature_importance["Importance"][:10]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Important Features")
    plt.gca().invert_yaxis()
    plt.savefig("feature_importance.png")
    plt.show()

    # Gender-based analysis
    if "Gender" in X_test.columns:
        male_test = X_test[X_test["Gender"] == 1]
        male_y_test = y_test[X_test["Gender"] == 1]

        female_test = X_test[X_test["Gender"] == 0]
        female_y_test = y_test[X_test["Gender"] == 0]

        if len(male_test) > 0:
            male_pred = model.predict(male_test)
            male_accuracy = accuracy_score(male_y_test, male_pred)
            print("Male Accuracy:", male_accuracy)

        if len(female_test) > 0:
            female_pred = model.predict(female_test)
            female_accuracy = accuracy_score(female_y_test, female_pred)
            print("Female Accuracy:", female_accuracy)


if __name__ == "__main__":
    evaluate_model()
