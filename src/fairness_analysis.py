import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


def fairness_analysis():
    """
    Evaluate the fairness of the trained Random Forest model
    by comparing prediction accuracy for male and female students.
    """

    # Load processed data
    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        "processed_data.pkl"
    )

    # Load original dataset
    df = pd.read_csv("data/data.csv", sep=";")

    # Encode target labels
    df["Target"] = df["Target"].map({
        "Dropout": 0,
        "Graduate": 1,
        "Enrolled": 2
    })

    # Load trained model
    model = joblib.load("dropout_model.pkl")

    # Display gender distribution
    print("=" * 50)
    print("Gender Distribution")
    print("=" * 50)
    print(df["Gender"].value_counts())

    print("\nUnique Gender Values:")
    print(df["Gender"].unique())

    # Count male and female students
    male_df = df[df["Gender"] == 1]
    female_df = df[df["Gender"] == 0]

    print("\nMale Students:", len(male_df))
    print("Female Students:", len(female_df))

    # Create gender-specific test datasets
    male_test = X_test[df.loc[X_test.index, "Gender"] == 1]
    male_y_test = y_test[df.loc[X_test.index, "Gender"] == 1]

    female_test = X_test[df.loc[X_test.index, "Gender"] == 0]
    female_y_test = y_test[df.loc[X_test.index, "Gender"] == 0]

    # Make predictions
    male_pred = model.predict(male_test)
    female_pred = model.predict(female_test)

    # Calculate accuracy
    male_accuracy = accuracy_score(male_y_test, male_pred)
    female_accuracy = accuracy_score(female_y_test, female_pred)

    # Display results
    print("\n" + "=" * 50)
    print("Fairness Analysis Results")
    print("=" * 50)

    print(f"Male Accuracy        : {male_accuracy:.3f}")
    print(f"Female Accuracy      : {female_accuracy:.3f}")

    accuracy_difference = abs(male_accuracy - female_accuracy)

    print(f"Accuracy Difference  : {accuracy_difference:.3f}")

    # Fairness conclusion
    if accuracy_difference < 0.05:
        print("\nConclusion:")
        print("The model demonstrates relatively fair performance across male and female students.")
    else:
        print("\nConclusion:")
        print("The model shows a noticeable performance difference between gender groups and may require further fairness improvements.")

    print("\nFairness analysis completed successfully.")


if __name__ == "__main__":
    fairness_analysis()
