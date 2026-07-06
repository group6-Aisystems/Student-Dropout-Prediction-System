import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance['Feature'][:10],
    feature_importance['Importance'][:10]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")

plt.show()
-----------------------------
df['Gender'].value_counts()
------------------------------
print(df['Gender'].unique())
---------------------------
male_df = df[df['Gender'] == 1]
female_df = df[df['Gender'] == 0]

print("Male Students:", len(male_df))
print("Female Students:", len(female_df))
------------------------------------
from sklearn.metrics import accuracy_score

# Male
male_test = X_test[df.loc[X_test.index, 'Gender'] == 1]
male_y_test = y_test[df.loc[X_test.index, 'Gender'] == 1]

male_pred = model.predict(male_test)

male_accuracy = accuracy_score(
    male_y_test,
    male_pred
)

# Female
female_test = X_test[df.loc[X_test.index, 'Gender'] == 0]
female_y_test = y_test[df.loc[X_test.index, 'Gender'] == 0]

female_pred = model.predict(female_test)

female_accuracy = accuracy_score(
    female_y_test,
    female_pred
)

print("Male Accuracy:", male_accuracy)
print("Female Accuracy:", female_accuracy)
--------------------------------------------------
import joblib

joblib.dump(model, "dropout_model.pkl")

