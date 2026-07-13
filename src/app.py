from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from schemas import StudentData


# FastAPI App


app = FastAPI(
    title="Student Dropout Prediction API",
    description="Machine Learning API for Student Dropout Prediction",
    version="2.0.0",
)


# Paths


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "dropout_model.pkl"
DATA_PATH = BASE_DIR / "data" / "data.csv"


# Load model


try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")

# Feature order


feature_columns = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

prediction_labels = {
    0: "Dropout",
    1: "Graduate",
    2: "Enrolled"
}


# Routes


@app.get("/")
def home():

    return {
        "status": "running",
        "model_loaded": model is not None,
        "message": "Student Dropout Prediction API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(student: StudentData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model could not be loaded."
        )

    try:

        values = student.model_dump()

        df = pd.DataFrame(
            [[values[col] for col in feature_columns]],
            columns=feature_columns
        )

        prediction = int(model.predict(df)[0])

        probabilities = model.predict_proba(df)[0]

        confidence = float(max(probabilities))

        return {
            "prediction": prediction_labels[prediction],
            "prediction_code": prediction,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
