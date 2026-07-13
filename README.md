# 🎓 AI-Based Student Dropout Prediction System

An AI-powered Student Dropout Prediction System developed as part of the **MSc AI Systems Engineering** module. The project applies Machine Learning and AI Systems Engineering principles to predict whether a student is likely to **Drop Out**, **Graduate**, or remain **Enrolled**, using academic and demographic information.

---

# Project Objectives

The objectives of this project are to:

- Predict student academic outcomes using Machine Learning.
- Identify students who are at risk of dropping out.
- Evaluate model performance using standard classification metrics.
- Demonstrate deployment of an AI model using FastAPI.
- Apply AI Systems Engineering concepts including fairness, security, monitoring, scalability and Responsible AI.

---

# Dataset

**Dataset Name**

UCI Student Dropout and Academic Success Dataset

**Number of Records**

4424 Students

**Target Classes**

- Dropout
- Graduate
- Enrolled

**Features**

The dataset contains demographic, academic, financial and enrollment information including:

- Admission Grade
- Previous Qualification
- Age at Enrollment
- Tuition Fees Status
- Curricular Units
- Scholarship Status
- Gender
- Parents' Qualification
- Previous Grades

---

# System Architecture

The complete AI system follows the workflow below:

```
Student Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Selection
        │
        ▼
Random Forest Classifier
        │
        ▼
Model Evaluation
        │
        ▼
Saved Model (.pkl)
        │
        ▼
FastAPI Deployment
        │
        ▼
Swagger Documentation
        │
        ▼
End User
```

See:

```
figures/architecture_diagram.png
```

---

# Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- FastAPI
- Uvicorn
- Joblib
- Pyngrok
- GitHub

---

# Machine Learning Model

Algorithm Used:

**Random Forest Classifier**

The trained model predicts one of three classes:

- Dropout
- Graduate
- Enrolled

The trained model is stored inside:

```
models/dropout_model.pkl
```

---

# Model Performance

Performance obtained during testing:

| Metric | Score |
|---------|-------|
| Accuracy | 77.5% |
| Precision | 76% |
| Recall | 78% |
| F1 Score | 76% |

Additional evaluation includes:

- Confusion Matrix
- Feature Importance Analysis
- Class-wise F1 Scores
- Fairness Analysis
- Security Analysis

Performance figures are available inside:

```
figures/
```

---

# Deployment

The trained model is deployed using **FastAPI**.

Interactive API testing is available through **Swagger UI**.

Example endpoints:

```
GET /
POST /predict
```

Swagger screenshots are available inside:

```
screenshots/
```

---

# AI Systems Engineering Considerations

The project also addresses several AI Systems Engineering requirements:

- Model Deployment
- Monitoring and Maintenance
- Security Analysis
- Fairness Analysis
- Responsible AI
- Scalability
- Reproducibility

---

# Repository Structure

```
Student-Dropout-Prediction-System/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── data.csv
│
├── models/
│   └── dropout_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── prediction.py
│   └── fairness_analysis.py
│
├── notebooks/
│   ├── Student_dropout_prediction.ipynb
│   └── fastAPI_Deployment.ipynb
│
├── figures/
│
└── docs/
```

---

# Installation

Clone the repository:
git clone https://github.com/group6-Aisystems/Student-Dropout-Prediction-System.git
cd Student-Dropout-Prediction-System

pip install -r requirements.txt

python src/preprocessing.py
python src/train.py
python src/prediction.py

uvicorn app:app --reload

Run the notebook to train the model or execute the FastAPI deployment notebook to launch the API.

---

# Contributors

- Collins Chikaodi Collins
- Afaq Ahmed
- Shajid Haque
- Shabab Anwar Khan

---

# Video Demonstration

A complete video demonstration showing:

- Dataset preprocessing
- Model training
- Model evaluation
- FastAPI deployment
- Swagger testing
- Prediction examples

is available in:

```
docs/video_demo_link.md
```

---

# License

This project was developed for academic purposes as part of the MSc AI Systems Engineering programme.
