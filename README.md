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
Student-Dropout-Prediction-System
│
├── README.md
├── requirements.txt
│
├── data
│   └── student_dropout_dataset.csv
│
├── notebooks
│   ├── Student_Dropout_Prediction.ipynb
│   └── FastAPI_Deployment.ipynb
│
├── models
│   └── dropout_model.pkl
│
├── figures
│   ├── architecture_diagram.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── performance_metrics.png
│   └── classwise_f1.png
│
├── screenshots
│   ├── swagger_home.png
│   └── swagger_prediction.png
│
└── docs
    ├── dataset_description.md
    ├── fairness_analysis.md
    ├── security_analysis.md
    ├── monitoring_and_maintenance.md
    ├── contributor_roles.md
    ├── video_demo_link.md
    └── Final_Report.pdf
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<repository-name>.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

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
