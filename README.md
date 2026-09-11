# Disease Prediction Using Machine Learning

A machine learning-based disease prediction project that analyzes patient health attributes and predicts the presence or absence of heart disease. The project includes data preprocessing, exploratory data analysis, multiple machine learning models, performance comparison, feature importance analysis, and cross-source evaluation using the UCI Heart Disease dataset.

## Project Overview

Early identification of potential heart disease can support better medical decision-making and risk assessment.

This project develops a supervised machine learning classification system using patient-related clinical attributes. Four classification algorithms are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The Random Forest model achieved the best performance on the held-out test set and was selected as the final model.

> **Important:** This project is an educational machine learning study and is not intended to provide medical diagnosis or replace professional medical advice.

---

## Objectives

- Build a machine learning model for heart disease prediction.
- Perform data cleaning and preprocessing.
- Analyze important patterns in the dataset.
- Compare multiple classification algorithms.
- Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Identify influential features using Random Forest feature importance.
- Perform an additional cross-source evaluation using UCI Heart Disease records.
- Save important experimental results for reproducibility.

---

## Dataset

### Primary Dataset

**Heart Disease Dataset — mexwell**

The dataset contains 1,190 patient records and 11 input features with a binary target variable.

Original file used:

`heart_statlog_cleveland_hungary_final.csv`

### Dataset Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex |
| chest pain type | Type of chest pain |
| resting bp s | Resting blood pressure |
| cholesterol | Cholesterol level |
| fasting blood sugar | Fasting blood sugar indicator |
| resting ecg | Resting electrocardiogram result |
| max heart rate | Maximum heart rate |
| exercise angina | Exercise-induced angina |
| oldpeak | ST depression |
| ST slope | Slope of the ST segment |

### Target

- `0` — No Disease
- `1` — Disease

### Dataset Quality

- Original records: **1,190**
- Features: **11**
- Target: **1**
- Missing values: **0**
- Duplicate records removed: **272**
- Records after duplicate removal: **918**

---

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub

---

## Machine Learning Workflow

```text
Dataset Acquisition
        ↓
Data Loading
        ↓
Data Quality Analysis
        ↓
Duplicate Removal
        ↓
Exploratory Data Analysis
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Evaluation
        ↓
Feature Importance
        ↓
Cross-Source Evaluation
        ↓
Results Saving

Data Preprocessing

The dataset was checked for:

Missing values
Duplicate records
Data types
Target distribution
Feature distributions
Feature relationships

Duplicate records were removed before model development.
Numerical features were standardized using StandardScaler.
Categorical features were transformed using OneHotEncoder.
A ColumnTransformer was used to apply the appropriate preprocessing to each feature group.

Train-Test Split
The cleaned dataset contains 918 records.
The data was divided using:
Training data: 80%
Testing data: 20%
Stratification: Yes
Random state: 42

The test set contains 184 records.

Models Used
1. Logistic Regression
Used as a baseline linear classification model.
2. Decision Tree
Used to model non-linear relationships through decision-based splits.
3. Random Forest
An ensemble of decision trees used to improve prediction performance and robustness.
4. Gradient Boosting
An ensemble learning method that sequentially improves weak learners to produce a stronger classifier.

Model Comparison
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	88.59%	87.16%	93.14%	90.05%	93.01%
Decision Tree	80.43%	82.35%	82.35%	82.35%	85.31%
Random Forest	90.22%	88.89%	94.12%	91.43%	93.92%
Gradient Boosting	89.13%	90.20%	90.20%	90.20%	93.19%
Best Model

The Random Forest Classifier was selected as the final model because it achieved the highest ROC-AUC and the strongest overall performance on the held-out test set.

Test Set Performance
Accuracy: 90.22%
Precision: 88.89%
Recall: 94.12%
F1-Score: 91.43%
ROC-AUC: 93.92%

The recall of 94.12% indicates that the model correctly identified a high proportion of disease-positive cases in the held-out test set.

Feature Importance

The Random Forest model identified the following features among the most influential model inputs:

ST slope
Cholesterol
Maximum heart rate
Oldpeak
Chest pain type
Age
Resting blood pressure
Exercise angina
Sex
Fasting blood sugar

Feature importance represents the contribution of features to the model's predictive decisions. It should not be interpreted as medical causation.

Cross-Source Evaluation

An additional evaluation was performed using the UCI Heart Disease dataset (UCI ID 45).

The UCI dataset contains 303 records. The target was converted into a binary classification:

0 → No Disease
1–4 → Disease

The feature names were mapped to match the primary dataset where applicable.
Cross-Source Results
Accuracy: 97.36%
Precision: 97.12%
Recall: 97.12%
F1-Score: 97.12%
ROC-AUC: 98.62%
Important Validation Limitation
An exact feature-level overlap check was performed between the cleaned training dataset and the UCI evaluation records.
The result showed:
Training dataset unique feature rows: 918
UCI evaluation rows: 303
Exact feature-row overlap: 303

Therefore, all 303 UCI records were already represented in the training dataset.
Consequently, the UCI evaluation is reported as a cross-source consistency check, not as independent external validation or independent generalization testing.
The primary model performance reported for this project is therefore the held-out 20% test-set performance.

Results Files
The following files are included in the results/ directory:

results/
│
├── external_roc_curve.png
├── external_validation_results.csv
├── final_model_results.csv
└── validation_summary.csv
File Description
external_roc_curve.png
- ROC curve generated from the UCI cross-source evaluation.
external_validation_results.csv
- Contains individual UCI evaluation records, actual labels, predicted labels, probabilities, and prediction correctness.
final_model_results.csv
- Contains the final performance comparison for the held-out test set and UCI cross-source check.
validation_summary.csv
- Contains the dataset overlap analysis and validation summary.

Project Structure
CodeAlpha_Disease-Prediction/
│
├── CodeAlpha_Disease_Prediction.ipynb
├── README.md
├── requirements.txt
│
└── results/
    ├── external_roc_curve.png
    ├── external_validation_results.csv
    ├── final_model_results.csv
    └── validation_summary.csv
## Google colab
