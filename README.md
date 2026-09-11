# Disease Prediction Using Machine Learning

## Project Overview

This project focuses on developing a machine learning-based system for predicting the presence of heart disease from patient health and clinical features.

The project follows a complete end-to-end machine learning workflow, including dataset acquisition, data quality analysis, data cleaning, exploratory data analysis, preprocessing, model development, model comparison, evaluation, feature importance analysis, and cross-source consistency checking.

The main objective is to build a reliable classification model that can distinguish between patients with and without heart disease based on available clinical attributes.

> **Disclaimer:** This project is developed for educational and research purposes. The predictions are not intended to replace professional medical diagnosis or clinical decision-making.

---

## Objectives

The main objectives of this project are:

- To analyze a heart disease dataset using machine learning techniques.
- To perform data quality assessment and data cleaning.
- To explore relationships between patient attributes and heart disease.
- To preprocess numerical and categorical features appropriately.
- To develop and compare multiple machine learning classification models.
- To evaluate models using multiple performance metrics.
- To identify important features contributing to model predictions.
- To perform an additional cross-source consistency check using the UCI Heart Disease dataset.
- To store reproducible evaluation results and visualizations.

---

## Dataset

The primary dataset used in this project is the **Heart Disease Dataset** obtained from Kaggle.

**Dataset Source:** Kaggle – Heart Disease Dataset by mexwell

The dataset contains patient-related clinical attributes that can be used to predict whether heart disease is present.

### Dataset Size

- Original records: **1,190**
- Input features: **11**
- Target variable: **1**
- Total columns: **12**
- Missing values: **0**
- Duplicate records: **272**
- Duplicate percentage: **22.86%**

After removing duplicate records:

- Cleaned records: **918**
- Input features: **11**
- Target variable: **1**
- Remaining duplicates: **0**

---

## Dataset Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex of the patient |
| chest pain type | Type of chest pain |
| resting bp s | Resting blood pressure |
| cholesterol | Cholesterol level |
| fasting blood sugar | Fasting blood sugar indicator |
| resting ecg | Resting electrocardiographic result |
| max heart rate | Maximum heart rate achieved |
| exercise angina | Exercise-induced angina indicator |
| oldpeak | ST depression induced by exercise |
| ST slope | Slope of the peak exercise ST segment |
| target | Heart disease target |

### Target Classes

- **0 → No Disease**
- **1 → Disease**

After duplicate removal:

- No Disease: **410 records (44.66%)**
- Disease: **508 records (55.34%)**

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
- UCI ML Repository
- Git
- GitHub

---

## Machine Learning Workflow

The project follows the following machine learning workflow:

1. Dataset Acquisition
2. Data Quality Assessment
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Preparation
6. Data Preprocessing
7. Train-Test Split
8. Model Development
9. Model Training
10. Model Evaluation
11. Model Comparison
12. Best Model Selection
13. Feature Importance Analysis
14. Cross-Source Consistency Check
15. Results Storage

---

## Project Implementation

The project was implemented as a complete end-to-end machine learning workflow. The implementation covers dataset acquisition, data quality assessment, preprocessing, exploratory analysis, model development, evaluation, feature analysis, and cross-source consistency checking.

### 1. Dataset Acquisition

- Downloaded the primary heart disease dataset using KaggleHub.
- Loaded the dataset into a Pandas DataFrame.
- Inspected the available files and dataset structure.

### 2. Data Quality Assessment

- Checked dataset dimensions and data types.
- Verified missing values.
- Identified duplicate records.
- Examined target-class distribution.
- Analyzed unique values.
- Checked numerical feature ranges.

### 3. Data Cleaning

- No missing values were found in the primary dataset.
- Detected **272 duplicate records**.
- Removed duplicate records before model training.
- Obtained **918 unique records** for further analysis.

### 4. Exploratory Data Analysis

The following analyses were performed:

- Target-class distribution analysis.
- Numerical feature distribution analysis.
- Histograms.
- Box plots.
- Categorical feature analysis.
- Correlation heatmap.

These visualizations were used to understand the structure of the dataset and relationships among the clinical attributes.

### 5. Feature Preparation

The input features were divided into numerical and categorical groups.

#### Numerical Features

- age
- resting bp s
- cholesterol
- max heart rate
- oldpeak

#### Categorical Features

- sex
- chest pain type
- fasting blood sugar
- resting ecg
- exercise angina
- ST slope

### 6. Data Preprocessing

Different preprocessing techniques were applied according to the feature type.

For numerical features:

- `StandardScaler` was used for feature scaling.

For categorical features:

- `OneHotEncoder` was used to convert categorical variables into numerical representations.

The preprocessing operations were combined using `ColumnTransformer`.

### 7. Train-Test Splitting

The cleaned dataset was divided into training and testing sets.

- Training data: **80%**
- Testing data: **20%**
- Split method: **Stratified**
- Random state: **42**

The stratified split was used to maintain the target-class distribution between the training and testing sets.

### 8. Model Development

Four machine learning classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Each model was integrated with the preprocessing pipeline.

### 9. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

These metrics provide a broader evaluation than accuracy alone.

### 10. Model Comparison

The model performances were compared using the held-out test dataset.

### 11. Best Model Selection

The **Random Forest Classifier** was selected as the final model because it achieved the highest overall performance on the held-out test set, including the highest ROC-AUC and F1-Score among the evaluated models.

### 12. Feature Importance Analysis

Feature importance was extracted from the Random Forest model to understand which input variables contributed most strongly to the model's predictions.

Important model features included:

- ST slope
- Cholesterol
- Maximum heart rate
- Oldpeak
- Chest pain type
- Age
- Resting blood pressure
- Exercise angina

### 13. Cross-Source Consistency Check

An additional evaluation was performed using the **UCI Heart Disease dataset**.

The UCI dataset contains **303 records** and was used to check whether the trained model produced consistent predictions on records represented in another source.

The UCI feature names were mapped to the corresponding feature names used by the primary dataset.

The UCI target was converted into a binary target:

- `0 → No Disease`
- `1–4 → Disease`

### 14. Dataset Overlap Verification

Before interpreting the UCI results as independent validation, an exact feature-level overlap check was performed.

The analysis found:

- Primary training dataset unique feature rows: **918**
- UCI evaluation rows: **303**
- Exact overlapping rows: **303**

Therefore, all 303 UCI records were found to overlap with records already present in the primary dataset.

Because of this overlap, the UCI evaluation is **not considered independent external validation**.

Instead, it is reported as a:

> **Cross-Source Consistency Check**

This verification prevents the cross-source results from being incorrectly presented as independent validation.

### 15. Results Storage

The project stores important evaluation outputs inside the `results/` directory.

The saved files include:

- External validation prediction results
- Final model performance results
- Validation summary
- ROC curve visualization

---

## Data Quality Analysis

The original dataset contained:

- **1,190 rows**
- **12 columns**
- **0 missing values**
- **272 duplicate rows**

The duplicate records represented approximately **22.86%** of the original dataset.

After removing duplicates:

- **918 unique records** remained.
- No duplicate records remained.
- No missing values were present.

This cleaning step was performed before splitting the data and training the models.

---

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the characteristics of the dataset.

### Target Distribution

The target variable was analyzed to understand the distribution between:

- No Disease
- Disease

After cleaning, the dataset contained:

- No Disease: **410**
- Disease: **508**

### Numerical Feature Analysis

The numerical features were analyzed using distributions and box plots.

The numerical features include:

- Age
- Resting blood pressure
- Cholesterol
- Maximum heart rate
- Oldpeak

### Categorical Feature Analysis

Categorical variables were analyzed using count plots to understand their distributions.

These include:

- Sex
- Chest pain type
- Fasting blood sugar
- Resting ECG
- Exercise angina
- ST slope

### Correlation Analysis

A correlation heatmap was generated to study relationships among numerical and encoded clinical variables.

---

## Data Preprocessing

The dataset contains both numerical and categorical variables.

### Numerical Preprocessing

`StandardScaler` was applied to:

- age
- resting bp s
- cholesterol
- max heart rate
- oldpeak

### Categorical Preprocessing

`OneHotEncoder` was applied to:

- sex
- chest pain type
- fasting blood sugar
- resting ecg
- exercise angina
- ST slope

The preprocessing pipeline was implemented using Scikit-learn's `ColumnTransformer`.

---

## Train-Test Split

The cleaned dataset was divided using an 80:20 stratified split.

Training Set → 80%

Testing Set → 20%

Random State → 42

The test set was kept separate from model training and was used to obtain the main performance results.

---

## Machine Learning Models

Four classification algorithms were evaluated.

### 1. Logistic Regression

Used as a baseline linear classification model.

### 2. Decision Tree

Used to model non-linear relationships through decision-based splitting.

### 3. Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

### 4. Gradient Boosting

An ensemble technique that builds models sequentially to improve prediction performance.

---

## Model Comparison

The following results were obtained on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 90.22% | 88.89% | 94.12% | 91.43% | 93.92% |
| Gradient Boosting | 89.13% | 90.20% | 90.20% | 90.20% | 93.19% |
| Logistic Regression | 88.59% | 87.16% | 93.14% | 90.05% | 93.01% |
| Decision Tree | 80.43% | 82.35% | 82.35% | 82.35% | 85.31% |

---

## Best Model

The **Random Forest Classifier** was selected as the final model.

### Held-Out Test Set Performance

- **Accuracy:** 90.22%
- **Precision:** 88.89%
- **Recall:** 94.12%
- **F1-Score:** 91.43%
- **ROC-AUC:** 93.92%

The model achieved a particularly high recall of **94.12%**, meaning that it correctly identified most of the disease-positive cases in the held-out test set.

---

## Confusion Matrix

The Random Forest confusion matrix on the held-out test set was:

    [[70, 12],
     [ 6, 96]]

This represents:

- True Negatives: 70
- False Positives: 12
- False Negatives: 6
- True Positives: 96

The model correctly classified **166 out of 184** test cases.

---

## Classification Report

The Random Forest classification performance on the held-out test set was:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Disease | 0.92 | 0.85 | 0.89 | 82 |
| Disease | 0.89 | 0.94 | 0.91 | 102 |
| **Accuracy** | | | **0.90** | **184** |

---

## Feature Importance

Random Forest feature importance was analyzed to identify influential variables.

The most important model features included:

| Feature | Importance |
|---|---:|
| ST slope_1 | 0.1363 |
| ST slope_2 | 0.1082 |
| cholesterol | 0.0955 |
| max heart rate | 0.0947 |
| oldpeak | 0.0881 |
| chest pain type_4 | 0.0774 |
| age | 0.0703 |
| resting bp s | 0.0651 |
| exercise angina_1 | 0.0524 |
| exercise angina_0 | 0.0523 |

These values represent model-derived feature importance and should not be interpreted as medical causation.

---

## Cross-Source Consistency Check

To further evaluate the trained model, the UCI Heart Disease dataset was used.

### UCI Dataset

- Dataset: UCI Heart Disease
- Records: **303**
- Features used: **11**
- Target converted to binary:
  - 0 → No Disease
  - 1–4 → Disease

### Cross-Source Results

The model produced the following results:

- Accuracy: **97.36%**
- Precision: **97.12%**
- Recall: **97.12%**
- F1-Score: **97.12%**
- ROC-AUC: **98.62%**

Confusion matrix:

    [[160, 4],
     [  4, 135]]

However, an exact feature-level overlap analysis showed that:

    Primary dataset unique rows : 918
    UCI evaluation rows         : 303
    Exact overlapping rows      : 303

Therefore, these results **must not be interpreted as independent external validation**.

They are retained as a **cross-source consistency check** because the UCI records overlap with the primary dataset.

---

## ROC-AUC Analysis

ROC curves were used to evaluate the ability of the classifier to distinguish between disease and no-disease cases across different classification thresholds.

### Held-Out Test Set

**Random Forest ROC-AUC: 0.9392**

### Cross-Source Consistency Check

**ROC-AUC: 0.9862**

The cross-source ROC-AUC is reported only as a consistency result because the underlying records overlap with the primary dataset.

---

## Results Directory

The `results/` directory contains the generated evaluation outputs.

    results/
    ├── external_roc_curve.png
    ├── external_validation_results.csv
    ├── final_model_results.csv
    └── validation_summary.csv

### File Descriptions

**external_roc_curve.png**

ROC curve generated from the UCI cross-source consistency check.

**external_validation_results.csv**

Contains individual UCI records with:

- Actual target
- Predicted target
- Disease probability
- No Disease probability
- Actual label
- Predicted label
- Correct prediction indicator
- Probability-based risk band

**final_model_results.csv**

Contains the main performance comparison between:

- Held-out Test Set
- UCI Cross-Source Check

**validation_summary.csv**

Contains the cross-source evaluation and dataset-overlap information.

---

## Project Structure

    CodeAlpha_Disease-Prediction/
    ├── CodeAlpha_Disease_Prediction.ipynb
    ├── README.md
    ├── requirements.txt
    └── results/
        ├── external_roc_curve.png
        ├── external_validation_results.csv
        ├── final_model_results.csv
        └── validation_summary.csv

---

## How to Run

### Option 1: Google Colab

1. Open the project notebook in Google Colab.
2. Install the required Python packages.
3. Run the notebook cells sequentially.
4. The primary dataset will be downloaded using KaggleHub.
5. The machine learning models will be trained.
6. Evaluation metrics and visualizations will be generated.
7. Results will be saved in the `results/` directory.

### Option 2: Local Environment

Clone the repository:

    git clone https://github.com/sureshsaripalle2004/CodeAlpha_Disease-Prediction.git

Navigate to the project directory:

    cd CodeAlpha_Disease-Prediction

Install the dependencies:

    pip install -r requirements.txt

Run the Jupyter Notebook:

    jupyter notebook

Open:

    CodeAlpha_Disease_Prediction.ipynb

and execute the cells sequentially.

---

## Requirements

The required Python packages are listed in `requirements.txt`.

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    kagglehub
    ucimlrepo

---

## Reproducibility

The project uses fixed random states where applicable to improve reproducibility.

The main train-test split uses:

    random_state = 42

The Random Forest model also uses a fixed random state.

The preprocessing and model training operations are implemented through Scikit-learn pipelines.

---

## Future Improvements

Future work can improve the project by:

- Testing the model on a genuinely independent clinical dataset.
- Applying hyperparameter optimization.
- Performing cross-validation and systematic model tuning.
- Exploring XGBoost and other advanced ensemble models.
- Applying explainable AI techniques such as SHAP.
- Performing calibration analysis for predicted probabilities.
- Evaluating fairness across different demographic groups.
- Developing a secure prediction interface for research demonstration.
- Increasing dataset size and diversity.
- Conducting prospective clinical validation with appropriate ethical and medical oversight.

---

## Conclusion

This project developed an end-to-end machine learning pipeline for heart disease prediction using patient clinical features.

The workflow included:

- Dataset acquisition
- Data quality assessment
- Duplicate removal
- Exploratory data analysis
- Feature preprocessing
- Model development
- Model comparison
- Performance evaluation
- Feature importance analysis
- Cross-source consistency checking

Four machine learning models were compared: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

Among these models, **Random Forest achieved the best overall performance on the held-out test set**, with:

- **90.22% Accuracy**
- **88.89% Precision**
- **94.12% Recall**
- **91.43% F1-Score**
- **93.92% ROC-AUC**

The project also demonstrated responsible evaluation by checking the UCI dataset for overlap before interpreting its results. Since all 303 UCI records overlapped with records in the primary dataset, the UCI results were correctly classified as a **cross-source consistency check rather than independent external validation**.

Overall, the project demonstrates how machine learning can be applied to structured clinical data for disease prediction while emphasizing appropriate evaluation, reproducibility, and the limitations of predictive models in healthcare.

---

## Author

**Saripalle Suresh**

M.Tech Artificial Intelligence & Robotics  
CBIT, Hyderabad

GitHub: `sureshsaripalle2004`

---

## Internship Project

This project was developed as part of the **CodeAlpha Machine Learning Internship – 2026**.

**Project:** Disease Prediction Using Machine Learning

**Repository:** `CodeAlpha_Disease-Prediction`

---

## Google colab
https://colab.research.google.com/drive/1wzHNhplgTu9DfT0BNo7EGE56g8Ewwfrq?usp=sharing
