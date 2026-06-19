python version 3.11

Sergei Patrushev

# Spam Classification with Machine Learning: A Classifier Shootout

## Overview

This project explores supervised machine learning techniques for email spam detection using the UCI Spambase dataset. The objective is to build, evaluate, and compare multiple classification algorithms while applying industry-standard preprocessing and validation techniques.

The project demonstrates a complete machine learning workflow:

* Data exploration and feature analysis
* Feature scaling and preprocessing
* Dimensionality reduction using PCA
* Training multiple classification models
* Model evaluation using multiple metrics
* Cross-validation for robust performance estimation
* Feature importance analysis
* Deployment-ready prediction pipelines

---

## Dataset

**Source:** UCI Machine Learning Repository – Spambase Dataset

The dataset contains:

* 4,601 email messages
* 57 numerical features
* 1 binary target variable

Target labels:

* **1 = Spam**
* **0 = Ham (Not Spam)**

Features represent:

* Word frequencies
* Character frequencies
* Capital letter statistics

---

## Project Objectives

* Investigate characteristics that distinguish spam from legitimate email.
* Compare multiple machine learning classifiers.
* Evaluate the impact of feature scaling.
* Evaluate dimensionality reduction using PCA.
* Analyze model stability using cross-validation.
* Build reusable machine learning pipelines.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Machine Learning Workflow

### 1. Exploratory Data Analysis

Performed:

* Dataset inspection
* Class distribution analysis
* Feature distribution visualization
* Spam vs Ham comparison using boxplots

Key observations:

* Dataset is moderately imbalanced (~39% spam, ~61% ham)
* Many features are sparse and heavily concentrated near zero
* Feature scales vary significantly
* Certain spam-related features show strong separation between classes

---

### 2. Data Preprocessing

Applied:

* Train/Test split with stratification
* Standardization using `StandardScaler`
* Principal Component Analysis (PCA)

PCA was used to determine the number of components required to retain approximately 90% of the dataset variance.

---

### 3. Classification Models

The following classifiers were trained and compared:

#### K-Nearest Neighbors (KNN)

Evaluated:

* Unscaled features
* Scaled features
* PCA-transformed features

#### Decision Tree Classifier

Evaluated with multiple depths:

* Max depth = 3
* Max depth = 5
* Max depth = 10
* Unlimited depth

#### Random Forest Classifier

Evaluated:

* Standard feature space
* PCA-transformed feature space

#### Logistic Regression

Evaluated:

* Scaled features
* PCA-transformed features

---

## Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example evaluation workflow:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
classification_report()
confusion_matrix()
```

---

## Key Findings

### Feature Scaling Matters

KNN performance improved substantially after standardization because distance-based algorithms are sensitive to feature magnitude.

### PCA Preserved Most Information

PCA reduced dimensionality while maintaining strong predictive performance, although a small decrease in accuracy was observed compared to using all features.

### Tree-Based Models

Decision Trees showed increasing risk of overfitting as depth increased.

Random Forests improved generalization by combining multiple trees and consistently outperformed individual Decision Trees.

### Logistic Regression

Logistic Regression achieved the strongest overall balance of:

* Accuracy
* Precision
* Recall
* Stability across folds

It demonstrated excellent performance after feature scaling.

---

## Cross-Validation Results

5-Fold Cross Validation was used to estimate generalization performance.

Cross-validation confirmed observations from the train/test split:

* Logistic Regression and Random Forest achieved the most reliable results.
* KNN required feature scaling to perform competitively.
* Deep Decision Trees exhibited higher variance and reduced stability.

---

## Feature Importance Analysis

Random Forest feature importance was used to identify the most predictive indicators of spam.

Important features included:

* Spam-related keywords
* Character frequency metrics
* Capitalization statistics

These findings align with common characteristics of unsolicited email.

---

## Prediction Pipelines

To prevent preprocessing inconsistencies and data leakage, reusable Scikit-Learn pipelines were created.

### Logistic Regression Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])
```

### PCA + Logistic Regression Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("classifier", LogisticRegression())
])
```

### Random Forest Pipeline

```python
Pipeline([
    ("classifier", RandomForestClassifier())
])
```

Benefits:

* Reproducibility
* Reduced risk of preprocessing errors
* Easier deployment
* Cleaner code organization

---

## Skills Demonstrated

### Machine Learning

* Classification
* Model Selection
* Hyperparameter Analysis
* Cross Validation
* Performance Evaluation

### Data Science

* Exploratory Data Analysis (EDA)
* Feature Engineering Concepts
* Dimensionality Reduction
* Statistical Interpretation

### Python & Scikit-Learn

* Pipelines
* PCA
* Model Evaluation
* Visualization
* Data Preprocessing

### Software Engineering

* Reusable ML workflows
* Pipeline architecture
* Clean project structure
* Reproducible experiments

---

## Results Summary

| Model               | Notes                                               |
| ------------------- | --------------------------------------------------- |
| KNN (Unscaled)      | Poor performance due to feature scale sensitivity   |
| KNN (Scaled)        | Significant improvement after standardization       |
| Decision Tree       | Good performance but prone to overfitting           |
| Random Forest       | Strong performance and stability                    |
| Logistic Regression | Best overall balance of metrics                     |
| PCA Models          | Slight accuracy reduction with lower dimensionality |

---

## Future Improvements

* Hyperparameter optimization using GridSearchCV
* Feature selection techniques
* ROC-AUC analysis
* Precision-Recall curve analysis
* Model persistence using Joblib
* Deployment via Flask or FastAPI
* Experiment tracking with MLflow

---

## Author

Machine Learning project demonstrating practical experience with data preprocessing, dimensionality reduction, classification algorithms, model evaluation, cross-validation, and deployment-ready pipelines using Scikit-Learn.
