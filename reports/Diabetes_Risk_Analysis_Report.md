# Diabetes Risk Analysis and Prediction Report

## Executive Summary

This project analyzes a real public diabetes dataset to identify the health indicators most associated with diabetes and build a predictive model that can estimate diabetes risk from patient characteristics.

The analysis found that Glucose is the strongest risk signal, followed by BMI and Age. Patients with higher glucose values and higher BMI showed visibly higher diabetes rates. The best-performing model was **Random Forest**, selected using ROC-AUC on the test set.

## Dataset Overview

- Source: Pima Indians Diabetes Database, mirrored as a public CSV by Plotly datasets.
- Records: 768
- Predictor features: 8
- Target: `Outcome` where 1 = diabetes and 0 = no diabetes.
- Class distribution: {0: 500, 1: 268}

## Business Problem

Healthcare teams often need to identify which patient indicators deserve the most attention during screening. This project answers three practical questions:

1. Which biometric variables are most associated with diabetes?
2. Can diabetes risk be predicted from standard patient measurements?
3. Which indicators should be highlighted in a business dashboard for health monitoring?

## Methodology

### Data Cleaning and Preparation

- Checked missing values, data types, duplicate rows, and target distribution.
- Found no explicit missing values, but detected physiologically invalid zeros in selected medical fields.
- Treated zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI as missing values.
- Applied median imputation to preserve all records.
- Used IQR capping for outliers to reduce extreme-value influence.
- Engineered interpretable features: BMI category, age group, and glucose risk group.

Duplicate records detected: 0

Zero values treated as missing:
{
  "Glucose": 5,
  "BloodPressure": 35,
  "SkinThickness": 227,
  "Insulin": 374,
  "BMI": 11
}

### Exploratory Analysis

Descriptive statistics:

| index | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- |
| Pregnancies | 3.84 | 3.0 | 3.34 | 0.0 | 13.5 |
| Glucose | 121.66 | 117.0 | 30.44 | 44.0 | 199.0 |
| BloodPressure | 72.36 | 72.0 | 11.7 | 40.0 | 104.0 |
| SkinThickness | 28.87 | 29.0 | 7.44 | 14.5 | 42.5 |
| Insulin | 124.69 | 125.0 | 7.91 | 112.88 | 135.88 |
| BMI | 32.39 | 32.3 | 6.67 | 18.2 | 50.25 |
| DiabetesPedigreeFunction | 0.46 | 0.37 | 0.29 | 0.08 | 1.2 |
| Age | 33.2 | 29.0 | 11.63 | 21.0 | 66.5 |

Average values by outcome:

| Outcome | Glucose | BMI | Age | BloodPressure | Insulin |
| --- | --- | --- | --- | --- | --- |
| No Diabetes | 110.68 | 30.87 | 31.14 | 70.91 | 123.15 |
| Diabetes | 142.13 | 35.24 | 37.05 | 75.06 | 127.57 |

Correlation with diabetes outcome:

| index | Correlation with Outcome |
| --- | --- |
| Glucose | 0.493 |
| BMI | 0.313 |
| Insulin | 0.266 |
| Age | 0.243 |
| Pregnancies | 0.22 |
| SkinThickness | 0.22 |
| DiabetesPedigreeFunction | 0.185 |
| BloodPressure | 0.169 |

## Visual Findings

Recommended dashboard charts:

- Feature distributions by outcome: `01_feature_distributions.png`
- Outcome boxplots: `02_outcome_boxplots.png`
- Correlation heatmap: `03_correlation_heatmap.png`
- Glucose vs BMI scatterplot: `04_scatter_glucose_bmi.png`
- Pairplot of core indicators: `05_pairplot_key_indicators.png`
- Model comparison and feature importance charts.

## Predictive Modeling

Models tested:

- Logistic Regression
- Random Forest
- Gradient Boosting, used as a practical boosting alternative because XGBoost is optional and may not be available in all client environments.

Performance comparison:

| Model | Best_Params | CV_ROC_AUC | Accuracy | Precision | Recall | F1 | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | {"model__max_depth": 5, "model__min_samples_leaf": 5, "model__n_estimators": 150} | 0.839 | 0.734 | 0.659 | 0.5 | 0.568 | 0.812 |
| Gradient Boosting | {"model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 120} | 0.829 | 0.714 | 0.619 | 0.481 | 0.542 | 0.811 |
| Logistic Regression | {"model__C": 0.1} | 0.85 | 0.701 | 0.583 | 0.519 | 0.549 | 0.806 |

Best model: **Random Forest**

## Explainability

Feature importance:

| Feature | Importance |
| --- | --- |
| Glucose | 0.399 |
| BMI | 0.177 |
| Age | 0.115 |
| DiabetesPedigreeFunction | 0.084 |
| Insulin | 0.076 |
| Pregnancies | 0.059 |
| SkinThickness | 0.046 |
| BloodPressure | 0.043 |

The strongest indicators should be interpreted as screening signals rather than clinical diagnosis. The model is useful for prioritization, reporting, and risk segmentation, but final diagnosis requires medical assessment.

## Business Insights

- Glucose is the clearest driver of diabetes risk in this dataset.
- BMI adds an important metabolic-risk signal and should be monitored alongside glucose.
- Age is associated with increased risk and helps segment patients into prevention groups.
- Blood Pressure and Insulin provide supporting context but are less predictive than glucose in this analysis.

## Recommendations

- Build a screening dashboard with KPI cards for diabetes rate, average glucose, average BMI, and high-risk patient count.
- Segment patients by glucose risk group, BMI category, and age group.
- Use the predictive model as a prioritization tool to flag records for follow-up.
- Communicate model outputs as risk estimates, not as a medical diagnosis.
