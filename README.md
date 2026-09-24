# alzheimers-diagnosis-classification
Machine learning classification of Alzheimer's diagnosis using Logistic Regression, Random Forest and XGBoost, with EDA, model comparison and interpretation.
> **Educational project. Not a clinical tool or medical advice.** The dataset shows signs of being synthetic (see Limitations section below), so results should not be assumed to transfer to real patients.
Overview
The notebook loads a tabular dataset of 2,149 older adults (ages 60-90) and predicts a binary `Diagnosis` label (0 = no diagnosis, 1 = diagnosis; about 35% positive). It covers:
Data quality checks (missing values, duplicates, class balance)
Exploratory data analysis (distributions, scatter plots, correlation heatmap)
Preprocessing (one-hot encoding of `Ethnicity`, stratified 75/25 split, standard scaling for Logistic Regression)
Three models of increasing flexibility, each tuned with 5-fold cross-validated grid search (refit on F1)
Model interpretation (shallow decision tree, XGBoost feature importance)
A risk-factor-only experiment with the clinical assessment variables removed
ROC-AUC comparison of all three models
Results (test set)
Model	F1	ROC-AUC
Logistic Regression	0.756	0.895
Random Forest	0.884	0.943
XGBoost	0.933	0.944
XGBoost is the strongest model, but its AUC is only marginally above Random Forest. With 190 positive cases in the test set, that gap is within noise.
Key findings
Five variables drive almost all predictive signal: `FunctionalAssessment`, `ADL`, `MMSE`, `MemoryComplaints` and `BehavioralProblems`.
Lifestyle, demographic, cholesterol, blood pressure and medical-history variables show almost no association with diagnosis, including age.
Risk-factor-only experiment: after removing the five key variables, XGBoost falls to F1 of about 0.03 and accuracy of about 0.64, which is no better than always predicting "negative" (about 65%). The remaining variables carry essentially no usable signal.
Limitations
The model recognises a diagnosis rather than forecasting risk. The five key variables are clinical assessment and symptom measures, which are close to how the diagnosis is made.
The data appear synthetic. Many variables are evenly spread across their ranges, predictors are almost uncorrelated with each other, and age has no correlation with diagnosis. Confirm the provenance of your copy before drawing real-world conclusions.
Single train/test split. Metrics carry wide uncertainty. Repeated cross-validation or bootstrapped confidence intervals would show whether XGBoost and Random Forest truly differ.
Feature importance uses XGBoost's default split counts, which favours continuous variables. Permutation importance or SHAP would be more reliable.
Repository structure
```
alzheimers-diagnosis-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook.ipynb
└── data/
```
Getting started
```bash
git clone https://github.com/Pntiamoah27/alzheimers-diagnosis-classification.git
cd alzheimers-diagnosis-classification
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
Data
The notebook expects `alzheimers_disease_data.csv` in the working directory (or `data/`). Source: add a link to the dataset and check its license before redistributing the file.
Possible improvements
Print and save `best_params_` for each search, and save the fitted models
Tune the decision threshold to raise recall if missed cases are costly
Check probability calibration
Use permutation importance or SHAP
Add confidence intervals via repeated cross-validation or bootstrapping
https://github.com/Pntiamoah27/
