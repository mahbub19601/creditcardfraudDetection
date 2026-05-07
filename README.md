# 💳 End-to-End Credit Card Fraud Detection System



A production-ready machine learning pipeline designed to solve the "needle in a haystack" problem of credit card fraud detection. 

This project implements a robust, object-oriented architecture to handle extreme class imbalance (<0.2% fraud rate) using **SMOTE** and **XGBoost**, while strictly enforcing a leakage-free training process.

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Strategic Engineering Decisions](#-strategic-engineering-decisions)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)

---

## 🎯 Overview
Credit card fraud datasets are notoriously imbalanced. If trained normally, a model will just guess "Not Fraud" every time and achieve 99.8% accuracy—missing every actual fraudulent transaction. This project utilizes Synthetic Minority Over-sampling Technique (SMOTE) paired with Gradient Boosted Decision Trees (XGBoost) to accurately identify fraudulent patterns without compromising legitimate transactions.

---

## ✨ Key Features
* **Automated Data Ingestion:** Directly downloads the standard European Credit Card Fraud dataset from Kaggle using `kagglehub`.
* **Leakage-Free Pipeline:** Utilizes `imblearn.pipeline.Pipeline` to ensure SMOTE is *only* applied to the training folds.
* **Optimized XGBoost Classifier:** Configured with `tree_method='hist'` for faster training on large datasets, alongside tuned hyperparameters.
* **Robust Preprocessing:** Handles extreme outliers in transaction amounts using `StandardScaler`.
* **Professional Logging:** Built-in logging module to track pipeline execution states in real-time.

---

## 🧠 Strategic Engineering Decisions

### 1. Leakage Prevention
Standard oversampling is often implemented incorrectly by applying it to the entire dataset before splitting. This leads to **Data Leakage** and artificially inflated test scores. 
> **Solution:** This project utilizes the `imblearn.pipeline.Pipeline`. This ensures that SMOTE is **only applied during the `fit` stage** of the training process. The test data remains purely "real-world," providing an honest evaluation.

### 2. Handling Extreme Imbalance
* **SMOTE:** Synthetically generates new fraud cases by interpolating between existing ones, allowing the model to learn the "boundary" of fraud rather than just memorizing it.
* **XGBoost:** Chosen for its ability to handle complex, non-linear relationships and its built-in support for regularization (`subsample`, `colsample_bytree`), which prevents the model from over-fitting on the synthetic data.

### 3. Metric Selection: Why not Accuracy?
In fraud detection, accuracy is a misleading metric. 
> **Solution:** This system prioritizes **AUPRC (Area Under the Precision-Recall Curve)**. The goal is to maximize **Recall** (catching every fraudster) while maintaining acceptable **Precision** (not blocking legitimate customers).

---

## 🛠️ Tech Stack

* **Core Logic:** `Python 3.10+`
* **Model:** `xgboost` 
* **Sampling:** `imbalanced-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Data Ingestion:** `kagglehub`
* **Scaling & Validation:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **Persistence:** `joblib`

---

## ⚙️ Installation & Setup
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn kagglehub joblib

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/fraud-detection-system.git](https://github.com/yourusername/fraud-detection-system.git)
   cd fraud-detection-system
