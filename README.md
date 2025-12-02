# 📦 Forecast App - Demand Prediction Tool

A lightweight and user-friendly desktop software built with **Electron + Python**, designed to perform sales demand forecasting based on uploaded CSV data. It supports advanced model ensembles, automated hyperparameter optimization, and visualized forecasting results.

---

## ✨ Features

- 📁 Upload your raw CSV sales data
- 🔄 Step-by-step interface: from data aggregation to forecast
- 🧠 Uses hybrid ML models: XGBoost, LightGBM, CatBoost, and more
- 🔧 Auto-tuning via Optuna (Bayesian Optimization)
- 📊 SHAP feature importance visualization
- 🖼 Forecast chart generated and displayed instantly

---

## 🛠 Installation Guide

### Option 1: Use the Packaged `.exe`

> Simply double-click the installer or `forecast_app.exe` 

---

### Option 2: Install required Python packages❗❗❗

Make sure you have Python 3.8~3.12. Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

1. 📂 Click “Select CSV File” to upload your sales data
2. 🧮 Click “Step 1: Data Aggregation” to clean and enrich the raw data
3. 📈 Click “Step 2: Run Forecast” to generate predictions
4. 🖼 Forecast image and performance metrics will be shown directly in the app

Your forecast results will be saved in the working directory as:

- `predictions_enhanced_plot.png`
- `forecast_results.txt`

📂The csv file in the Example folder can be used as an example to test the functionality of the software.


---

## 📦 Python Requirements

If you're using the scripts directly, make sure these packages are installed:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
autogluon.tabular
tensorflow
scikit-optimize
statsmodels
pmdarima
joblib
shap
optuna
```

---

## 📌 Notes

The uploaded csv file needs to have the following content:

---

Date: YY/MM/DD
Quantity Sold (kilo)
Unit Selling Price (RMB/kg)
Discount (Yes/No)

---


