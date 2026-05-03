# 🚗 Car.LK — Sri Lanka Car Market Analysis Dashboard

![Car.LK Banner](logo.png)

An interactive data analysis and machine learning dashboard for the Sri Lankan second-hand car market, built with **Streamlit** and **Plotly**.

> **ST3011 — Statistical Data Analysis | Individual Project**
> Dinusha Priyashan | s16798 | Department of Statistics, University of Colombo

---

## 📌 Project Overview

Car.LK is a web-based dashboard that analyses car listing data from the Sri Lankan market. It provides interactive visualisations, statistical hypothesis tests, machine learning price predictions, and factor/cluster analysis - all in one place.

---

## 🚀 Live App

👉 [View on Streamlit Community Cloud](https://st3011dataanalysiscarlk-lfde7nabgnf7yuw9esrxbb.streamlit.app/)

---

## 📂 Project Structure

```
├── app.py                        # Main Streamlit application entry point
├── train_model.py                # Script to train and save ML models
├── factor_analysis.py            # Script to run FAMD & clustering analysis
├── data_prep.py                  # Data loading and cleaning utilities
├── best_model.pkl                # Trained best ML model (Random Forest)
├── model_diagnostics.pkl         # Model diagnostics for the app
├── factor_analysis_results.pkl   # Pre-computed FAMD & cluster results
├── car_price_dataset .csv        # Raw dataset
├── model_report.txt              # Human-readable model performance report
├── logo.png                      # App logo
├── requirements.txt              # Python dependencies
├── pages/
│   ├── data_explorer.py          # 📊 Data Explorer page
│   ├── visualisations.py         # 📉 Visualisations page
│   ├── regression.py             # 📈 Car Price Prediction page
│   ├── hypothesis_testing.py     # 🧪 Hypothesis Testing page
│   └── help.py                   # ❓ Help page
├── utils/
│   ├── data_loader.py            # Cached data loading function
│   └── config.py                 # Shared Plotly layout config
└── assets/
    └── styles.css                # Custom CSS styling
```

---

## 📊 Features

### 📊 Data Explorer
- Browse and filter the full car listings dataset
- Summary statistics and data quality overview
- Filter by brand, fuel type, province, year, and price range

### 📉 Visualisations
- Price distribution by brand, fuel type, and province
- Year of manufacture vs price trends
- Mileage vs price scatter plots
- Interactive Plotly charts throughout

### 📈 Car Price Prediction
- Predict car prices using a trained **Random Forest** model
- Compare 7 regression models: Linear, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting
- SHAP feature importance charts
- Residual diagnostics and actual vs predicted plots

### 🧪 Hypothesis Testing
- Statistical tests on car price differences across groups
- Post-hoc analysis using `scikit_posthocs`
- Results presented with clear interpretations

### 🔍 Factor & Cluster Analysis
- **FAMD** (Factor Analysis of Mixed Data) using `prince`
- Scree plot and cumulative variance analysis
- K-Means clustering (k=2–10) with silhouette scoring
- Per-cluster scatter and silhouette plots

---

## 🤖 Model Performance

| Model | Test RMSE | Test R² | Test MAE |
|---|---|---|---|
| Linear Regression | 36.42 | 0.4415 | 18.09 |
| Ridge | 36.46 | 0.4403 | 17.98 |
| Lasso | 36.95 | 0.4253 | 17.96 |
| Elastic Net | 36.85 | 0.4283 | 17.62 |
| Decision Tree | 29.07 | 0.6442 | 11.21 |
| **Random Forest ✅ BEST** | **27.45** | **0.6829** | **10.82** |
| Gradient Boosting | 27.85 | 0.6734 | 11.03 |

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/DPriyashan/ST_3011_Data_Analysis_Individule_project_Car.LK.git
cd ST_3011_Data_Analysis_Individule_project_Car.LK
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (run once)
```bash
python train_model.py
```

### 4. Run factor analysis (run once)
```bash
python factor_analysis.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
plotly
pandas
numpy
scikit-learn
scipy
xgboost
shap
prince
pillow
scikit_posthocs
```

---

## 📋 Dataset

The dataset contains car listings scraped from the Sri Lankan car market with the following features:

| Feature | Description |
|---|---|
| Brand | Car manufacturer |
| Model | Car model name |
| YOM | Year of manufacture |
| Price | Listed price (LKR millions) |
| Millage(KM) | Odometer reading |
| Engine (cc) | Engine displacement |
| Fuel Type | Petrol / Diesel / Hybrid / Electric |
| Gear | Manual / Automatic |
| Condition | New / Reconditioned / Used |
| Town | Listing location |
| Leasing | Leasing availability |

---

## 👤 Author

**Dinusha Priyashan**
Student ID: s16798
Department of Statistics
University of Colombo
ST3011 — Statistical Data Analysis

---

## 📄 License

This project is submitted as an individual academic assignment for ST3011 at the University of Colombo. All rights reserved © 2026 Dinusha Priyashan.
