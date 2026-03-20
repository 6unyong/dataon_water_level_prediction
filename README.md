# dataon_water_level_prediction
Predicting the water level of Jamsu Bridge using historical AWS weather data and XGBoost.

---

# 🌊 Water Level Prediction

A machine learning project to predict the water level of the Jamsu Bridge (잠수교) using historical water level data and local meteorological conditions. 

This repository contains the complete end-to-end pipeline, including extensive time-series data preprocessing, feature engineering, model training using **XGBoost**, Bayesian hyperparameter tuning, and **Explainable AI (SHAP)** visualization.

## 📌 Project Overview
Predicting river water levels is crucial for flood management and traffic control (especially for submersible bridges like Jamsu Bridge during the monsoon season). This project fuses hourly water level data with AWS (Automatic Weather Station) data from 2020 to 2022 (May to October) to build a robust time-series regression model.

### Key Features
* **Time-Series Preprocessing:** Melts and cleans raw hourly data, handles date-time anomalies, and uses linear interpolation for missing values.
* **Feature Engineering:** Extracts advanced temporal features including month, week, day, and cyclic time features using Sine/Cosine transformations (`sin_time`, `cos_time`) to capture daily seasonality.
* **Target Shifting:** Implements a rolling mean on a shifted target variable (`target_shift`) to construct realistic predictive features.
* **Advanced Tuning:** Uses `Hyperopt` (Tree-structured Parzen Estimator) for Bayesian hyperparameter optimization of the XGBoost model.
* **Model Explainability (XAI):** Utilizes `SHAP` (SHapley Additive exPlanations) values to interpret feature importance and model decisions through waterfall and force plots.

---

## 🛠 Prerequisites & Installation

This project is built using Python, XGBoost, and various data science libraries.

~~~bash
pip install pandas numpy scikit-learn xgboost sktime hyperopt shap matplotlib seaborn
~~~

---

## 📂 Dataset Structure

The project expects yearly water level data and AWS weather data in the root directory. Ensure your files are structured as follows:

~~~text
Project_Root/
├── 2020.csv         # 2020 Water Level Data
├── 2021.csv         # 2021 Water Level Data
├── 2022.csv         # 2022 Water Level Data
├── 2020aws.csv      # 2020 Weather Data (Temp, Rain, Wind, etc.)
├── 2021aws.csv      # 2021 Weather Data
├── 2022aws.csv      # 2022 Weather Data
└── water_level_prediction.ipynb
~~~
*Note: The script automatically filters the data to focus on the active monsoon/typhoon season (May 15th to October 15th).*

---

## 🚀 Modeling & Training Details

* **Algorithm:** `XGBRegressor` (XGBoost)
* **Data Splitting:** Evaluated using both `temporal_train_test_split` (sktime) to respect chronological order, and standard `train_test_split`.
* **Hyperparameter Optimization:** 100 evaluations using `Hyperopt` optimizing for MAPE.
  * *Tuned Parameters:* `max_depth`, `learning_rate`, `n_estimators`, L1/L2 Regularization (`reg_alpha`, `reg_lambda`), and sampling techniques.
* **Evaluation Metrics:** Evaluated using MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and a custom SMAPE (Symmetric MAPE) function.

---

## 👁️ Explainable AI: SHAP Visualizations

To avoid the "black-box" nature of complex tree models, this project integrates SHAP.
* **Waterfall Plots:** Shows how each feature contributed to the specific prediction of a single data point.
* **Force Plots:** Visualizes the push and pull of different features (like cumulative rainfall or current temperature) driving the predicted water level up or down.

*(📸 Tip: Insert a screenshot of your SHAP waterfall plot output here to demonstrate the model's interpretability)*

---

## 💻 How to Run

1. Ensure all 6 dataset CSV files are placed in the same directory as the notebook.
2. Open `water_level_prediction.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells sequentially. The script will automatically parse the Korean date-time formats, merge the datasets, tune the model, and display the SHAP plots.
