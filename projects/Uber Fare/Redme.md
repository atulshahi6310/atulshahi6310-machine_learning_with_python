# 🚕 Uber Fare Price Prediction Machine Learning Project
---
This repository contains a complete **machine learning pipeline** for predicting Uber fare prices using linear and regularized regression models. The goal of the project is to explore the strengths and limitations of linear models on a real-world pricing dataset and demonstrate a clean ML workflow from data cleaning to model evaluation.

---

## 📁 Project Structure

```
📦Uber Fare
 ├── Uber_Fare_Prediction.ipynb       # Jupyter Notebook
 ├── uber_training_dataset_clean.csv  # Cleaned dataset
 ├── README.md 
 ├── model_training.ipynb ## model training notebook
 ├── uber.csv ## unclean data sets
 # Project documentation
```

---

## 📌 Project Summary

This project uses the **Uber Fare Prediction dataset from Kaggle** and follows an end-to-end data science workflow:

1. **Data Cleaning & Preprocessing**
   • Removed irrelevant columns and handled missing values
   • Corrected inconsistencies and outliers
   • Final clean dataset saved as `uber_training_dataset_clean.csv`

2. **Exploratory Data Analysis (EDA)**
   • Visual exploration of relationships between features and fare
   • Feature distributions, correlation analysis, and insights

3. **Feature Engineering**
   • Extracted meaningful variables such as distance and datetime components
   • Prepared model-ready features

4. **Train–Test Split**
   • Separated data into training and testing sets

5. **Modeling**
   • Built regression models to understand performance on this dataset:

   * Linear Regression
   * Ridge Regression
   * Lasso Regression
   * ElasticNet Regression

6. **Pipeline Implementation**
   • Applied `StandardScaler` for feature scaling
   • Used scikit-learn **Pipeline** to combine preprocessing and models
   • Compared model performance using **MAE, RMSE, and R² score**

7. **Model Evaluation**
   • Results presented in a comparison table
   • Insights into model limitations and strengths

---

## 📊 Key Insights

* Linear models serve as a useful baseline.
* Regularized regression improves stability and reduces overfitting.
* **This dataset contains complex non-linear relationships**, so simple linear models are limited in prediction performance — making this a great learning dataset to explore model behavior. ([GitHub][1])

---

## 🧠 Why This Project Matters

Rather than focusing solely on *highest accuracy*, this project emphasizes **understanding model behavior** on real-world regression problems. It teaches:

✔ How to build clean, reproducible ML pipelines
✔ Why linear models may fall short on non-linear datasets
✔ How regularization affects model performance
✔ How to compare multiple models consistently

---

## 🛠️ Tools & Technologies

* Python
* Pandas, NumPy
* scikit-learn (Pipeline, Regression models)
* Matplotlib / Seaborn
* Jupyter Notebook
* Kaggle Dataset

---

## 📌 How to Run

1. Clone the repository:

```bash
git clone https://github.com/atulshahi6310/machine_learning_with_python.git
```

2. Open the notebook:

```bash
cd machine_learning_with_python/projects/Uber Fare
jupyter notebook Uber_Fare_Prediction.ipynb
```

3. Execute all cells from start to finish.

---

## 🚀 Future Enhancements

✔ Add **tree-based models** (Random Forest, Gradient Boosting)
✔ Try **XGBoost / LightGBM** for better performance
✔ Include **hyperparameter tuning** like GridSearchCV
✔ Deploy a **Streamlit app** for real-time predictions

---

## 📝 Conclusion

This project showcases a full machine learning workflow from raw data to model evaluation, with a focus on interpreting results and understanding why certain models work better on specific types of data.

---

