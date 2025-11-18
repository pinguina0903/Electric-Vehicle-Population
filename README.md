# 🚗 ML Project: Electric Vehicle Population Data Classification
## 📌 Introduction

This Machine Learning project focuses on analyzing the **Electric Vehicle Population Data** registered in Washington State (USA). The main objective is to build a classification model to identify key patterns and features that determine the type of electric vehicle (Battery Electric Vehicle - BEV, or Plug-in Hybrid Electric Vehicle - PHEV) or its manufacturer.

The model selection is centered on **Decision Trees**, given their interpretability, their ability to handle mixed categorical and numerical data well, and the nature of the dataset.

## 🎯 Project Objectives

1.  **EV (Electric Vehicle) Type Classification:** Develop a model to classify whether a registered vehicle is a **BEV (Battery Electric Vehicle)** or a **PHEV (Plug-in Hybrid Electric Vehicle)**.
2.  **Pattern Analysis:** Identify which characteristics (e.g., county, model year, electric range) are the **most important predictors** for the classification.
3.  **Visualization:** Create visualizations that show the structure of the Decision Tree, making the model **interpretable** and easy to explain.
4.  **Model Evaluation:** Measure the performance of the Decision Tree using appropriate metrics such as Accuracy, Confusion Matrix, and F1-Score.

---

## 💾 Dataset

* **Dataset Name:** Electric Vehicle Population Data
* **Source:** [Washington State Department of Licensing (via Data.gov)](https://catalog.data.gov/dataset/electric-vehicle-population-data)
* **Description:** The dataset contains detailed information on all licensed alternative fuel vehicles in Washington State.
* **Key Variables of Interest:**
    * `Electric Vehicle Type` (Target variable for classification).
    * `County` / `City` (Geographic variables).
    * `Make` (Manufacturer).
    * `Model Year`.
    * `Electric Range` (Suitable for prediction).

---

## 🛠️ Methodology (Decision Trees)

The project will be developed through the following stages:

### 1. Data Preparation and Cleaning
* Loading the dataset (`.csv`).
* Handling missing values (especially in `Electric Range` and geographic variables).
* Encoding categorical variables (e.g., *One-Hot Encoding* for `Make` or `County`), required for Scikit-learn implementation.

### 2. Exploratory Data Analysis (EDA)
* Visualization of the distribution of the target variable (`Electric Vehicle Type`).
* Bar charts to analyze the EV distribution by `County` and `Make`.

### 3. Decision Tree Modeling
* Splitting the data into training and testing sets (Train/Test Split).
* Training a `DecisionTreeClassifier` model.
* **Hyperparameter Tuning:** Optimizing the model (e.g., `max_depth`, `min_samples_leaf`) using cross-validation to prevent *overfitting*.

### 4. Evaluation and Results
* Generation of the **Confusion Matrix**.
* Calculation of performance metrics (Accuracy, Recall, F1-Score).
* **Interpretation:** Analysis of the **Feature Importance** used by the tree for classification.
* Visualization of the final tree for project interpretability.

---

## 💻 Requirements and Technologies

The project runs in a Python environment and requires the following libraries:

| Package | Purpose |
| :--- | :--- |
| **Python** | Version 3.x |
| **Pandas** | Data loading, manipulation, and cleaning. |
| **NumPy** | Numerical operations and arrays. |
| **Scikit-learn** | Implementation of `DecisionTreeClassifier` and evaluation metrics. |
| **Matplotlib / Seaborn** | Data visualization (EDA) and results. |
| **Graphviz (Optional)** | Graphical visualization of the Decision Tree. |

### Installing Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
