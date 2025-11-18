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
* **Public** This dataset is intended for public access and use.
* **Description:** The dataset contains detailed information on all licensed alternative fuel vehicles in Washington State.
* Metadata Created Date	November 10, 2020
* Metadata Updated Date	October 18, 2025
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
* Loading the dataset (`.csv`). URL: https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD
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
>
> ```bash
> pip install pandas numpy scikit-learn matplotlib seaborn
>
> 
## License
This project is developed for educational purposes as part of the ML Zoomcamp course.
Open Data Commons Open Database License (ODbL) v1.0.  Please see the follow link:
https://opendatacommons.org/licenses/odbl/1-0/

## Acknowledgments
DataTalksClub for the Machine Learning Zoomcamp

Data.Gov for providing the data


# STEPS

# --- 1. Environment Setup and Data Loading ---
# This section loads the necessary libraries and attempts to download the dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# URL for the Electric Vehicle Population Data dataset (raw CSV link)
DATA_URL = 'https://raw.githubusercontent.com/datasets/ev-population-data/main/data/ev_population_data.csv'

# Load the data directly from the URL
try:
    df = pd.read_csv(DATA_URL)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame() # Use an empty DataFrame if loading fails

# --- 2. Initial Exploration and Cleaning ---

if not df.empty:
    print("\n--- First 5 rows ---")
    print(df.head())

    # Identify the target column
    TARGET_COLUMN = 'Electric Vehicle Type'
    
    # Simple handling of missing values:
    # Fill NaN in the 'Electric Range' column with the mean.
    if 'Electric Range' in df.columns:
        df['Electric Range'] = df['Electric Range'].fillna(df['Electric Range'].mean())

    # Drop rows with missing values in the target variable
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    print(f"\nDistribution of the target variable ('{TARGET_COLUMN}'):")
    print(df[TARGET_COLUMN].value_counts())

    # --- 3. Preparation for Decision Trees ---
    
    # Define the predictor variables (features)
    FEATURES = ['Model Year', 'Electric Range', 'Make', 'County']

    # Subset the data
    data = df[FEATURES + [TARGET_COLUMN]].copy()

    # One-Hot Encoding for categorical variables ('Make' and 'County')
    data_encoded = pd.get_dummies(data, columns=['Make', 'County'], drop_first=True)
    
    # Prepare the X (features) and y (target) matrices
    X = data_encoded.drop(TARGET_COLUMN, axis=1)
    y = data_encoded[TARGET_COLUMN]

    # Split the data into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData ready for modeling:")
    print(f"  X_train size: {X_train.shape}")
    print(f"  X_test size: {X_test.shape}")

    # --- 4. Decision Tree Modeling and Prediction ---

    # Create and train the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Decision Tree Results ---")
    print(f"Model Accuracy on the test set: {accuracy:.4f}")
    
else:
    print("Could not proceed with modeling due to data loading error.")




