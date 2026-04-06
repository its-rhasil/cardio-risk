# Heart Disease Prediction using Machine Learning

## Project Overview
This project focuses on predicting the presence of heart disease using machine learning techniques. The goal is to build a robust and scalable pipeline that can handle different types of data and provide reliable predictions.

---

## Objective
- Predict whether a person has heart disease (Yes / No)
- Handle mixed data types (numerical, categorical, ordinal)
- Build a clean and reusable ML pipeline

---

##  Dataset
The dataset contains health-related features such as:
- BMI  
- Alcohol consumption  
- Smoking history  
- Exercise 
- Diabetes
- Depression
- Arthritis
- Skin and other cancer
- Fruit, green vegetable and fried potato consumption  
- Sex
- General health  
- Age category  

Target Variable:
- Heart_Disease (0 = No, 1 = Yes)

---

## Approach

### 1. Data Preprocessing
A structured preprocessing pipeline was designed using ColumnTransformer:

- Numerical Features
  - Scaled using StandardScaler

- Categorical Features
  - Encoded using OneHotEncoder
  - Used drop='first' to avoid multicollinearity
  - Handled unknown categories using handle_unknown='ignore'

- Ordinal Features
  - Custom ordering applied:
    - General Health: Poor → Excellent  
    - Age Category: Young → Old  
  - Encoded using OrdinalEncoder

---

### 2. Pipeline Design
A full pipeline was built using Pipeline to prevent data leakage:

- Preprocessing step  
- Model step (Logistic Regression)  

This ensures consistent transformations during both training and testing.

---

### 3. Handling Class Imbalance
- Used class_weight='balanced' in Logistic Regression  
- Helps the model give importance to minority class (heart disease cases)

---

### 4. Train-Test Split
- Used stratified splitting to preserve class distribution:

`python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Models Used

### 1. Logistic Regression
- max_iter=1000
- class_weight='balanced'

Chosen as a strong baseline model due to:
- Simplicity  
- Interpretability  
- Good performance on structured data  

---

### 2. Balanced Random Forest (BRF)
- Handles class imbalance internally using resampling
- Captures non-linear relationships

Why used:
- Better performance on imbalanced datasets  
- Robust to noise and feature interactions  

---

### 3. XGBoost (Extreme Gradient Boosting)
- Gradient boosting-based ensemble model
- Highly efficient and powerful for tabular data

Why used:
- Captures complex patterns  
- Often provides state-of-the-art performance  
- Handles feature interactions effectively  

---

##  Key Highlights
- Clean and modular ML pipeline  
- Compared multiple models (Logistic Regression, BRF, XGBoost)  
- Proper handling of numerical, categorical, and ordinal features  
- Prevention of data leakage using Pipeline  
- Consideration of class imbalance  
- Use of domain-aware ordinal encoding  

---

## How to Run

### 1. Clone the repository
```bash
git clone github.com/its-rhasil/cardio-risk
cd cardio-risk
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the notebook
```bash
notebooks/04_modelling.ipynb
``` 
using jupyter notebook or vs code

## Future Improvements
- Hyperparameter tuning for improved model performance  
- Threshold tuning to optimize recall for medical use-cases  
- Model explainability using SHAP or feature importance analysis  
- Deployment as a web application (Flask / FastAPI)  
- Integration with real-time health data inputs  

---

## Learnings
- Importance of structured preprocessing pipelines  
- Handling numerical, categorical, and ordinal data correctly  
- Avoiding data leakage using Pipeline and ColumnTransformer  
- Managing imbalanced datasets effectively  
- Comparing multiple models to select the best approach  

---

## Author
**Muhammed Rhasil**  