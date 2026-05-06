import pandas as pd
import streamlit as st # Import streamlit for st.error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso # Added Ridge and Lasso
from sklearn.metrics import accuracy_score, r2_score
from pandas.api.types import is_numeric_dtype
from xgboost import XGBClassifier, XGBRegressor # Import XGBoost models
from catboost import CatBoostClassifier, CatBoostRegressor # Import CatBoost models
from lightgbm import LGBMClassifier, LGBMRegressor # Import LightGBM models
from sklearn.naive_bayes import GaussianNB # Import Gaussian Naive Bayes
from sklearn.svm import SVC, SVR # Import SVM models

def train_model(df, target_col, selected_model_name="Random Forest"):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = None
    metric_name = ""
    model_type = ""

    # Determine if it's a classification or regression problem
    if is_numeric_dtype(y) and y.nunique() > 20: # Heuristic for continuous targets
        model_type = "regression"
    else:
        model_type = "classification"

    if model_type == "regression":
        if selected_model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif selected_model_name == "Linear Regression":
            model = LinearRegression()
        elif selected_model_name == "Ridge Regression":
            model = Ridge(random_state=42)
        elif selected_model_name == "Lasso Regression":
            model = Lasso(random_state=42)
        elif selected_model_name == "XGBoost":
            model = XGBRegressor(random_state=42)
        elif selected_model_name == "CatBoost":
            model = CatBoostRegressor(random_state=42, verbose=0) # verbose=0 to suppress CatBoost output
        elif selected_model_name == "LightGBM":
            model = LGBMRegressor(random_state=42)
        elif selected_model_name == "SVM":
            model = SVR()
        # Add other regression models here
        metric_name = "R2 Score"
    else: # classification
        if selected_model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif selected_model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
        elif selected_model_name == "XGBoost":
            model = XGBClassifier(random_state=42)
        elif selected_model_name == "CatBoost":
            model = CatBoostClassifier(random_state=42, verbose=0) # verbose=0 to suppress CatBoost output
        elif selected_model_name == "LightGBM":
            model = LGBMClassifier(random_state=42)
        elif selected_model_name == "Naive Bayes":
            model = GaussianNB()
        elif selected_model_name == "SVM":
            model = SVC(random_state=42)
        # Add other classification models here
        metric_name = "Accuracy"

    if model is None:
        st.error(f"No suitable model found for {selected_model_name} and {model_type} problem.")
        return None, None

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if model_type == "regression":
        metric_value = r2_score(y_test, predictions)
    else:
        metric_value = accuracy_score(y_test, predictions)

    return metric_value, metric_name
