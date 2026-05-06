import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
import pandas as pd
import warnings

from src.utils.helpers import sanitize_col_names # Import from helpers

# Suppress specific CatBoost warnings and other common warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module='catboost')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _get_model_instance(model_name, problem_type, params=None):
    """Helper to get a model instance with optional parameters."""
    if params is None:
        params = {}

    if problem_type == "classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == "Logistic Regression":
            return LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, **params)
        elif model_name == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
        elif model_name == "CatBoost":
            return CatBoostClassifier(random_state=42, verbose=0, allow_categorical_features=True, **params)
        elif model_name == "LightGBM":
            return LGBMClassifier(random_state=42, **params)
        elif model_name == "Naive Bayes":
            return GaussianNB(**params)
        elif model_name == "SVM":
            return SVC(random_state=42, **params)
    elif problem_type == "regression":
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, **params)
        elif model_name == "Linear Regression":
            return LinearRegression(**params)
        elif model_name == "Ridge Regression":
            return Ridge(random_state=42, **params)
        elif model_name == "Lasso Regression":
            return Lasso(random_state=42, **params)
        elif model_name == "XGBoost":
            return XGBRegressor(random_state=42, **params)
        elif model_name == "CatBoost":
            return CatBoostRegressor(random_state=42, verbose=0, allow_categorical_features=True, **params)
        elif model_name == "LightGBM":
            return LGBMRegressor(random_state=42, **params)
        elif model_name == "SVM":
            return SVR(**params)
    raise ValueError(f"Unsupported model: {model_name} for {problem_type} problem type.")

def objective_optuna(trial, X_train, X_test, y_train, y_test, model_name, problem_type):
    # Define hyperparameters to tune for each model
    if problem_type == "classification":
        if model_name == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif model_name == "Logistic Regression":
            C = trial.suggest_loguniform('C', 1e-4, 1e2)
            params = {'C': C}
        elif model_name == "XGBoost":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        elif model_name == "CatBoost":
            iterations = trial.suggest_int('iterations', 50, 200)
            depth = trial.suggest_int('depth', 3, 10)
            l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 1e-2, 1.0)
            params = {'iterations': iterations, 'depth': depth, 'l2_leaf_reg': l2_leaf_reg}
        elif model_name == "LightGBM":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        elif model_name == "Naive Bayes":
            params = {} # No hyperparameters to tune with Optuna for GaussianNB
        elif model_name == "SVM":
            C = trial.suggest_loguniform('C', 1e-4, 1e2)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            params = {'C': C, 'kernel': kernel}
        else:
            raise ValueError(f"Unsupported classification model for Optuna tuning: {model_name}")

        model = _get_model_instance(model_name, problem_type, params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)

    elif problem_type == "regression":
        if model_name == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif model_name == "Linear Regression":
            params = {} # No hyperparameters to tune with Optuna for basic Linear Regression
        elif model_name == "Ridge Regression":
            alpha = trial.suggest_loguniform('alpha', 1e-4, 1e2)
            params = {'alpha': alpha}
        elif model_name == "Lasso Regression":
            alpha = trial.suggest_loguniform('alpha', 1e-4, 1e2)
            params = {'alpha': alpha}
        elif model_name == "XGBoost":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        elif model_name == "CatBoost":
            iterations = trial.suggest_int('iterations', 50, 200)
            depth = trial.suggest_int('depth', 3, 10)
            l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 1e-2, 1.0)
            params = {'iterations': iterations, 'depth': depth, 'l2_leaf_reg': l2_leaf_reg}
        elif model_name == "LightGBM":
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
        elif model_name == "SVM":
            C = trial.suggest_loguniform('C', 1e-4, 1e2)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            params = {'C': C, 'kernel': kernel}
        else:
            raise ValueError(f"Unsupported regression model for Optuna tuning: {model_name}")

        model = _get_model_instance(model_name, problem_type, params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return r2_score(y_test, predictions)


def _get_param_grid(model_name, problem_type):
    """Define parameter grids/distributions for GridSearchCV/RandomizedSearchCV."""
    if problem_type == "classification":
        if model_name == "Random Forest":
            return {
                'n_estimators': [50, 100, 150],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif model_name == "Logistic Regression":
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        elif model_name == "XGBoost":
             return {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        elif model_name == "CatBoost":
            return {
                'iterations': [50, 100],
                'depth': [3, 6],
                'l2_leaf_reg': [1, 3]
            }
        elif model_name == "LightGBM":
             return {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        elif model_name == "SVM":
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        # Add more classification models here

    elif problem_type == "regression":
        if model_name == "Random Forest":
            return {
                'n_estimators': [50, 100, 150],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif model_name == "Linear Regression":
            return {} # No tuneable params for basic Linear Regression
        elif model_name == "Ridge Regression":
            return {'alpha': [0.1, 1.0, 10.0]}
        elif model_name == "Lasso Regression":
            return {'alpha': [0.1, 1.0, 10.0]}
        elif model_name == "XGBoost":
            return {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        elif model_name == "CatBoost":
            return {
                'iterations': [50, 100],
                'depth': [3, 6],
                'l2_leaf_reg': [1, 3]
            }
        elif model_name == "LightGBM":
             return {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        elif model_name == "SVM":
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        # Add more regression models here

    return {} # Return empty for models without defined grids

def optimize_hyperparameters(
    df,
    target_col,
    model_name,
    problem_type,
    tuning_method="Optuna", # New argument
    n_trials=50, # For Optuna
    n_iter_random=10, # For RandomizedSearchCV
    cv_folds=5 # For GridSearchCV and RandomizedSearchCV
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X)
    X = sanitize_col_names(X) # Use imported function

    # Handle cases where target column might be categorical for classification (e.g., Label Encoding if needed)
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if problem_type == "classification" and len(pd.unique(y)) > 1 else None # Use pd.unique(y) for consistency
    )

    best_model = None
    best_params = {}
    final_metric_value = None
    metric_name = ""

    if problem_type == "classification":
        metric_name = "Accuracy"
        scoring = 'accuracy'
    else:
        metric_name = "R2 Score"
        scoring = 'r2'

    if tuning_method == "Optuna":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_optuna(trial, X_train, X_test, y_train, y_test, model_name, problem_type), n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        final_metric_value = best_trial.value
        best_model = _get_model_instance(model_name, problem_type, best_params)
        best_model.fit(X_train, y_train)

    elif tuning_method in ["GridSearchCV", "RandomizedSearchCV"]:
        estimator = _get_model_instance(model_name, problem_type)
        param_grid = _get_param_grid(model_name, problem_type)

        if not param_grid:
            print(f"Warning: No parameter grid defined for {model_name}. Skipping tuning.")
            return None, None, {}, None

        if tuning_method == "GridSearchCV":
            tuner = GridSearchCV(estimator, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
        else: # RandomizedSearchCV
            tuner = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter_random, cv=cv_folds, scoring=scoring, n_jobs=-1, random_state=42)

        tuner.fit(X_train, y_train)
        best_model = tuner.best_estimator_
        best_params = tuner.best_params_

        predictions = best_model.predict(X_test)
        if problem_type == "classification":
            final_metric_value = accuracy_score(y_test, predictions)
        else:
            final_metric_value = r2_score(y_test, predictions)

    else:
        raise ValueError(f"Unknown tuning method: {tuning_method}")

    return final_metric_value, metric_name, best_params, best_model
