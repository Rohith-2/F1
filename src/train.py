"""
F1 Race Performance Prediction Model Training Script.

This module implements a machine learning pipeline for predicting Formula 1 race performance
using various regression models. It includes hyperparameter optimization using Optuna and
supports multiple model types including Ridge, Lasso, RandomForest, and Stacking ensembles.

The script handles:
- Data loading and preprocessing
- Model selection and hyperparameter tuning
- Performance evaluation and model comparison
- Model persistence for later use

Python 3.14 Compatible
Requires: numpy, pandas, scikit-learn, xgboost, optuna

Author: Rohith
Date: October 2025
"""

from __future__ import annotations  # Python 3.14 type hint support

# Standard library imports
import argparse
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import joblib
from joblib import Parallel, delayed, Memory

# Initialize memory cache for computation results
CACHE_DIR = Path('.cache/joblib')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

# Machine learning imports
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load and preprocess the F1 race data from cached NumPy files.
    
    Returns:
        tuple: Contains:
            - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            - y_lap (np.ndarray): Target values for lap times
            - y_pos (np.ndarray): Target values for race positions
            - driver_numbers (Dict[str, int]): Mapping of driver codes to their car numbers
    
    Note:
        Missing values in X are filled with -1 before conversion to numpy array.
    """
    X = pd.DataFrame(np.load('.cache/hist_data/X_train.npy', allow_pickle=True)).fillna(-1).to_numpy()
    y_lap = np.load('.cache/hist_data/y_train_lap.npy', allow_pickle=True)
    y_pos = np.load('.cache/hist_data/y_train_pos.npy', allow_pickle=True)

    driver_numbers = {
        "HAM": 44,  # Lewis Hamilton
        "RUS": 63,  # George Russell
        "LEC": 16,  # Charles Leclerc
        "PIA": 81,  # Oscar Piastri
        "NOR": 4,   # Lando Norris
        "VER": 1,   # Max Verstappen
        "SAI": 55,  # Carlos Sainz
        "ALB": 23,  # Alexander Albon
        "HUL": 27,  # Nico Hulkenberg
        "ALO": 14,  # Fernando Alonso
        "TSU": 22,  # Yuki Tsunoda
        "GAS": 10,  # Pierre Gasly
        "STR": 18,  # Lance Stroll
        "OCO": 31,  # Esteban Ocon
        "COL": 43,  # Franco Colapinto
        "ANT": 12,  # Andrea Kimi Antonelli
        "BEA": 87,  # Oliver Bearman
        "LAW": 30,  # Liam Lawson
        "DOO": 7,   # Jack Doohan
        "HAD": 6,   # Isack Hadjar
        "BOR": 5    # Gabriel Bortoleto
        }
    return X, y_lap, y_pos, driver_numbers

X, y_lap, y_pos, driver_numbers = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y_lap, random_state=94)
# -----------------------------------
# Objective function for Optuna
# -----------------------------------
#@memory.cache(ignore=['trial'])  # Cache results but ignore trial object
def objective(
    trial: optuna.trial.Trial,
    model_name: str = "RandomForest"
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    
    This function defines the search space for hyperparameters and evaluates model performance
    using cross-validation. It supports multiple model types and scaling options.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        model_name: Name of the model to optimize (default: "RandomForest")
    
    Returns:
        float: Mean cross-validation negative MAE score
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    # Normalization hyperparameter
    normalize = trial.suggest_categorical("normalize", [True, False])
    scaler_choice = trial.suggest_categorical("scaler", ["standard", "minmax"])

    if normalize:
        try:
            scaler = StandardScaler() if scaler_choice == "standard" else MinMaxScaler()
        except Exception as e:
            raise ValueError(f"Invalid scaler choice: {str(e)}")

    # per-model search spaces
    if model_name == "Ridge":
        try:
            alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            model = Ridge(alpha=alpha, solver=solver)
        except ValueError as e:
            raise ValueError(f"Invalid Ridge parameters: {str(e)}")

    elif model_name == "Lasso":
        try:
            alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 3000)
            tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
            selection = trial.suggest_categorical("selection", ["cyclic", "random"])
            model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, selection=selection)
        except ValueError as e:
            raise ValueError(f"Invalid Lasso parameters: {str(e)}")

    elif model_name == "ElasticNet":
        alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 2000),
            max_depth=trial.suggest_int("max_depth", 3, 100),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),  # Updated: None -> 1.0
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs=-1,  # Explicit parallel processing
            random_state=42
        )

    elif model_name == "ExtraTrees":
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 2000),
            max_depth=trial.suggest_int("max_depth", 3, 100),
            criterion=trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),  # Added friedman_mse
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),  # Added parameter
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),    # Added parameter
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),  # Added parameter
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs=-1,  # Explicit parallel processing
            random_state=42
        )

    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            criterion=trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),  # Updated criterion options
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),  # Added parameter
            random_state=42,
            validation_fraction=0.1  # Added validation fraction for early stopping
        )

    elif model_name == "SVR":
        model = SVR(
            C=trial.suggest_float("C", 1e-2, 1e3, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),  # scale is now the default
            degree=trial.suggest_int("degree", 2, 15),
            coef0=trial.suggest_float("coef0", 0.0, 10.0),  # Added parameter for non-linear kernels
            tol=trial.suggest_float("tol", 1e-5, 1e-3, log=True),  # Added tolerance parameter
            cache_size=500  # Added cache size for faster computation
        )

    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 3000),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.1, 1.0),
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "Stacking":
        alpha = trial.suggest_float("alpha_ridge", 1e-3, 1e2, log=True)
        solver = trial.suggest_categorical("solver_ridge", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        final_estimator = Ridge(alpha=alpha, solver=solver)

        # base models used for stacking (their params sampled here)
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators_rf", 100, 2000),
                max_depth=trial.suggest_int("max_depth_rf", 3, 100),
                min_samples_split=trial.suggest_int("min_samples_split_rf", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf_rf", 1, 10),
                max_features=trial.suggest_categorical("max_features_rf", ["sqrt", "log2", None]),
                bootstrap=trial.suggest_categorical("bootstrap_rf", [True, False]),
                random_state=42
            )),
            ('ext_reg', ExtraTreesRegressor(
                n_estimators=trial.suggest_int("n_estimators_ext", 100, 2000),
                max_depth=trial.suggest_int("max_depth_ext", 3, 100),
                criterion=trial.suggest_categorical("criterion_ext", ["squared_error", "absolute_error"]),
                bootstrap=trial.suggest_categorical("bootstrap_ext", [True, False]),
                random_state=42
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators_xgb", 100, 3000),
                learning_rate=trial.suggest_float("learning_rate_xgb", 1e-4, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth_xgb", 3, 100),
                subsample=trial.suggest_float("subsample_xgb", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree_xgb", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha_xgb", 1e-6, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda_xgb", 1e-6, 10.0, log=True),
                gamma=trial.suggest_float("gamma_xgb", 0.1, 1.0),
                n_jobs=-1,
                random_state=42
            ))
        ]
        model = StackingRegressor(estimators=base_models, final_estimator=final_estimator, n_jobs=-1)

    else:
        raise ValueError("Unsupported model")

    # Create pipeline
    if normalize:
        pipe = Pipeline([("scaler", scaler), ("model", model)])
    else:
        pipe = Pipeline([("model", model)])

    # Cross-validation MAE
    scores = cross_val_score(
        pipe, 
        X_train, 
        y_train, 
        cv=5, 
        scoring="neg_mean_absolute_error",
        error_score='raise'  # Raise error instead of returning NaN
    )
    score = float(scores.mean())  # Ensure we return a float
    return score


# -----------------------------------
# Optimization 
# -----------------------------------

def tune_model(
    model_name: str,
    n_trials: int = 30,
    n_jobs: int = -1,
    name: str = 'init'
) -> Dict[str, Any]:
    """
    Tune hyperparameters for a specific model using parallel Optuna optimization.
    
    This function performs parallel hyperparameter optimization using Optuna,
    fits the model with the best parameters, and evaluates its performance.
    
    Args:
        model_name: Name of the model to tune
        n_trials: Number of optimization trials to perform (default: 30)
        n_jobs: Number of parallel jobs for optimization (default: -1, uses all cores)
    
    Returns:
        dict: Contains model performance metrics and the fitted model:
            - model: Name of the model
            - best_params: Best hyperparameters found
            - r2: R-squared score
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - model_object: Fitted Pipeline object
            
    Raises:
        ValueError: If an unsupported model type is specified
    """

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),  # More efficient sampling
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=3
        ),
        study_name=model_name+f"_{name}",
    )

    study.optimize(
        lambda trial: objective(trial, model_name),
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=True, # Catch all exceptions to prevent study from failing
    )

    best_params = study.best_params  # Create a copy to avoid modifying original
    print(f"🔍 Best params for {model_name}: {best_params}")

    # extract normalization choices without removing from best_params
    normalize = best_params.pop("normalize", False)
    scaler_choice = best_params.pop("scaler", "standard")


    # Build the chosen model from model_params
    if model_name == "Ridge":
        best_model = Ridge(**best_params)
    elif model_name == "Lasso":
        best_model = Lasso(**best_params)
    elif model_name == "ElasticNet":
        best_model = ElasticNet(**best_params)
    elif model_name == "RandomForest":
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_name == "GradientBoosting":
        best_model = GradientBoostingRegressor(**best_params, random_state=42)
    elif model_name == "KNN":
        best_model = KNeighborsRegressor(**best_params)
    elif model_name == "DecisionTree":
        best_model = DecisionTreeRegressor(**best_params, random_state=42)
    elif model_name == "ExtraTrees":
        best_model = ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)

    elif model_name == "Stacking":
        # Reconstruct base estimators' params from the model parameters
        # Extract parameters for each model without removing them from the original
        rf_params = {k.replace('_rf',''): best_params[k] for k in list(best_params.keys()) if k.endswith('_rf')}
        ext_params = {k.replace('_ext',''): best_params[k] for k in list(best_params.keys()) if k.endswith('_ext')}
        xgb_params = {k.replace('_xgb',''): best_params[k] for k in list(best_params.keys()) if k.endswith('_xgb')}
        ridge_params = {k.replace('_ridge',''): best_params[k] for k in list(best_params.keys()) if k.endswith('_ridge')}

        # Remove the extracted parameters from model_params to avoid duplication
        for k in list(best_params.keys()):
            if any(k.endswith(suffix) for suffix in ['_rf', '_ext', '_xgb', '_ridge']):
                del best_params[k]

        estimators = [
            ('rf', RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
            ('ext', ExtraTreesRegressor(**ext_params, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1))
        ]

        final_estimator = Ridge(**ridge_params)

        best_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)

    else:
        raise ValueError("Unsupported model")

    # Create pipeline with scaler if requested
    try:
        if normalize:
            scaler = StandardScaler() if scaler_choice == "standard" else MinMaxScaler()
            pipe = Pipeline([("scaler", scaler), ("model", best_model)])
        else:
            pipe = Pipeline([("model", best_model)])
    except Exception as e:
        raise RuntimeError(f"Failed to create pipeline for {model_name}: {str(e)}")

    # Fit and predict
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Failed to fit {model_name} model: {str(e)}")

    try:
        preds = pipe.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Failed to generate predictions with {model_name} model: {str(e)}")

    try:
        metrics = {
            "model": model_name,
            "best_params": study.best_params,  # Use original best parameters
            "r2": r2_score(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
            'model_object': pipe,
        }
        return metrics
    except Exception as e:
        raise RuntimeError(f"Failed to compute metrics for {model_name} model: {str(e)}")
    
def main() -> None:
    """
    Main entry point for the F1 race prediction model training script.
    
    Handles command line arguments for model tuning and training:
    - With --tune flag: Performs hyperparameter optimization for all models
    - Without --tune flag: Loads and uses the best model from previous tuning
    
    Command line arguments:
        --tune: Boolean flag to enable hyperparameter tuning
        --init_trials: Number of initial trials for all the model tuning
        --final_trials: Number of final trials for best performing model tuning

    Outputs:
        - Saves model leaderboard to '.model/model_leaderboard.pkl'
        - Saves best model to './model/model.joblib'
        
    Raises:
        FileNotFoundError: If model files are not found when loading
        RuntimeError: If model training or saving fails
        ValueError: If invalid parameters are provided
    """
    try:
        parser = argparse.ArgumentParser(description="Train the F1 Race Prediction ML Model")
        parser.add_argument('--tune', action='store_true', help='Re-tune hyperparameters for all models')
        parser.add_argument('--init_trials', type=int, help='Initial number of trials for all the model tuning')
        parser.add_argument('--final_trials', type=int, help='Final number of trials for best performing model tuning')
        args = parser.parse_args()
    except Exception as e:
        raise RuntimeError(f"Failed to parse command line arguments: {str(e)}")

    if args.tune:
        models = [
            "Ridge", "Lasso", "ElasticNet", "RandomForest", 
            "ExtraTrees", "XGBoost", "Stacking",
        ]
        n_trials = args.init_trials if args.init_trials is not None else 15
        n_jobs_per_model = max(1, (os.cpu_count() or 1) // len(models))  # Distribute CPUs across models

        # Configure parallel backend for better performance
        with Parallel(
            n_jobs=len(models),  # One job per model
            backend='loky',  # More robust backend
            verbose=1,
            prefer="processes",  # Use processes for better parallelism
            max_nbytes=None,  # No limit on memory per job
            temp_folder='/tmp'  # Use temp directory for memory management
        ) as parallel:
            results = parallel(
                delayed(tune_model)(
                    model, 
                    n_trials=n_trials,
                    n_jobs=n_jobs_per_model  # Pass CPU allocation to each model
                ) for model in models
            )

        # Process results and save
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="r2", ascending=False)
        
        # Save with compression for better I/O performance
        df_sorted.to_pickle('./model/model_leaderboard.pkl', compression='gzip')
        
        print("\n🏆 Model Leaderboard (Sorted by Best Value):")
        print(df_sorted[["model", "r2", "rmse", 'mae']])

        # Final tuning of best model with more resources
        final_n_trials = args.final_trials if args.final_trials is not None else 150
        best_model_name = df_sorted['model'].values[0]
        print(f"\n🔄 Fine-tuning best model: {best_model_name}")
        
        m = tune_model(
            best_model_name, 
            n_trials=final_n_trials,
            n_jobs=-1,  # Use all CPUs for final tuning
            name='final'
        )

        best_model = m['model_object']
    else:
        best_model = pd.read_pickle('./model/model_leaderboard.pkl').iloc[0]['model_object']

    best_model.fit(X,y_lap)

    joblib.dump(best_model, './model/model.joblib')

if __name__ == "__main__":
    main()