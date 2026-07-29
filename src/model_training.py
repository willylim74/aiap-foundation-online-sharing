# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor



class ModelTraining:
    """
    A class used to train and evaluate machine learning models on HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.
        """

        logging.info("Starting data splitting.")

        X = df.drop(columns=self.config["target_column"])

        # log transform target
        y = np.log1p(df[self.config["target_column"]]) 

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.config["random_state"]
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.config["random_state"]
        )
        logging.info("Data split into train, validation, test sets.")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.\n\n")

        random_state = self.config["random_state"]

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "XGBoost": XGBRegressor(random_state=random_state),
        }

        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor), 
                ("model", model)
                ]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, 
                X_val, 
                y_val, 
                model_name
            )

        return pipelines, metrics


    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        # Predict (log scale)
        y_pred_log = model.predict(X_val)

        # Convert predictions back to original scale
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val)

        # Metrics (orignal scale and log scale)
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAE_log": mean_absolute_error(y_val, y_pred_log),
            "RMSE_log": np.sqrt(mean_squared_error(y_val, y_pred_log)),
            "R2_log": r2_score(y_val, y_pred_log)

        }


        logging.info(f"=== Validation Metrics: {model_name} ===")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.5f}")

        logging.info(f"\n")

        return metrics



    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for baseline models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")

        tuned_models = {}
        tuned_metrics = {}

        all_param_grids = self.config["hyperparameters"]
        search_results = {}

        cv = self.config["cv"]
        n_iter = self.config["n_iter"]
        scoring = self.config["scoring"]
        random_state = self.config["random_state"]

        models = {
                "Random Forest": RandomForestRegressor(random_state=random_state),
                "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
                "XGBoost": XGBRegressor(random_state=random_state)
            }

        for model_name, model in models.items():
            logging.info(f"Tuning model: {model_name}")
            
            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor), 
                ("model", model)
                ]
            )
            

            param_grid = all_param_grids.get(model_name, {})

            if not param_grid:
                logging.warning(f"No param grid found for {model_name}, skipping.")
                continue

            random_search = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=scoring,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=1
            )

            random_search.fit(X_train, y_train)

            tuned_models[model_name] = random_search.best_estimator_

            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], 
                X_val, 
                y_val, 
                model_name + " (tuned)"
            )

            search_results[model_name] = random_search

            logging.info(f"Best params for {model_name}: {random_search.best_params_}")
            logging.info(f"Best CV R2(log-shares) for {model_name}: {random_search.best_score_:.5f}\n\n")

        logging.info("Hyperparameter tuning completed.")

        
    
        summary = []

        for name, search in search_results.items():
            summary.append({
                "Model": name,
                "CV R2(log-shares)": search.best_score_
            })

        results_df = pd.DataFrame(summary).sort_values(by="CV R2(log-shares)", ascending=False)
        logging.info("\n%s\n\n", results_df)

        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """

        # Predict (log scale)
        y_pred_log = model.predict(X_test)

        # Convert to original scale
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)

        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAE_log": mean_absolute_error(y_test, y_pred_log),
            "RMSE_log": np.sqrt(mean_squared_error(y_test, y_pred_log)),
            "R2_log": r2_score(y_test, y_pred_log),
        }
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    

