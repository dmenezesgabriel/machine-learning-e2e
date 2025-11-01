import os
import sys
from dataclasses import dataclass
from typing import Optional

from catboost import CatBoostRegressor  # type: ignore
from sklearn.ensemble import AdaBoostRegressor  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, train_array, test_array, preprocessor_path: Optional[str] = None
    ):
        try:
            logger.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "random-forest": RandomForestRegressor(),
                "decision-tree": DecisionTreeRegressor(),
                "gradient-boosting": GradientBoostingRegressor(),
                "linear-regression": LinearRegression(),
                "k-neighbors-classifier": KNeighborsRegressor(),
                "xgb-classifier": XGBRegressor(),
                "cat-boost-classifier": CatBoostRegressor(),
                "ada-boost-classifier": AdaBoostRegressor(),
            }
            params = {
                "decision-tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "random-forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "gradient-boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "linear-regression": {},
                "k-neighbors-classifier": {},
                "xgb-classifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "cat-boost-classifier": {
                    # "depth": [6, 8, 10],
                    # "learning_rate": [0.01, 0.05, 0.1],
                    # "iterations": [30, 50, 100],
                },
                "ada-boost-classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(Exception("No best model found"), sys)

            logger.info(
                f"Best model on both training and testing dataset: {best_model_name}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as error:
            CustomException(error, sys)
