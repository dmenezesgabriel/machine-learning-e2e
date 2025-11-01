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

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
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
