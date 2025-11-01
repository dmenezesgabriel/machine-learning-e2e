import os
import sys

import dill  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score  # type: ignore
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_object:
            dill.dump(object, file_object)

    except Exception as error:
        logger.exception(error)
        raise CustomException(error, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for index in range(len(list(models))):
            model = list(models.values())[index]
            parameters = params[list(models.keys())[index]]

            gs = GridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_prediction = model.predict(X_train)
            y_test_prediction = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_prediction)
            test_model_score = r2_score(y_test, y_test_prediction)

            report[list(models.keys())[index]] = test_model_score

        return report

    except Exception as error:
        logger.exception(error)
        raise CustomException(error, sys)
