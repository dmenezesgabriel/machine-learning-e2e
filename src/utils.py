import os
import sys

import dill  # type: ignore
import numpy as np
import pandas as pd

from src.exception import CustomException


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_object:
            dill.dump(object, file_object)

    except Exception as error:
        raise CustomException(error, sys)
